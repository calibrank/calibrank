"""Calibration text sampled from the dense model itself.

Sequences start from the model's document-start token (BOS, or EOS for models
without BOS, which is how they see document boundaries in packed pretraining
data) and are extended by ancestral sampling at temperature T, with no top-k or
top-p truncation. Generation does not stop at EOS: the model starts a new
document, as it would in a packed stream.

At T = 1 the windows are exact samples from the dense model, so the token-level
KL averaged over them is an unbiased estimate of the sequence-level divergence
KL(P_dense || P_compressed) (see the manuscript, Section 2).
"""

from __future__ import annotations

import math
import time

import torch


def start_token(tokenizer) -> int:
    tok = tokenizer.bos_token_id
    return tok if tok is not None else tokenizer.eos_token_id


@torch.no_grad()
def sample(model, tokenizer, n: int, seqlen: int, temperature: float = 1.0,
           batch_size: int = 32, seed: int = 0, device="cuda", log=print) -> torch.Tensor:
    """(n, seqlen) token ids, the first column the start token."""
    gen = torch.Generator(device=device).manual_seed(seed)
    start = start_token(tokenizer)
    vocab = model.get_output_embeddings().weight.shape[0]
    n_tok = len(tokenizer)          # ids >= len(tokenizer) are padding rows of the head
    out, t0 = [], time.time()
    for b in range(math.ceil(n / batch_size)):
        B = min(batch_size, n - b * batch_size)
        cur = torch.full((B, 1), start, dtype=torch.long, device=device)
        seq = [cur]
        past = None
        for _ in range(seqlen - 1):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[:, -1, :].float()
            if n_tok < vocab:
                logits[:, n_tok:] = float("-inf")
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, -1)
            cur = torch.multinomial(probs, 1, generator=gen)
            seq.append(cur)
        out.append(torch.cat(seq, 1).cpu())
        del past
        log(f"  selfgen batch {b + 1}/{math.ceil(n / batch_size)} ({time.time() - t0:.0f}s)")
    return torch.cat(out, 0)


@torch.no_grad()
def sample_documents(model, tokenizer, n_tokens: int, max_len: int = 512,
                     temperature: float = 1.0, batch_size: int = 256, seed: int = 0,
                     device="cuda", log=print) -> list[list[int]]:
    """Independent documents, each sampled from a fresh start token.

    A document ends at the model's first EOS or after ``max_len`` tokens, and is
    returned without the start token and without the EOS. Sampling stops once
    the documents hold ``n_tokens`` tokens counting one separator each. Long
    exact samples drift into high-entropy text (the per-token entropy of
    Llama-3.2-1B rises from about 3 to 6 nats over 2048 tokens); restarting
    every document bounds that drift while keeping each document an exact
    sample of the model's distribution truncated at ``max_len``.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    start, eos = start_token(tokenizer), tokenizer.eos_token_id
    vocab = model.get_output_embeddings().weight.shape[0]
    n_tok = len(tokenizer)
    docs, total, t0, b = [], 0, time.time(), 0
    while total < n_tokens:
        cur = torch.full((batch_size, 1), start, dtype=torch.long, device=device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        seq, past = [], None
        for _ in range(max_len):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[:, -1, :].float()
            if n_tok < vocab:
                logits[:, n_tok:] = float("-inf")
            if temperature != 1.0:
                logits = logits / temperature
            cur = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)
            seq.append(cur)
            done |= cur[:, 0] == eos
            if bool(done.all()):
                break
        ids = torch.cat(seq, 1).cpu().tolist()
        del past
        for row in ids:
            if eos in row:
                row = row[: row.index(eos)]
            if row:
                docs.append(row)
                total += len(row) + 1
        b += 1
        log(f"  selfdoc batch {b}: {len(docs)} documents, {total} tokens ({time.time() - t0:.0f}s)")
    return docs


@torch.no_grad()
def sample_responses(model, tokenizer, users: list[str], n_tokens: int, max_len: int = 512,
                     temperature: float = 1.0, batch_size: int = 128, seed: int = 0,
                     device="cuda", log=print) -> list[list[int]]:
    """The instruct model's own answers to ``users`` prompts, as chat documents.

    Each document is the chat-formatted user turn with the assistant header
    (without the leading BOS) followed by the sampled response and the
    end-of-turn token: the same layout as a chat-formatted corpus example, with
    the response drawn from the model at ``temperature``. Prompts are consumed in
    order until the documents hold ``n_tokens`` tokens.
    """
    torch.manual_seed(seed)
    eot = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eot
    docs, total, i, t0 = [], 0, 0, time.time()
    while total < n_tokens and i < len(users):
        batch = users[i:i + batch_size]
        i += len(batch)
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": u}],
                                                 add_generation_prompt=True, tokenize=True)
                   for u in batch]
        L = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), L), pad, dtype=torch.long)
        att = torch.zeros((len(prompts), L), dtype=torch.long)
        for r, p in enumerate(prompts):          # left padding
            ids[r, L - len(p):] = torch.tensor(p)
            att[r, L - len(p):] = 1
        out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                             do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                             max_new_tokens=max_len, eos_token_id=eot, pad_token_id=pad)
        for r, p in enumerate(prompts):
            resp = out[r, L:].tolist()
            if eot in resp:
                resp = resp[: resp.index(eot)]
            resp = [t for t in resp if t != pad]
            prompt = p[1:] if bos is not None and p and p[0] == bos else p
            doc = prompt + resp + [eot]
            docs.append(doc)
            total += len(doc) + 1
        log(f"  selfresp {i} prompts: {len(docs)} documents, {total} tokens ({time.time() - t0:.0f}s)")
    return docs
