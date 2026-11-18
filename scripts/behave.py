#!/usr/bin/env python3
"""Behavioral evaluation of a finished run (or of the dense model): what the
compressed model does when it generates, beyond perplexity and multiple choice.

    uv run --no-sync python scripts/behave.py --run-dir out/extended/runs/llama-3.2-1b/proj-c4-r30-s0 \
        [--long-context] [--gen-tasks gsm8k,nq_open]

The model is rebuilt from the run's config.json, alloc.json and best_params.pt
(learned projectors, merged into factors); a run named ``dense`` is the
uncompressed model. Writes ``<run-dir>/behave.json``:

  open_gen   continuations of 200 held-out C4 prompts (32 tokens, 128 new tokens),
             sampled at T = 1 and greedy: the dense model's NLL of the continuation
             (how plausible the text is to the model being replaced), the model's
             own NLL, distinct-n and the rate of repeated 4-grams
  long_ctx   perplexity on 8,192-token windows of 16 PG-19 test books, per position
             bucket (calibration windows hold 2,048 tokens), and passkey retrieval
             accuracy at contexts of about 2k, 4k and 8k tokens
  tasks      lm-eval generation tasks at their default few-shot settings
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsc import arch, data, evaluate, project  # noqa: E402
from fsc.utils import env_info, log, read_json, write_json  # noqa: E402

import compress  # noqa: E402

N_PROMPTS, PROMPT_LEN, NEW_TOKENS = 200, 32, 128
LONG_LEN, LONG_BOOKS = 8192, 16
PASSKEY_LENGTHS, PASSKEY_N = (2000, 4000, 8000), 40


def rebuild(run_dir: str, device, dtype):
    """(model, tokenizer, parts or None, config): parts merge the trained projectors."""
    cfg = read_json(os.path.join(run_dir, "config.json"))
    model, tok = compress.load_model(cfg["model"], dtype, device)
    if cfg.get("mode") == "dense":
        return model, tok, None, cfg
    if cfg.get("mode") != "proj":
        raise SystemExit(f"behave.py rebuilds learned projectors; {run_dir} has mode {cfg.get('mode')}")
    units = arch.discover_units(model, tie=not cfg.get("untied", False),
                                square_side=cfg.get("square_side", "in"))
    Vs = torch.load(os.path.join(run_dir, "best_params.pt"), map_location="cpu")
    bank = project.ProjectorBank(units, Vs).to(device)
    bases = {k: v.to(device) for k, v in bank.export().items()}
    parts = compress.merge_parts(model, units, bases, dtype)
    del bank, bases
    return model, tok, parts, cfg


def compressed(model, parts, dtype):
    return project.swapped(model, parts, dtype) if parts is not None else _null()


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def prompts_c4(tok, n: int, length: int) -> torch.Tensor:
    """BOS + the first length - 1 tokens of the last n C4 validation documents with enough
    text (the perplexity and KL sets use the first ones)."""
    docs = data._read_jsonl_gz(data.raw_file("c4", "validation"))
    bos = tok.bos_token_id
    body = length - 1 if bos is not None else length
    rows = []
    for d in reversed(docs):
        ids = tok(d, add_special_tokens=False)["input_ids"]
        if len(ids) >= body + NEW_TOKENS:
            rows.append(([bos] if bos is not None else []) + ids[:body])
        if len(rows) == n:
            break
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def generate(model, prompts: torch.Tensor, new: int, sample: bool, seed: int, device,
             batch: int = 50) -> torch.Tensor:
    """Continuations (n, new) by ancestral sampling at T = 1 or greedy decoding."""
    out = []
    g = torch.Generator(device=device).manual_seed(seed)
    for i in range(0, prompts.shape[0], batch):
        x = prompts[i:i + batch].to(device)
        past, cur, seq = None, x, []
        for _ in range(new):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[:, -1].float()
            if sample:
                cur = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
            else:
                cur = logits.argmax(-1, keepdim=True)
            seq.append(cur)
        out.append(torch.cat(seq, 1).cpu())
        del past
    return torch.cat(out)


@torch.no_grad()
def continuation_nll(model, prompts, conts, device, batch: int = 25) -> list[float]:
    """Per-sequence mean NLL of the continuation tokens given the prompt."""
    res = []
    P = prompts.shape[1]
    for i in range(0, prompts.shape[0], batch):
        x = torch.cat([prompts[i:i + batch], conts[i:i + batch]], 1).to(device)
        lp = model(input_ids=x, use_cache=False).logits[:, P - 1:-1].float().log_softmax(-1)
        nll = -lp.gather(-1, x[:, P:, None])[..., 0]
        res += nll.mean(1).tolist()
    return res


def text_stats(conts: torch.Tensor) -> dict:
    """distinct-n over all continuations, and the mean rate of repeated 4-grams."""
    out = {}
    rows = conts.tolist()
    for n in (1, 2, 3):
        grams = [tuple(r[j:j + n]) for r in rows for j in range(len(r) - n + 1)]
        out[f"distinct{n}"] = len(set(grams)) / max(len(grams), 1)
    rep = []
    for r in rows:
        grams = [tuple(r[j:j + 4]) for j in range(len(r) - 3)]
        rep.append(1.0 - len(set(grams)) / max(len(grams), 1))
    out["rep4"] = sum(rep) / len(rep)
    return out


def open_generation(model, tok, parts, dtype, device) -> dict:
    prompts = prompts_c4(tok, N_PROMPTS, PROMPT_LEN)
    res = {"n_prompts": prompts.shape[0], "prompt_len": PROMPT_LEN, "new_tokens": NEW_TOKENS}
    for mode, sample in (("sample", True), ("greedy", False)):
        n = prompts.shape[0] if sample else 100
        p = prompts[:n]
        with compressed(model, parts, dtype):
            conts = generate(model, p, NEW_TOKENS, sample, 0, device)
            own = continuation_nll(model, p, conts, device)
        teacher = continuation_nll(model, p, conts, device)      # dense model
        r = {"teacher_nll": sum(teacher) / len(teacher), "own_nll": sum(own) / len(own),
             "teacher_nll_items": teacher}
        r.update(text_stats(conts))
        r["examples"] = [tok.decode(torch.cat([p[j], conts[j]]).tolist()) for j in range(3)]
        res[mode] = r
        log(f"    {mode}: teacher NLL {r['teacher_nll']:.3f} own {r['own_nll']:.3f} "
            f"rep4 {r['rep4']:.3f} distinct2 {r['distinct2']:.3f}")
    return res


@torch.no_grad()
def long_context(model, tok, parts, dtype, device) -> dict:
    """Perplexity per position bucket on 8,192-token PG-19 windows."""
    w = data.book_windows(tok, LONG_BOOKS, LONG_LEN, seed=4321)
    edges = [0, 2048, 4096, LONG_LEN - 1]
    sums = [0.0] * (len(edges) - 1)
    cnts = [0] * (len(edges) - 1)
    per_book = []
    with compressed(model, parts, dtype):
        for i in range(w.shape[0]):
            x = w[i:i + 1].to(device)
            logits = model(input_ids=x, use_cache=False).logits[0, :-1]
            nll = torch.empty(logits.shape[0], device=device)
            for j in range(0, logits.shape[0], 1024):
                lp = logits[j:j + 1024].float().log_softmax(-1)
                nll[j:j + 1024] = -lp.gather(-1, x[0, j + 1:j + 1025, None])[:, 0]
            per_book.append(float(nll.mean()))
            for b in range(len(edges) - 1):
                seg = nll[edges[b]:edges[b + 1]]
                sums[b] += float(seg.sum())
                cnts[b] += seg.numel()
            del logits
    buckets = {f"{edges[b]}-{edges[b + 1] + 1}": math.exp(sums[b] / cnts[b]) for b in range(len(sums))}
    log("    long-context ppl " + " ".join(f"{k}={v:.2f}" for k, v in buckets.items()))
    return {"ppl_by_position": buckets, "n_books": int(w.shape[0]), "nll_per_book": per_book}


FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
          "There and back again. ")


@torch.no_grad()
def passkey(model, tok, parts, dtype, device) -> dict:
    """Retrieval of a 5-digit key hidden at a random depth of filler text."""
    rng = random.Random(1234)
    res = {}
    head = ("There is an important info hidden inside a lot of irrelevant text. Find it and "
            "memorize it. I will quiz you about the important information there.\n\n")
    tail = "\n\nWhat is the pass key? The pass key is"
    n_fill = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    with compressed(model, parts, dtype):
        for L in PASSKEY_LENGTHS:
            hits = []
            reps = max(1, (L - 60) // n_fill)
            for _ in range(PASSKEY_N):
                key = rng.randint(10000, 99999)
                depth = rng.randint(0, reps)
                needle = f"The pass key is {key}. Remember it. {key} is the pass key. "
                text = head + FILLER * depth + needle + FILLER * (reps - depth) + tail
                ids = tok(text, return_tensors="pt").input_ids.to(device)
                out = generate(model, ids.cpu(), 8, False, 0, device, batch=1)
                hits.append(str(key) in tok.decode(out[0].tolist()))
            res[str(L)] = sum(hits) / len(hits)
            log(f"    passkey ~{L} tokens: {res[str(L)]:.2f}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--long-context", action="store_true")
    ap.add_argument("--gen-tasks", default="")
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--task-limit", type=int, default=None, help="items per task (smoke tests)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args(argv)

    out_path = os.path.join(args.run_dir, "behave.json")
    # resume: a finished file, else the parts saved before an interruption
    res = read_json(out_path, None) or read_json(out_path + ".partial", {})
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = "cuda"
    torch.manual_seed(0)
    model, tok, parts, cfg = rebuild(args.run_dir, device, dtype)
    res.setdefault("env", []).append(env_info())
    t0 = time.time()
    if "open_gen" not in res:
        log("open-ended generation")
        res["open_gen"] = open_generation(model, tok, parts, dtype, device)
        write_json(out_path + ".partial", res)
    if args.long_context and "long_ctx" not in res:
        log("long context")
        res["long_ctx"] = long_context(model, tok, parts, dtype, device)
        res["passkey"] = passkey(model, tok, parts, dtype, device)
        write_json(out_path + ".partial", res)
    tasks = [t for t in args.gen_tasks.split(",") if t]
    if tasks and "tasks" not in res:
        log(f"generation tasks {tasks}")
        # The harness caps generations at max_gen_toks (256) through max_length; a
        # max_new_tokens in the model's own generation config (Qwen3: 2048) would
        # override that cap, so every model is held to the harness's setting.
        model.generation_config.max_new_tokens = None
        items = {}
        with compressed(model, parts, dtype):
            res["tasks"] = evaluate.zero_shot(model, tok, tasks, args.gen_batch,
                                              limit=args.task_limit, items=items, num_fewshot=None)
        res["task_items"] = {t: {k: v.tolist() for k, v in d.items()} for t, d in items.items()}
        log("    " + str({t: {k: v for k, v in r.items() if "stderr" not in k}
                          for t, r in res["tasks"].items()}))
    res["seconds"] = res.get("seconds", 0) + time.time() - t0
    write_json(out_path, res)
    if os.path.exists(out_path + ".partial"):
        os.remove(out_path + ".partial")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
