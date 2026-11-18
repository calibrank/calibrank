"""Calibration sources and evaluation corpora.

Every corpus is read from a raw file of a Hub dataset repository, at the
commit pinned in ``fsc.hub``, via ``hf_hub_download``. Once
``scripts/prefetch.py`` has cached them, jobs run with HF_HUB_OFFLINE=1.

Calibration windows are built the same way for every source: documents are
sampled without replacement (seeded), tokenized without special tokens, packed
into one stream separated by EOS, and cut into windows of ``seqlen - 1`` tokens
preceded by BOS (``seqlen`` tokens, no BOS, for models without one). Training
and validation windows come from disjoint documents of the same source.
"""

from __future__ import annotations

import gzip
import json
import random

import torch

# Raw files, by path; the repository commit is pinned in fsc.hub.DATASET_REVISIONS.
FILES = {
    "wikitext2": ("Salesforce/wikitext", {
        "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
        "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    }),
    "c4": ("allenai/c4", {
        "train": "en/c4-train.00000-of-01024.json.gz",
        "validation": "en/c4-validation.00000-of-00008.json.gz",
    }),
    "slimpajama": ("DKYoon/SlimPajama-6B", {
        "train": "data/train-00000-of-00048-ab2b35705f029d94.parquet",
        "validation": "data/validation-00000-of-00001-4fb685c22a3f91ef.parquet",
        "test": "data/test-00000-of-00001-9f769cf7ce219017.parquet",
    }),
    "alpaca": ("tatsu-lab/alpaca", {
        "train": "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    }),
    # Held-out domains chosen independently of every calibration source:
    # human-written instructions, grade-school math, biomedical abstracts, Python,
    # eight non-English languages, and 19th-century books.
    "dolly": ("databricks/databricks-dolly-15k", {"test": "databricks-dolly-15k.jsonl"}),
    "gsm8k": ("openai/gsm8k", {"test": "main/test-00000-of-00001.parquet"}),
    "pubmedqa": ("qiaojin/PubMedQA", {"test": "pqa_labeled/train-00000-of-00001.parquet"}),
    "mbpp": ("google-research-datasets/mbpp", {
        "test": "full/test-00000-of-00001.parquet",
        "train": "full/train-00000-of-00001.parquet",
        "validation": "full/validation-00000-of-00001.parquet",
        "prompt": "full/prompt-00000-of-00001.parquet",
    }),
    # OPUS-100 names a pair alphabetically: de-en but en-fr.
    "opus": ("Helsinki-NLP/opus-100", {
        "de": "de-en/test-00000-of-00001.parquet", "fr": "en-fr/test-00000-of-00001.parquet",
        "es": "en-es/test-00000-of-00001.parquet", "ru": "en-ru/test-00000-of-00001.parquet",
        "zh": "en-zh/test-00000-of-00001.parquet", "ja": "en-ja/test-00000-of-00001.parquet",
        "ar": "ar-en/test-00000-of-00001.parquet", "vi": "en-vi/test-00000-of-00001.parquet",
    }),
    "pg19": ("emozilla/pg19-test", {"test": "data/test-00000-of-00001-29a571947c0b5ccc.parquet"}),
}

# Sources that are only evaluated, never calibrated on.
EVAL_DOMAINS = ("dolly", "gsm8k", "pubmedqa", "mbpp", "opus", "pg19")
OPUS_BLOCK = 20     # sentences per pseudo-document, one language per document
PG19_SPAN = 30_000  # characters read around a random offset of a book, per window

# RedPajama subsets inside SlimPajama, by their meta.redpajama_set_name.
SP_SUBSETS = {
    "cc": "RedPajamaCommonCrawl",
    "c4": "RedPajamaC4",
    "github": "RedPajamaGithub",
    "book": "RedPajamaBook",
    "arxiv": "RedPajamaArXiv",
    "wiki": "RedPajamaWikipedia",
    "stack": "RedPajamaStackExchange",
}

ALPACA_HOLDOUT = 2000       # last examples of the train file, reserved for evaluation
C4_PPL_DOCS = 1100          # first validation documents, used for perplexity (GPTQ convention)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task{ctx}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{input_block}### Response:\n{output}"
)


def raw_file(source: str, split: str) -> str:
    from huggingface_hub import hf_hub_download
    from .hub import dataset_revision
    repo, files = FILES[source]
    return hf_hub_download(repo, files[split], repo_type="dataset",
                           revision=dataset_revision(repo))


def _read_parquet(path: str, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pydict()


def _read_jsonl_gz(path: str, limit: int | None = None) -> list[str]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if limit is not None and len(out) >= limit:
                break
    return out


def alpaca_records() -> list[tuple[str, str, str]]:
    """(instruction, input, output) of every Alpaca example, stripped, in file order."""
    d = _read_parquet(raw_file("alpaca", "train"))
    return [(ins.strip(), (inp or "").strip(), out.strip())
            for ins, inp, out in zip(d["instruction"], d["input"], d["output"])]


def alpaca_documents(variant: str = "template") -> list[str]:
    """Alpaca examples as text.

    template: the Alpaca prompt template (the calibration set of SVD-LLM and LSP).
    plain:    the same instruction, input and response, separated by blank lines,
              without the template's header and markers.
    resp:     the responses alone.
    """
    docs = []
    for ins, inp, out in alpaca_records():
        if variant == "template":
            docs.append(ALPACA_TEMPLATE.format(
                ctx=", paired with an input that provides further context" if inp else "",
                instruction=ins, input_block=f"### Input:\n{inp}\n\n" if inp else "",
                output=out))
        elif variant == "plain":
            docs.append("\n\n".join(x for x in (ins, inp, out) if x))
        elif variant == "resp":
            docs.append(out)
        else:
            raise ValueError(variant)
    return docs


def alpaca_chat_documents(tokenizer) -> list[str]:
    """Alpaca examples in the model's own chat format (instruct models): the
    instruction (and input) as the user turn, the response as the assistant turn.
    The template's leading BOS text is dropped: windows add BOS themselves."""
    docs = []
    bos_text = tokenizer.bos_token or ""
    for ins, inp, out in alpaca_records():
        user = ins if not inp else f"{ins}\n\n{inp}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": out}],
            tokenize=False)
        if bos_text and text.startswith(bos_text):
            text = text[len(bos_text):]
        docs.append(text)
    return docs


def wikitext_articles(split: str) -> list[str]:
    """WikiText-2 split into its articles, at the top-level ' = Title = ' lines."""
    import re
    lines = _read_parquet(raw_file("wikitext2", split), columns=["text"])["text"]
    title = re.compile(r"^ = [^=].* = \n?$")
    arts, cur = [], []
    for ln in lines:
        if title.match(ln) and cur:
            arts.append("".join(cur).strip())
            cur = []
        cur.append(ln)
    if cur:
        arts.append("".join(cur).strip())
    return [a for a in arts if a]


def eval_documents(source: str) -> list[str]:
    """Documents of the held-out evaluation domains (never used for calibration)."""
    if source == "dolly":
        out = []
        with open(raw_file("dolly", "test"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                parts = [r["instruction"].strip(), (r.get("context") or "").strip(),
                         r["response"].strip()]
                out.append("\n\n".join(p for p in parts if p))
        return out
    if source == "gsm8k":
        d = _read_parquet(raw_file("gsm8k", "test"))
        return [f"{q.strip()}\n{a.strip()}" for q, a in zip(d["question"], d["answer"])]
    if source == "pubmedqa":
        d = _read_parquet(raw_file("pubmedqa", "test"))
        out = []
        for q, ctx, ans in zip(d["question"], d["context"], d["long_answer"]):
            body = " ".join(ctx["contexts"]) if isinstance(ctx, dict) else str(ctx)
            out.append(f"{q.strip()}\n{body.strip()}\n{ans.strip()}")
        return out
    if source == "mbpp":
        out = []
        for split in ("test", "train", "validation", "prompt"):
            d = _read_parquet(raw_file("mbpp", split))
            for text, code, tests in zip(d["text"], d["code"], d["test_list"]):
                code = code.replace("\r\n", "\n")
                out.append(f"{text.strip()}\n{code.strip()}\n" + "\n".join(tests))
        return out
    if source == "opus":
        # One pseudo-document per OPUS_BLOCK consecutive non-English sentences of a
        # language; languages interleaved so any prefix of the list mixes all eight.
        per_lang = []
        for lang in FILES["opus"][1]:
            d = _read_parquet(raw_file("opus", lang), columns=["translation"])["translation"]
            sents = [t[lang].strip() for t in d if t.get(lang) and t[lang].strip()]
            per_lang.append([" ".join(sents[i:i + OPUS_BLOCK])
                             for i in range(0, len(sents), OPUS_BLOCK)])
        out = []
        for i in range(max(len(x) for x in per_lang)):
            out += [docs[i] for docs in per_lang if i < len(docs)]
        return out
    if source == "pg19":
        return _read_parquet(raw_file("pg19", "test"), columns=["text"])["text"]
    raise ValueError(f"no evaluation reader for {source!r}")


def documents(source: str, split: str) -> list[str]:
    """Plain-text documents of a source split (order as stored)."""
    if source == "c4":
        return _read_jsonl_gz(raw_file("c4", split))
    if source == "alpaca":
        return alpaca_documents("template")
    if source in ("alpaca-plain", "alpaca-resp"):
        return alpaca_documents(source.split("-", 1)[1])
    if source == "wikitext2-art":
        return wikitext_articles(split)
    if source == "slimpajama" or source.startswith("sp-"):
        d = _read_parquet(raw_file("slimpajama", split), columns=["text", "meta"])
        if source == "slimpajama":
            return d["text"]
        want = SP_SUBSETS[source[3:]]
        return [t for t, m in zip(d["text"], d["meta"]) if _set_name(m) == want]
    raise ValueError(f"no document reader for source {source!r}")


def _set_name(meta) -> str:
    if isinstance(meta, dict):
        return meta.get("redpajama_set_name", "")
    if isinstance(meta, str):
        try:
            return json.loads(meta).get("redpajama_set_name", "")
        except json.JSONDecodeError:
            return ""
    return ""


def wikitext_text(split: str) -> str:
    d = _read_parquet(raw_file("wikitext2", split), columns=["text"])
    return "\n\n".join(d["text"])


# ----------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------

def special_ids(tokenizer) -> tuple[int | None, int]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS token to separate documents")
    return bos, eos


def _cut(stream: list[int], n: int, seqlen: int, bos: int | None) -> torch.Tensor:
    body = seqlen - 1 if bos is not None else seqlen
    need = n * body
    if len(stream) < need:
        raise ValueError(f"stream has {len(stream)} tokens, need {need}")
    t = torch.tensor(stream[:need], dtype=torch.long).view(n, body)
    if bos is not None:
        t = torch.cat([torch.full((n, 1), bos, dtype=torch.long), t], 1)
    return t


def _pack(tokenizer, docs: list[str], n_tokens: int, eos: int, start: int = 0,
          batch: int = 256, doc_bos: int | None = None) -> tuple[list[int], int]:
    """Tokenize docs[start:] until ``n_tokens`` tokens are packed; returns (stream, next_index).

    With ``doc_bos``, every document starts with that token (BOS) as well as ending with EOS.
    """
    stream: list[int] = []
    i = start
    while len(stream) < n_tokens:
        if i >= len(docs):
            raise ValueError(f"ran out of documents after {len(stream)} tokens")
        chunk = docs[i:i + batch]
        enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        for ids in enc:
            if doc_bos is not None:
                stream.append(doc_bos)
            stream.extend(ids)
            stream.append(eos)
            i += 1
            if len(stream) >= n_tokens:
                break
    return stream, i


def _pack_ids(docs: list[list[int]], n_tokens: int, eos: int, start: int = 0
              ) -> tuple[list[int], int]:
    """Pack already-tokenized documents, EOS-separated, until n_tokens."""
    stream: list[int] = []
    i = start
    while len(stream) < n_tokens:
        if i >= len(docs):
            raise ValueError(f"ran out of documents after {len(stream)} tokens")
        stream.extend(docs[i])
        stream.append(eos)
        i += 1
    return stream, i


def _pool_documents(path: str) -> list[list[int]]:
    """Token documents of a self-generated pool written with --documents or --responses."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    flat, lengths = blob["docs_flat"].tolist(), blob["doc_lengths"].tolist()
    docs, pos = [], 0
    for n in lengths:
        docs.append(flat[pos:pos + n])
        pos += n
    return docs


def _segment_windows(segs: list[list[int]], n: int, seqlen: int, bos: int | None,
                     eos: int) -> torch.Tensor:
    """Windows of equal-length segments separated by EOS (the matched-segment format).

    With BOS: [BOS] s1 [EOS] s2 ... [EOS] sS; without: s1 [EOS] ... sS [EOS]. Either
    way every window has S segments, S - 1 or S separators and one BOS at most.
    """
    L = len(segs[0])
    S = seqlen // (L + 1)
    if S * (L + 1) != seqlen:
        raise ValueError(f"segment length {L} does not tile windows of {seqlen}")
    if len(segs) < n * S:
        raise ValueError(f"{len(segs)} segments, need {n * S}")
    rows = []
    for w in range(n):
        row = [] if bos is None else [bos]
        for j in range(S):
            row += segs[w * S + j]
            if bos is None or j < S - 1:
                row.append(eos)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


SEGMENT_CAP = 32     # spans taken from one document, at most (matched-segment format)


def _doc_spans(ids: list[int], L: int, cap: int = SEGMENT_CAP) -> list[list[int]]:
    """Consecutive L-token spans from the start of a document, at most ``cap``."""
    return [ids[i * L:(i + 1) * L] for i in range(min(cap, len(ids) // L))]


def _span_pools(token_docs, L: int, need_train: int, need_val: int, seed: int
                ) -> tuple[list[list[int]], list[list[int]], dict]:
    """Training and validation spans from disjoint documents, each pool shuffled.

    ``token_docs`` yields token lists in a (seeded) random document order. The
    training pool takes documents until it holds ``need_train`` spans, the
    validation pool the following documents until ``need_val``.
    """
    tr, va, n_docs, n_short = [], [], 0, 0
    for ids in token_docs:
        n_docs += 1
        spans = _doc_spans(ids, L)
        if not spans:
            n_short += 1
            continue
        if len(tr) < need_train:
            tr += spans
        else:
            va += spans
            if len(va) >= need_val:
                break
    if len(tr) < need_train or len(va) < need_val:
        raise ValueError(f"{len(tr)}+{len(va)} spans of {L} tokens, need {need_train}+{need_val}")
    rng = random.Random(seed)
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr[:need_train], va[:need_val], {"docs_read": n_docs, "docs_too_short": n_short}


def _tokenized(tokenizer, docs: list[str], batch: int = 256):
    for i in range(0, len(docs), batch):
        yield from tokenizer(docs[i:i + batch], add_special_tokens=False)["input_ids"]


SEGMENT_STATS: dict = {}    # filled by calibration_windows in the segment format


def calibration_windows(tokenizer, source: str, n_train: int, n_val: int, seqlen: int,
                        seed: int = 0, selfgen_path: str | None = None, use_bos: bool = True,
                        doc_bos: bool = False, segment_len: int = 0
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """(train, val) windows of shape (n, seqlen) from disjoint documents of ``source``.

    ``use_bos=False`` cuts plain seqlen-token chunks even for models with a BOS
    token (the window-format ablation); self-generated windows always start at
    the model's own document-start token.

    ``segment_len > 0`` builds every window from equal segments instead of packing
    whole documents (the matched-segment format): consecutive ``segment_len``-token
    spans from the start of each document (at most SEGMENT_CAP per document;
    shorter documents contribute none), pooled and shuffled, EOS-separated. Every
    source then has the same number of boundaries per window and the same segment
    length; training and validation spans come from disjoint documents.
    """
    bos, eos = special_ids(tokenizer)
    if not use_bos:
        bos = None
    body = seqlen - 1 if bos is not None else seqlen

    if segment_len > 0:
        S = seqlen // (segment_len + 1)
        if source in ("selfdoc", "selfresp"):
            docs = _pool_documents(selfgen_path)
            random.Random(seed).shuffle(docs)
            stream = iter(docs)
        else:
            if source == "alpaca-chat":
                docs = alpaca_chat_documents(tokenizer)[:-ALPACA_HOLDOUT]
            else:
                docs = documents(source, "train")
                if source.startswith("alpaca"):
                    docs = docs[:-ALPACA_HOLDOUT]
            order = list(range(len(docs)))
            random.Random(seed).shuffle(order)
            stream = _tokenized(tokenizer, [docs[j] for j in order])
        tr, va, info = _span_pools(stream, segment_len, n_train * S, n_val * S, seed)
        SEGMENT_STATS.update(source=source, segments_per_window=S, **info)
        return (_segment_windows(tr, n_train, seqlen, bos, eos),
                _segment_windows(va, n_val, seqlen, bos, eos))

    if source in ("selfdoc", "selfresp"):
        if selfgen_path is None:
            raise ValueError(f"source {source!r} needs selfgen_path")
        docs = _pool_documents(selfgen_path)
        random.Random(seed).shuffle(docs)
        if doc_bos and bos is not None:
            docs = [[bos] + d for d in docs]
        s_tr, nxt = _pack_ids(docs, n_train * body, eos)
        s_va, _ = _pack_ids(docs, n_val * body, eos, start=nxt)
        return _cut(s_tr, n_train, seqlen, bos), _cut(s_va, n_val, seqlen, bos)

    if source == "selfpack":
        # Exact samples of length L restarted every L tokens: seqlen // L of them,
        # each beginning with the model's own start token, make one window.
        blob = torch.load(selfgen_path, map_location="cpu", weights_only=False)
        ids = blob["ids"] if isinstance(blob, dict) else blob
        L = ids.shape[1]
        k = seqlen // L
        if k * L != seqlen or ids.shape[0] < (n_train + n_val) * k:
            raise ValueError(f"pool {tuple(ids.shape)} cannot fill {n_train}+{n_val} windows of {seqlen}")
        g = torch.Generator().manual_seed(seed)
        ids = ids[torch.randperm(ids.shape[0], generator=g)][: (n_train + n_val) * k]
        w = ids.reshape(n_train + n_val, seqlen)
        return w[:n_train].clone(), w[n_train:].clone()

    if source == "selfgen":
        if selfgen_path is None:
            raise ValueError("source 'selfgen' needs selfgen_path")
        blob = torch.load(selfgen_path, map_location="cpu", weights_only=False)
        ids = blob["ids"] if isinstance(blob, dict) else blob
        if ids.shape[1] < seqlen or ids.shape[0] < n_train + n_val:
            raise ValueError(f"self-generated set {tuple(ids.shape)} too small for "
                             f"{n_train}+{n_val} windows of {seqlen}")
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(ids.shape[0], generator=g)
        ids = ids[perm, :seqlen]
        return ids[:n_train].clone(), ids[n_train:n_train + n_val].clone()

    if source == "wikitext2":
        rng = random.Random(seed)
        tr = tokenizer(wikitext_text("train"), add_special_tokens=False)["input_ids"]
        va = tokenizer(wikitext_text("validation"), add_special_tokens=False)["input_ids"]
        train = _random_chunks(tr, n_train, body, rng)
        val = _random_chunks(va, n_val, body, rng)
        return _add_bos(train, bos), _add_bos(val, bos)

    if source == "alpaca-chat":
        docs = alpaca_chat_documents(tokenizer)
    else:
        docs = documents(source, "train")
    if source.startswith("alpaca"):
        docs = docs[:-ALPACA_HOLDOUT]
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    docs = [docs[j] for j in order]
    db = bos if doc_bos else None
    s_tr, nxt = _pack(tokenizer, docs, n_train * body, eos, doc_bos=db)
    s_va, _ = _pack(tokenizer, docs, n_val * body, eos, start=nxt, doc_bos=db)
    return _cut(s_tr, n_train, seqlen, bos), _cut(s_va, n_val, seqlen, bos)


def mixed_windows(tokenizer, sources: list[tuple[str, str | None, float]], n_train: int,
                  n_val: int, seqlen: int, seed: int = 0, **kw) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows drawn from several sources in fixed proportions, shuffled together.

    ``sources`` is [(source, selfgen_path, fraction), ...]; fractions sum to one and
    the total budget is that of a single source.
    """
    fr = [f for _, _, f in sources]
    if abs(sum(fr) - 1.0) > 1e-6:
        raise ValueError(f"fractions {fr} do not sum to one")
    tr, va = [], []
    left_tr, left_va = n_train, n_val
    for i, (src, path, f) in enumerate(sources):
        last = i == len(sources) - 1
        nt = left_tr if last else int(round(n_train * f))
        nv = left_va if last else int(round(n_val * f))
        left_tr -= nt
        left_va -= nv
        a, b = calibration_windows(tokenizer, src, nt, nv, seqlen, seed=seed,
                                   selfgen_path=path, **kw)
        tr.append(a)
        va.append(b)
    g = torch.Generator().manual_seed(seed)
    tr, va = torch.cat(tr), torch.cat(va)
    return (tr[torch.randperm(tr.shape[0], generator=g)],
            va[torch.randperm(va.shape[0], generator=g)])


def _random_chunks(stream: list[int], n: int, body: int, rng: random.Random) -> torch.Tensor:
    """n non-overlapping chunks of a single stream, chosen at random."""
    n_avail = len(stream) // body
    if n_avail < n:
        raise ValueError(f"stream holds {n_avail} chunks of {body}, need {n}")
    idx = sorted(rng.sample(range(n_avail), n))
    t = torch.tensor(stream[: n_avail * body], dtype=torch.long).view(n_avail, body)
    return t[idx]


def _add_bos(t: torch.Tensor, bos: int | None) -> torch.Tensor:
    if bos is None:
        return t
    return torch.cat([torch.full((t.shape[0], 1), bos, dtype=torch.long), t], 1)


# ----------------------------------------------------------------------------
# Evaluation corpora
# ----------------------------------------------------------------------------

def eval_stream(tokenizer, name: str, special: bool = True) -> torch.Tensor:
    """1-D token stream for perplexity, following the GPTQ/SparseGPT conventions.

    ``special=False`` tokenizes without the leading BOS, for BOS-prefixed windows.
    """
    if name == "wikitext2":
        text = wikitext_text("test")
    elif name == "c4":
        text = " ".join(_read_jsonl_gz(raw_file("c4", "validation"), limit=C4_PPL_DOCS))
    else:
        raise ValueError(name)
    return tokenizer(text, return_tensors="pt", add_special_tokens=special).input_ids[0]


def eval_windows_sp(tokenizer, subset: str, n: int, seqlen: int, seed: int = 0) -> torch.Tensor:
    """Windows from SlimPajama validation documents of one RedPajama subset."""
    bos, eos = special_ids(tokenizer)
    body = seqlen - 1 if bos is not None else seqlen
    d = _read_parquet(raw_file("slimpajama", "validation"), columns=["text", "meta"])
    want = SP_SUBSETS[subset]
    docs = [t for t, m in zip(d["text"], d["meta"]) if _set_name(m) == want]
    random.Random(seed).shuffle(docs)
    stream, _ = _pack(tokenizer, docs, n * body, eos)
    return _cut(stream, n, seqlen, bos)


def kl_eval_windows(tokenizer, source: str, n: int, seqlen: int,
                    selfgen_path: str | None = None, seed: int = 1234) -> torch.Tensor:
    """Held-out windows of a source for measuring KL(dense || compressed).

    Disjoint from every calibration set: WikiText-2 test chunks, C4 validation
    documents after the perplexity set, SlimPajama validation documents, the
    held-out Alpaca examples, and a separately sampled self-generated pool.
    """
    bos, eos = special_ids(tokenizer)
    body = seqlen - 1 if bos is not None else seqlen
    if source == "selfgen":
        blob = torch.load(selfgen_path, map_location="cpu", weights_only=False)
        ids = blob["ids"] if isinstance(blob, dict) else blob
        return ids[:n, :seqlen].clone()
    if source == "wikitext2":
        stream = tokenizer(wikitext_text("test"), add_special_tokens=False)["input_ids"]
        n = min(n, len(stream) // body)
        t = torch.tensor(stream[: n * body], dtype=torch.long).view(n, body)
        return _add_bos(t, bos)
    if source == "pg19":
        return book_windows(tokenizer, n, seqlen, seed=seed)
    if source == "c4":
        docs = _read_jsonl_gz(raw_file("c4", "validation"))[C4_PPL_DOCS:]
    elif source == "slimpajama":
        docs = _read_parquet(raw_file("slimpajama", "validation"), columns=["text"])["text"]
    elif source == "alpaca":
        docs = documents("alpaca", "train")[-ALPACA_HOLDOUT:]
    elif source in EVAL_DOMAINS:
        docs = eval_documents(source)
    else:
        raise ValueError(f"no held-out KL set for {source!r}")
    docs = list(docs)
    random.Random(seed).shuffle(docs)
    try:
        stream, _ = _pack(tokenizer, docs, n * body, eos)
    except ValueError:
        # A small domain (MBPP) holds fewer tokens than n windows: use all of them.
        stream = _pack_all(tokenizer, docs, eos)
        n = len(stream) // body
    return _cut(stream, n, seqlen, bos)


def _pack_all(tokenizer, docs: list[str], eos: int, batch: int = 256) -> list[int]:
    stream: list[int] = []
    for i in range(0, len(docs), batch):
        for ids in tokenizer(docs[i:i + batch], add_special_tokens=False)["input_ids"]:
            stream.extend(ids)
            stream.append(eos)
    return stream


def book_windows(tokenizer, n: int, seqlen: int, seed: int = 1234,
                 span: int | None = None) -> torch.Tensor:
    """One window per PG-19 test book, at a seeded random offset of the book.

    Packing whole books would fill every window from one or two books; one window
    per book spreads them over n of the 100 test books. Windows start with BOS for
    models that have one; ``span`` characters are tokenized around the offset.
    """
    bos, _ = special_ids(tokenizer)
    body = seqlen - 1 if bos is not None else seqlen
    span = span or max(PG19_SPAN, 8 * seqlen)
    books = eval_documents("pg19")
    rng = random.Random(seed)
    order = list(range(len(books)))
    rng.shuffle(order)
    rows = []
    for j in order:
        text = books[j]
        if len(text) < span + 1000:
            continue
        start = rng.randrange(0, len(text) - span)
        ids = tokenizer(text[start:start + span], add_special_tokens=False)["input_ids"][1:]
        if len(ids) < body:
            continue
        rows.append(ids[:body])
        if len(rows) == n:
            break
    t = torch.tensor(rows, dtype=torch.long)
    return _add_bos(t, bos)
