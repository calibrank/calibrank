#!/usr/bin/env python3
"""What the dense models write when they calibrate themselves: repetition, script and
language, and kind of content of the self-generated pools, by position.

    uv run --no-sync python scripts/analyze_selfgen.py \
        --pools qwen3-1.7b:selfgen:out/calib/qwen3-1.7b/selfgen-T1.0-s0.pt ... \
        --out out/analysis/selfgen-composition.json

CPU only. For exact samples (selfgen), every 2,048-token sequence is cut into four
blocks of 512 positions; for documents (selfdoc), each document is one unit. Per
block, averaged over sequences:

  rep4          fraction of 4-gram occurrences that repeat an earlier 4-gram of the block
  distinct2     distinct token bigrams over all bigrams of the block
  gzip_ratio    compressed / raw bytes of the decoded text (low = repetitive)
  doc_starts    document-start tokens (BOS, or EOS where the model has no BOS)
  script_*      share of letters in Latin, CJK, Cyrillic, Arabic and other scripts
  english       share of Latin-script words that are among 50 common English words
  code          share of lines that look like source code
  math          share of lines with LaTeX or formula markup
  url           URLs per 1,000 characters
  digits        share of characters that are digits
  degenerate    share of blocks in which at least half of the 4-grams repeat (loops)
  cjk_major     share of blocks whose letters are mostly CJK

A pool given as model:kind:tag:path is stored under model:kind:tag, so that several
pools of the same model and kind (one per seed) can be compared.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import unicodedata

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fsc.utils import log  # noqa: E402

MODELS = {"opt-1.3b": "facebook/opt-1.3b", "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
          "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base", "llama-3.1-8b": "meta-llama/Llama-3.1-8B"}
ENGLISH = set("the of and to a in is that for it as was with be by on not he i this are or his "
              "from at which but have an they you were her she there had been one all we their "
              "has would when".split())
CODE = re.compile(r"(^\s*(def |class |import |from \w+ import|#include|public |private |"
                  r"function |var |let |const |return\b|for \(|if \(|while \()|[;{}]\s*$|=>|::)")
MATH = re.compile(r"(\\frac|\\begin|\\end\{|\\sum|\\int|\\alpha|\$[^$]+\$|\^\{|_\{)")
URL = re.compile(r"https?://|www\.")


def script_of(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for key, label in (("LATIN", "latin"), ("CJK", "cjk"), ("HIRAGANA", "cjk"),
                       ("KATAKANA", "cjk"), ("HANGUL", "cjk"), ("CYRILLIC", "cyrillic"),
                       ("ARABIC", "arabic")):
        if key in name:
            return label
    return "other"


def block_stats(ids: list[int], text: str, starts: set[int]) -> dict:
    grams = [tuple(ids[j:j + 4]) for j in range(len(ids) - 3)]
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    big = [tuple(ids[j:j + 2]) for j in range(len(ids) - 1)]
    raw = text.encode("utf-8", "ignore")
    letters = [c for c in text if c.isalpha()]
    scripts = {"latin": 0, "cjk": 0, "cyrillic": 0, "arabic": 0, "other": 0}
    for c in letters:
        scripts[script_of(c)] += 1
    nl = max(len(letters), 1)
    words = re.findall(r"[A-Za-z]+", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    nlines = max(len(lines), 1)
    out = {
        "rep4": rep / max(len(grams), 1),
        "distinct2": len(set(big)) / max(len(big), 1),
        "gzip_ratio": len(gzip.compress(raw)) / max(len(raw), 1),
        "doc_starts": sum(1 for t in ids if t in starts),
        "english": sum(w.lower() in ENGLISH for w in words) / max(len(words), 1),
        "code": sum(bool(CODE.search(ln)) for ln in lines) / nlines,
        "math": sum(bool(MATH.search(ln)) for ln in lines) / nlines,
        "url": 1000 * len(URL.findall(text)) / max(len(text), 1),
        "digits": sum(c.isdigit() for c in text) / max(len(text), 1),
    }
    out.update({f"script_{k}": v / nl for k, v in scripts.items()})
    out["degenerate"] = float(out["rep4"] >= 0.5)
    out["cjk_major"] = float(out["script_cjk"] > 0.5)
    return out


def mean_dicts(ds: list[dict]) -> dict:
    keys = ds[0].keys()
    return {k: float(np.mean([d[k] for d in ds])) for k in keys}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", required=True, help="model:kind:path")
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from fsc import hub

    res = {}
    for spec in args.pools:
        if spec.count(":") >= 3:
            model, kind, tag, path = spec.split(":", 3)
            key = f"{model}:{kind}:{tag}"
        else:
            model, kind, path = spec.split(":", 2)
            key = f"{model}:{kind}"
        tok = hub.load_tokenizer(MODELS[model])
        starts = {t for t in (tok.bos_token_id, tok.eos_token_id) if t is not None}
        blob = torch.load(path, map_location="cpu", weights_only=False)
        log(f"{model} {kind}: {path}")
        if kind == "selfgen":
            ids = blob["ids"][: args.max_seqs].tolist()
            blocks = {}
            for seq in ids:
                for b in range(0, len(seq), args.block):
                    chunk = seq[b:b + args.block]
                    text = tok.decode(chunk, skip_special_tokens=True)
                    blocks.setdefault(f"{b}-{b + len(chunk)}", []).append(
                        block_stats(chunk, text, starts))
            res[key] = {k: mean_dicts(v) for k, v in blocks.items()}
        else:
            flat, lengths = blob["docs_flat"].tolist(), blob["doc_lengths"].tolist()
            docs, pos = [], 0
            for n in lengths:
                docs.append(flat[pos:pos + n])
                pos += n
            sample = docs[: args.max_seqs * 4]
            stats = [block_stats(d, tok.decode(d, skip_special_tokens=True), starts)
                     for d in sample if d]
            res[key] = {"documents": mean_dicts(stats),
                                      "mean_len": float(np.mean(lengths)),
                                      "frac_max_len": float(blob.get("frac_truncated", float("nan")))}
        log(json.dumps(res[key], indent=None)[:600])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
