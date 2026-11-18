#!/usr/bin/env python3
"""Print what compressed models write after the passkey prompt (diagnostic for the
passkey column of the behavior table): greedy continuations with a longer budget, and
the hit rate when the key may appear anywhere in them.

    uv run --no-sync python scripts/passkey_probe.py out/runs/<model>/<run> [...] [--new 32] [--n 10]
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behave as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--length", type=int, default=2000)
    args = ap.parse_args()
    for rd in args.run_dirs:
        model, tok, parts, _ = B.rebuild(rd, "cuda", torch.bfloat16)
        rng = random.Random(1234)
        head = ("There is an important info hidden inside a lot of irrelevant text. Find it and "
                "memorize it. I will quiz you about the important information there.\n\n")
        tail = "\n\nWhat is the pass key? The pass key is"
        n_fill = len(tok(B.FILLER, add_special_tokens=False)["input_ids"])
        reps = max(1, (args.length - 60) // n_fill)
        hits8 = hitsN = 0
        with B.compressed(model, parts, torch.bfloat16):
            for i in range(args.n):
                key = rng.randint(10000, 99999)
                depth = rng.randint(0, reps)
                needle = f"The pass key is {key}. Remember it. {key} is the pass key. "
                text = head + B.FILLER * depth + needle + B.FILLER * (reps - depth) + tail
                ids = tok(text, return_tensors="pt").input_ids
                out = B.generate(model, ids, args.new, False, 0, "cuda", batch=1)[0].tolist()
                full = tok.decode(out)
                first8 = tok.decode(out[:8])
                hits8 += str(key) in first8
                hitsN += str(key) in full
                if i < 4:
                    print(f"{os.path.basename(rd)} key {key}: {full!r}")
        print(f"== {rd}: hit within 8 tokens {hits8}/{args.n}, within {args.new} tokens {hitsN}/{args.n}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
