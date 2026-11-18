#!/usr/bin/env python3
"""Measured memory and latency of the dense model and of its merged compressed versions.

    uv run --no-sync python scripts/bench.py --model meta-llama/Llama-3.2-1B --ratios 0.3,0.5 \
        --out out/extended/analysis/bench-llama-3.2-1b.json

Speed and memory depend only on the shapes of the factors, which the uniform
allocation fixes from the weight shapes alone: every calibration source gives the
same ranks. The compressed models are therefore built with the uniform ranks and
random orthonormal removed bases, merged into factors exactly as trained models
are (two dense matrix products per factorized layer, no custom kernel).

Measured, in bf16 on one GPU, median of repeated runs after warm-up:
  weights_gb      bytes of all parameters (embeddings and head included)
  prefill_ms      one forward pass over a 2,048-token prompt, batch 1 and 8
  decode_tok_s    tokens per second generating 128 tokens after a 512-token
                  prompt with the KV cache, batch 1 and 16
  peak_gb         the model's weights plus the peak memory added by the decode
                  (KV cache and activations), batch 1 and 16
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsc import alloc, arch, project  # noqa: E402
from fsc.utils import env_info, log, write_json  # noqa: E402

import compress  # noqa: E402


def weights_gb(model) -> float:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total / 2**30


def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def prefill_ms(model, batch: int, length: int, vocab: int, reps: int = 5) -> float:
    x = torch.randint(0, vocab, (batch, length), device="cuda")
    for _ in range(2):
        model(input_ids=x, use_cache=False)
    times = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        model(input_ids=x, use_cache=False)
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@torch.no_grad()
def decode_tok_s(model, batch: int, prompt: int, new: int, vocab: int, reps: int = 3):
    x = torch.randint(0, vocab, (batch, prompt), device="cuda")

    def run():
        o = model(input_ids=x, use_cache=True)
        past, cur = o.past_key_values, o.logits[:, -1:].argmax(-1)
        _sync()
        t0 = time.perf_counter()
        for _ in range(new):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past, cur = o.past_key_values, o.logits[:, -1:].argmax(-1)
        _sync()
        return batch * new / (time.perf_counter() - t0)

    run()
    # Peak memory above what is allocated before the decode (the KV cache and the
    # activations). The swapped-out dense layers stay allocated during a compressed
    # measurement, so the absolute peak would count them; measure() adds the
    # model's own weights to this increment instead.
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    rates = [run() for _ in range(reps)]
    return statistics.median(rates), (torch.cuda.max_memory_allocated() - base) / 2**30


def measure(model, vocab: int) -> dict:
    out = {"weights_gb": weights_gb(model)}
    for b in (1, 8):
        out[f"prefill_ms_b{b}"] = prefill_ms(model, b, 2048, vocab)
    for b in (1, 16):
        rate, extra = decode_tok_s(model, b, 512, 128, vocab)
        out[f"decode_tok_s_b{b}"] = rate
        out[f"peak_gb_b{b}"] = out["weights_gb"] + extra
    torch.cuda.empty_cache()
    log("    " + " ".join(f"{k}={v:.3g}" for k, v in out.items()))
    return out


def random_bases(units, ks, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for u in units:
        k = ks[u.name]
        if k > 0:
            q, _ = torch.linalg.qr(torch.randn(u.d, k, generator=g, dtype=torch.float64))
            out[u.name] = q.float().cuda()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ratios", default="0.3,0.5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dtype = torch.bfloat16
    model, tok = compress.load_model(args.model, dtype, "cuda")
    vocab = len(tok)
    res = {"model": args.model, "gpu": torch.cuda.get_device_name(0), "env": env_info()}
    log("dense")
    res["dense"] = measure(model, vocab)
    units = arch.discover_units(model)
    for r in [float(x) for x in args.ratios.split(",") if x]:
        ks = alloc.uniform_alloc(units, r)
        parts = compress.merge_parts(model, units, random_bases(units, ks), dtype)
        log(f"ratio {r}: realized {arch.realized_ratio(units, ks):.4f}")
        with project.swapped(model, parts, dtype):
            m = measure(model, vocab)
            m["realized_ratio"] = arch.realized_ratio(units, ks)
            m["stored_compressible"] = project.stored_params(model, units)
        m["dense_compressible"] = arch.total_dense(units)
        res[f"r{int(round(r * 100))}"] = m
        del parts
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_json(args.out, res)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
