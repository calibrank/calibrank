#!/usr/bin/env python3
"""Statistics of the extended evaluation: seeds, paired comparisons, ranking robustness.

    uv run --no-sync python scripts/run_stats.py [--records records/extended]
        [--out results/extended/run_stats.json]

Reads the run summaries (metrics.json) and per-item results (items_final.npz) under
<records>/runs. Everything the paper reports about seeds, confidence intervals and
rankings comes from the JSON this writes. On a CPU it takes about ten minutes.

Paired bootstrap (Efron and Tibshirani): for two calibration sources A and B of the
same model and ratio, every evaluation item (benchmark question, KL or perplexity
window) is first averaged over the seeds that both sources have, then items are
resampled with replacement (within each task for accuracy), the same indices for
A and B, and the statistic is recomputed; the 2.5% and 97.5% quantiles of 2000
replicates bound the 95% interval.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

SOURCES = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen", "selfdoc"]
MODELS = ["opt-1.3b", "llama-3.2-1b", "qwen3-1.7b", "llama-3.1-8b"]
TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa", "boolq"]
CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "hellaswag": 0.25,
          "winogrande": 0.5, "openbookqa": 0.25, "boolq": 0.5}
KL_SETS = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen"]
NEW_DOMAINS = ["dolly", "gsm8k", "pubmedqa", "mbpp", "opus", "pg19"]
BOOT = 2000


def rtag(r: float) -> str:
    return f"r{int(round(r * 100))}"


def load_metrics(runs: str, model: str, run: str) -> dict | None:
    p = os.path.join(runs, model, run, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return m if m.get("done") else None


def load_items(runs: str, model: str, run: str, phase: str = "final") -> dict | None:
    p = os.path.join(runs, model, run, f"items_{phase}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k.replace("|", "/"): z[k] for k in z.files}


def seeds_of(runs, model, source, ratio, tag="", seeds=(0, 1, 2)):
    out = {}
    for s in seeds:
        m = load_metrics(runs, model, f"proj-{source}-{rtag(ratio)}-s{s}{tag}")
        if m is not None:
            out[s] = m
    return out


def acc_means(zs: dict) -> dict:
    a7 = float(np.mean([zs[t]["acc"] for t in TASKS]))
    a6 = float(np.mean([zs[t]["acc"] for t in TASKS if t != "boolq"]))
    an = float(np.mean([(zs[t]["acc"] - CHANCE[t]) / (1 - CHANCE[t]) for t in TASKS]))
    return {"acc7": a7, "acc6": a6, "accnorm": an, "boolq": zs["boolq"]["acc"]}


def summary_metrics(m: dict, phase: str = "final") -> dict:
    f = m[phase]
    sp = [v for v in f["sp_ppl"].values()]
    out = {"wiki": f["ppl"]["wikitext2"], "c4": f["ppl"]["c4"],
           "sp": math.exp(sum(math.log(v) for v in sp) / len(sp)),
           "wiki_stream": f["ppl_stream"]["wikitext2"]}
    if "zs" in f:
        out.update(acc_means(f["zs"]))
    for k, v in f.get("kl", {}).items():
        out[f"kl_{k}"] = v
    for k, v in f.get("xppl", {}).items():
        out[f"xppl_{k}"] = v
    return out


def mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")


# ----------------------------------------------------------------------------
# per-item arrays, averaged over seeds
# ----------------------------------------------------------------------------

def item_acc(items_by_seed: list[dict], task: str) -> np.ndarray | None:
    arrs = [d.get(f"zs/{task}/acc") for d in items_by_seed]
    if any(a is None for a in arrs) or len({len(a) for a in arrs}) != 1:
        return None
    return np.mean(arrs, axis=0)


def item_kl(items_by_seed: list[dict], name: str) -> np.ndarray | None:
    """Per-window token-mean KL, averaged over seeds."""
    arrs = []
    for d in items_by_seed:
        kl, nt = d.get(f"kl/{name}"), d.get(f"ntok/{name}")
        if kl is None:
            return None
        arrs.append(kl / nt)
    if len({len(a) for a in arrs}) != 1:
        return None
    return np.mean(arrs, axis=0)


def paired_acc(itA: list[dict], itB: list[dict], rng, tasks=TASKS) -> dict | None:
    """Difference A - B of the mean accuracy over ``tasks``, with a 95% bootstrap interval."""
    per = []
    for t in tasks:
        a, b = item_acc(itA, t), item_acc(itB, t)
        if a is None or b is None or len(a) != len(b):
            return None
        per.append((a, b))
    point = float(np.mean([a.mean() - b.mean() for a, b in per]))
    reps = np.zeros(BOOT)
    for i in range(BOOT):
        diffs = []
        for a, b in per:
            idx = rng.integers(0, len(a), len(a))
            diffs.append(a[idx].mean() - b[idx].mean())
        reps[i] = np.mean(diffs)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"diff": point, "lo": float(lo), "hi": float(hi)}


def paired_logratio(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """log(mean a / mean b) over windows (e.g. KL of two sources), bootstrap 95% CI."""
    point = float(math.log(a.mean() / b.mean()))
    reps = np.zeros(BOOT)
    n = len(a)
    for i in range(BOOT):
        idx = rng.integers(0, n, n)
        reps[i] = math.log(a[idx].mean() / b[idx].mean())
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"logratio": point, "ratio": math.exp(point), "lo": math.exp(lo), "hi": math.exp(hi)}


def item_nll(items_by_seed: list[dict], key: str) -> np.ndarray | None:
    """Per-window summed NLL of a perplexity set (ppl/wikitext2, ppl/c4), averaged over seeds."""
    arrs = [d.get(key) for d in items_by_seed]
    if any(a is None for a in arrs) or len({len(a) for a in arrs}) != 1:
        return None
    return np.mean(arrs, axis=0)


def paired_ppl(a: np.ndarray, b: np.ndarray, tokens_per_window: int, rng) -> dict:
    """Perplexity ratio exp(mean NLL_a - mean NLL_b) over shared windows, bootstrap 95% CI."""
    n = len(a)
    da = (a - b) / tokens_per_window
    point = float(math.exp(da.mean()))
    reps = np.exp([da[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"ratio": point, "lo": float(lo), "hi": float(hi)}


def ranking_probability(kls: dict[str, dict[str, np.ndarray]], sets: list[str], rng,
                        weights: np.ndarray | None = None) -> dict:
    """P(source has the lowest geometric-mean relative KL) under resampling of windows.

    ``kls[source][set]`` are per-window KLs on the same windows for every source.
    """
    srcs = [s for s in kls if all(t in kls[s] for t in sets)]
    if len(srcs) < 2:
        return {}
    n = {t: len(kls[srcs[0]][t]) for t in sets}
    wins = {s: 0 for s in srcs}
    w = np.ones(len(sets)) / len(sets) if weights is None else weights
    for _ in range(BOOT):
        means = {s: [] for s in srcs}
        for t in sets:
            idx = rng.integers(0, n[t], n[t])
            for s in srcs:
                means[s].append(kls[s][t][idx].mean())
        M = np.array([means[s] for s in srcs])
        rel = M / M.min(axis=0)
        geo = np.exp(np.log(rel) @ w)
        wins[srcs[int(np.argmin(geo))]] += 1
    return {s: wins[s] / BOOT for s in srcs}


def geo_relative(kl: dict[str, dict[str, float]], sets: list[str]) -> dict:
    srcs = [s for s in kl if all(t in kl[s] for t in sets)]
    best = {t: min(kl[s][t] for s in srcs) for t in sets}
    return {s: {"geo": math.exp(np.mean([math.log(kl[s][t] / best[t]) for t in sets])),
                "max": max(kl[s][t] / best[t] for t in sets)} for s in srcs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="records/extended", help="root holding runs/<model>/<run>/")
    ap.add_argument("--out", default="results/extended/run_stats.json")
    args = ap.parse_args(argv)
    args.runs = os.path.join(args.records, "runs")
    rng = np.random.default_rng(0)
    res: dict = {"seeds": {}, "paired_acc": {}, "paired_kl": {}, "uniform": {}, "boolq": {}}

    for m in MODELS:
        ratios = (0.3, 0.5)
        for r in ratios:
            key = f"{m}/{rtag(r)}"
            per_src, items_src = {}, {}
            for s in SOURCES:
                runs = seeds_of(args.runs, m, s, r)
                if not runs:
                    continue
                vals = [summary_metrics(x) for x in runs.values()]
                agg = {}
                for k in vals[0]:
                    agg[k] = mean_sd([v[k] for v in vals if k in v])
                agg["n_seeds"] = len(vals)
                per_src[s] = agg
                its = [load_items(args.runs, m, f"proj-{s}-{rtag(r)}-s{sd}") for sd in runs]
                items_src[s] = [x for x in its if x is not None]
            if not per_src:
                continue
            res["seeds"][key] = per_src

            # paired accuracy: Alpaca and WikiText-2 against every other source
            pa = {}
            for a in ("alpaca", "wikitext2", "selfdoc"):
                for b in SOURCES:
                    if a == b or a not in items_src or b not in items_src:
                        continue
                    k = min(len(items_src[a]), len(items_src[b]))
                    if k == 0:
                        continue
                    d7 = paired_acc(items_src[a][:k], items_src[b][:k], rng)
                    d6 = paired_acc(items_src[a][:k], items_src[b][:k], rng,
                                    [t for t in TASKS if t != "boolq"])
                    if d7:
                        pa[f"{a}-{b}"] = {"acc7": d7, "acc6": d6, "n_seeds": k}
            res["paired_acc"][key] = pa

            # paired KL: on each held-out set, the source of that set against the others
            pk, kls = {}, {}
            for s, its in items_src.items():
                if not its:
                    continue
                kls[s] = {}
                for t in KL_SETS + NEW_DOMAINS:
                    v = item_kl(its, t)
                    if v is not None:
                        kls[s][t] = v
            for t in KL_SETS:
                own = t if t != "selfgen" else "selfdoc"
                if own not in kls or t not in kls[own]:
                    continue
                pk[t] = {b: paired_logratio(kls[b][t], kls[own][t], rng)
                         for b in kls if b != own and t in kls[b]}
            res["paired_kl"][key] = pk

            # paired perplexity on the WikiText-2 and C4 test windows: own source vs others
            pp = {}
            for t in ("wikitext2", "c4"):
                if t not in items_src or not items_src[t]:
                    continue
                own = item_nll(items_src[t], f"ppl/{t}")
                if own is None:
                    continue
                pp[t] = {}
                for b in items_src:
                    if b == t or not items_src[b]:
                        continue
                    other = item_nll(items_src[b], f"ppl/{t}")
                    if other is not None and len(other) == len(own):
                        pp[t][b] = paired_ppl(other, own, 2047, rng)
            res.setdefault("paired_ppl", {})[key] = pp

            # uniform fidelity: point estimates from seed means, and bootstrap ranking
            klmean = {s: {t: per_src[s][f"kl_{t}"][0] for t in KL_SETS + NEW_DOMAINS
                          if f"kl_{t}" in per_src[s]} for s in per_src}
            uni = {"five_sets": geo_relative(klmean, KL_SETS),
                   "real_sets": geo_relative(klmean, KL_SETS[:4]),
                   "new_domains": geo_relative(klmean, NEW_DOMAINS)}
            uni["p_best_five"] = ranking_probability(kls, KL_SETS, rng)
            uni["p_best_real"] = ranking_probability(kls, KL_SETS[:4], rng)
            uni["p_best_new"] = ranking_probability(kls, NEW_DOMAINS, rng)
            W = rng.dirichlet(np.ones(len(KL_SETS)), size=5000)
            srcs = [s for s in klmean if all(t in klmean[s] for t in KL_SETS)]
            if srcs:
                L = np.log(np.array([[klmean[s][t] for t in KL_SETS] for s in srcs]))
                L = L - L.min(axis=0)
                wins = (L @ W.T).argmin(axis=0)
                uni["p_best_dirichlet"] = {srcs[i]: float((wins == i).mean()) for i in range(len(srcs))}
            res["uniform"][key] = uni

            # BoolQ against majority and chance
            res["boolq"][key] = {s: {k: per_src[s][k] for k in ("boolq", "acc7", "acc6", "accnorm")
                                     if k in per_src[s]} for s in per_src}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}: {len(res['seeds'])} model-ratio settings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
