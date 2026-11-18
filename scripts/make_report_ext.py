#!/usr/bin/env python3
"""Tables and figures of the extended evaluation.

    uv run --no-sync python scripts/make_report_ext.py [--records records/extended]
        [--main results] [--results results/extended] [--only TABLE,...]

Reads the run tables of the main grid (<main>/runs.csv, dense.csv) and of the
extended evaluation (<results>/runs.csv, dense.csv, written by aggregate.py), the
paired statistics (<results>/run_stats.json, written by run_stats.py), and the run
directories and analyses under <records>. Writes the LaTeX table bodies to
<results>/tables/ and the PDF figures to <results>/figures/. Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_report as MR  # noqa: E402  (also fixes the PDF metadata, for reproducible files)

SOURCES = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen", "selfdoc"]
SOURCE_LABEL = {"wikitext2": "WikiText-2", "c4": "C4", "slimpajama": "SlimPajama",
                "alpaca": "Alpaca", "selfgen": "Self (seq.)", "selfdoc": "Self (doc.)",
                "alpaca-plain": "Alpaca, plain", "alpaca-resp": "Alpaca, responses",
                "wikitext2-art": "WikiText-2", "mix": "SlimPajama + Self (doc.)"}
MODEL_LABEL = {"opt-1.3b": "OPT-1.3B", "llama-3.2-1b": "Llama-3.2-1B", "qwen3-1.7b": "Qwen3-1.7B",
               "llama-3.1-8b": "Llama-3.1-8B", "opt-125m": "OPT-125M",
               "llama-3.2-1b-it": "Llama-3.2-1B-Instruct"}
MAIN = ["opt-1.3b", "llama-3.2-1b", "qwen3-1.7b"]
TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa", "boolq"]
# Set by main(): inputs under the records, outputs under the results.
RECORDS = "records/extended"
TAB = "results/extended/tables"
FIG = "results/extended/figures"


def rtag(r: float) -> str:
    return f"r{int(round(r * 100))}"


def fmt(x, nd=1, bold=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) >= 1000:
        s = f"{x:,.0f}".replace(",", "{,}")
    elif abs(x) >= 100:
        s = f"{x:.0f}"
    else:
        s = f"{x:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write(name: str, text: str) -> None:
    os.makedirs(TAB, exist_ok=True)
    path = os.path.join(TAB, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


def pick(df, model, run):
    rows = df[(df.model == model) & (df.run == run) & (df.done)]
    return None if rows.empty else rows.iloc[0]


# ----------------------------------------------------------------------------
# first token (main grid)
# ----------------------------------------------------------------------------

def table_bosall(main_grid: pd.DataFrame, dense: pd.DataFrame) -> None:
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\textbf{Dense} & \\textbf{Trained, range over sources} \\\\",
             "\\midrule"]
    for m in MAIN + ["llama-3.1-8b"]:
        d = dense[dense.model == m].iloc[0]
        dr = d.pplstream_wikitext2 / d.ppl_wikitext2
        for r in (0.3, 0.5):
            vals = []
            for s in SOURCES:
                x = pick(main_grid, m, f"proj-{s}-{rtag(r)}-s0")
                if x is not None:
                    vals.append(x.final_pplstream_wikitext2 / x.final_ppl_wikitext2)
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            rng = f"{fmt(lo, 2)} to {fmt(hi, 2)}"
            if m == "qwen3-1.7b":
                rng = "1.00 (no BOS token)"
            lines.append(f"{MODEL_LABEL[m]} & $-{int(r * 100)}\\%$ & {fmt(dr, 2)} & {rng} \\\\")
    write("bosall.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# three seeds, full grid
# ----------------------------------------------------------------------------

def _seed_rows(ext, model, source, ratio, tag="", seeds=(0, 1, 2)):
    rows = [pick(ext, model, f"proj-{source}-{rtag(ratio)}-s{s}{tag}") for s in seeds]
    return [r for r in rows if r is not None]


def acc6(row, prefix="final_"):
    return float(np.mean([row[f"{prefix}zs_{t}_acc"] for t in TASKS if t != "boolq"]))


def ms(vals, nd=1, scale=1.0):
    """mean (sd) of a list, formatted."""
    a = np.array(vals, dtype=float) * scale
    if len(a) == 0:
        return "n/a", float("nan")
    m = a.mean()
    if len(a) == 1:
        return fmt(m, nd), m
    sd = a.std(ddof=1)
    return f"{fmt(m, nd)}\\,({fmt(sd, nd) if sd < 100 else fmt(sd, 0)})", m


def table_seeds(ext, ratio):
    """Mean (sd) over three seeds: WikiText-2, C4, SlimPajama perplexity and accuracy."""
    metrics = [("final_ppl_wikitext2", "Wiki", 1, 1.0, "min"), ("final_ppl_c4", "C4", 1, 1.0, "min"),
               ("final_sp_geo", "SP", 1, 1.0, "min"), ("acc6", "Acc", 1, 100.0, "max")]
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{MODEL_LABEL[m]}}}}}" for m in MAIN)
    lines = [f"\\textbf{{Calibration}} & {head} \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3)),
             " & " + " & ".join(" & ".join(f"\\textit{{{lab}}}" for _, lab, *_ in metrics) for _ in MAIN) + " \\\\",
             "\\midrule"]
    cells, means = {}, {}
    for m in MAIN:
        for s in SOURCES:
            rows = _seed_rows(ext, m, s, ratio)
            for key, lab, nd, sc, _ in metrics:
                vals = [acc6(r) if key == "acc6" else r[key] for r in rows]
                cells[(m, s, lab)], means[(m, s, lab)] = ms(vals, nd, sc)
    for s in SOURCES:
        row = [SOURCE_LABEL[s]]
        for m in MAIN:
            for key, lab, nd, sc, best in metrics:
                col = [means[(m, x, lab)] for x in SOURCES if not math.isnan(means[(m, x, lab)])]
                b = (min(col) if best == "min" else max(col)) if col else None
                c = cells[(m, s, lab)]
                row.append(f"\\textbf{{{c}}}" if b is not None and means[(m, s, lab)] == b else c)
        lines.append(" & ".join(row) + " \\\\")
    write(f"seeds_{rtag(ratio)}.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# initialization against the documented LSP choices (Appendix A)
# ----------------------------------------------------------------------------

LSP_INIT = {"opt-125m": {30: 582, 50: 664, 70: 1538}, "opt-1.3b": {30: 26.1, 50: 74.4, 70: 814}}
INIT_VARIANTS = [("base", "As in this study (square layers: input side)"),
                 ("ridge", "Ridge $0.01\\bar{g}$ on output Gram matrices"),
                 ("fp32", "Statistics and factorizations in fp32"),
                 ("nobos", "Calibration windows without BOS"),
                 ("sqout", "Square layers on their output side"),
                 ("sqout-ridge", "Output side and ridge")]


def table_initcmp(ext):
    lines = ["\\textbf{Initialization} & \\multicolumn{3}{c}{\\textbf{OPT-125M}} & "
             "\\multicolumn{3}{c}{\\textbf{OPT-1.3B}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             " & $-30\\%$ & $-50\\%$ & $-70\\%$ & $-30\\%$ & $-50\\%$ & $-70\\%$ \\\\",
             "\\midrule"]
    for v, label in INIT_VARIANTS:
        cells = []
        for m in ("opt-125m", "opt-1.3b"):
            for r in (30, 50, 70):
                x = pick(ext, m, f"initcmp-{v}-r{r}")
                cells.append(fmt(x.init_pplstream_wikitext2, 1) if x is not None else "n/a")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    pub = [fmt(LSP_INIT[m][r], 1) for m in ("opt-125m", "opt-1.3b") for r in (30, 50, 70)]
    lines.append("Published (LSP) & " + " & ".join(pub) + " \\\\")
    write("initcmp.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# helpers on rows of the extended runs
# ----------------------------------------------------------------------------

KLSETS = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen"]
NEWDOM = ["dolly", "gsm8k", "pubmedqa", "mbpp", "opus", "pg19"]
NEWDOM_LABEL = {"dolly": "Dolly", "gsm8k": "GSM8K", "pubmedqa": "PubMed", "mbpp": "MBPP",
                "opus": "OPUS", "pg19": "PG-19"}


def row_metrics(x, prefix="final_"):
    """Wiki, C4, SP perplexity; accuracies (7 tasks, 6 without BoolQ); KLs; xppl."""
    out = {"wiki": x[f"{prefix}ppl_wikitext2"], "c4": x[f"{prefix}ppl_c4"],
           "sp": x.get(f"{prefix}sp_geo", float("nan"))}
    if f"{prefix}zs_boolq_acc" in x and not pd.isna(x[f"{prefix}zs_boolq_acc"]):
        out["acc7"] = 100 * float(np.mean([x[f"{prefix}zs_{t}_acc"] for t in TASKS]))
        out["acc6"] = 100 * float(np.mean([x[f"{prefix}zs_{t}_acc"] for t in TASKS if t != "boolq"]))
        out["boolq"] = 100 * x[f"{prefix}zs_boolq_acc"]
    for t in KLSETS + NEWDOM:
        k = f"{prefix}kl_{t}"
        if k in x and not pd.isna(x[k]):
            out[f"kl_{t}"] = x[k]
    return out


def geo_rel(kl: dict, sets):
    srcs = [s for s in kl if all(f"kl_{t}" in kl[s] for t in sets)]
    if not srcs:
        return {}
    best = {t: min(kl[s][f"kl_{t}"] for s in srcs) for t in sets}
    return {s: (math.exp(np.mean([math.log(kl[s][f"kl_{t}"] / best[t]) for t in sets])),
                max(kl[s][f"kl_{t}"] / best[t] for t in sets)) for s in srcs}


def seed_mean_metrics(ext, model, source, ratio, tag="", seeds=(0, 1, 2)):
    rows = _seed_rows(ext, model, source, ratio, tag, seeds)
    if not rows:
        return None
    ms_ = [row_metrics(r) for r in rows]
    keys = set.intersection(*[set(m) for m in ms_])
    return {k: float(np.mean([m[k] for m in ms_])) for k in keys} | {"n": len(rows)}


# ----------------------------------------------------------------------------
# uniform fidelity: which evaluation sets, which weights
# ----------------------------------------------------------------------------

def table_uniform(ext, stats):
    """Geometric-mean relative KL of Self (doc.) and of the best other source, on the five
    held-out sets, on the four real ones, and on the six independent domains; P(best)."""
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\multicolumn{2}{c}{\\textbf{Five sets}} & "
             "\\multicolumn{2}{c}{\\textbf{Four real sets}} & \\multicolumn{2}{c}{\\textbf{Six new domains}} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
             " & & \\textit{Self (doc.)} & \\textit{Best other} & \\textit{Self (doc.)} & \\textit{Best other} "
             "& \\textit{Self (doc.)} & \\textit{Best other} \\\\",
             "\\midrule"]
    for m in MAIN + ["llama-3.1-8b"]:
        for r in (0.3, 0.5):
            per = {s: seed_mean_metrics(ext, m, s, r) for s in SOURCES}
            per = {s: v for s, v in per.items() if v}
            if "selfdoc" not in per:
                continue
            cells = []
            for sets in (KLSETS, KLSETS[:4], NEWDOM):
                g = geo_rel(per, sets)
                if "selfdoc" not in g:
                    cells += ["n/a", "n/a"]
                    continue
                others = {s: v for s, v in g.items() if s != "selfdoc"}
                bo = min(others, key=lambda s: others[s][0])
                sd = g["selfdoc"][0]
                win = sd <= others[bo][0]
                cells.append(fmt(sd, 2, bold=win))
                cells.append(f"{fmt(others[bo][0], 2, bold=not win)} ({SOURCE_LABEL[bo]})")
            lines.append(f"{MODEL_LABEL[m]} & $-{int(r * 100)}\\%$ & " + " & ".join(cells) + " \\\\")
    write("uniform.tex", "\n".join(lines) + "\n")


def table_newdomains(ext, ratio=0.5):
    """Divergence on the six independent domains relative to the best source (seed means)."""
    lines = ["\\textbf{Calibration} & " + " & ".join(f"\\textbf{{{MODEL_LABEL[m]}}}" for m in MAIN)
             + " \\\\", "\\midrule"]
    vals = {}
    for m in MAIN:
        per = {s: seed_mean_metrics(ext, m, s, ratio) for s in SOURCES}
        per = {s: v for s, v in per.items() if v}
        vals[m] = geo_rel(per, NEWDOM)
    for s in SOURCES:
        row = [SOURCE_LABEL[s]]
        for m in MAIN:
            g = vals[m]
            if s not in g:
                row.append("n/a")
                continue
            best = min(v[0] for v in g.values())
            row.append(f"{fmt(g[s][0], 2, bold=g[s][0] == best)} / {fmt(g[s][1], 2)}")
        lines.append(" & ".join(row) + " \\\\")
    write(f"newdomains_{rtag(ratio)}.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# accuracy: paired differences and BoolQ
# ----------------------------------------------------------------------------

def _ci(d):
    if not d:
        return "n/a"
    return f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


def table_paired(stats):
    """Alpaca against the runner-up, and WikiText-2 against the next-lowest real source:
    differences of the 6-task and 7-task means, 95% paired bootstrap intervals."""
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\multicolumn{2}{c}{\\textbf{Alpaca minus runner-up}} & "
             "\\multicolumn{2}{c}{\\textbf{WikiText-2 minus next real source}} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             " & & \\textit{6 tasks} & \\textit{7 tasks} & \\textit{6 tasks} & \\textit{7 tasks} \\\\",
             "\\midrule"]
    for m in MAIN + ["llama-3.1-8b"]:
        for r in (0.3, 0.5):
            key = f"{m}/{rtag(r)}"
            seeds = stats.get("seeds", {}).get(key)
            pa = stats.get("paired_acc", {}).get(key)
            if not seeds or not pa:
                continue
            acc6 = {s: v["acc6"][0] for s, v in seeds.items() if "acc6" in v}
            others = {s: a for s, a in acc6.items() if s != "alpaca"}
            ru = max(others, key=others.get) if others else None
            real = {s: a for s, a in acc6.items() if s in ("c4", "slimpajama", "alpaca")}
            nw = min(real, key=real.get) if real else None
            a = pa.get(f"alpaca-{ru}", {})
            w = pa.get(f"wikitext2-{nw}", {})
            lines.append(f"{MODEL_LABEL[m]} & $-{int(r * 100)}\\%$ & {_ci(a.get('acc6'))} ({SOURCE_LABEL.get(ru, '')}) "
                         f"& {_ci(a.get('acc7'))} & {_ci(w.get('acc6'))} ({SOURCE_LABEL.get(nw, '')}) & {_ci(w.get('acc7'))} \\\\")
    write("paired.tex", "\n".join(lines) + "\n")


def table_boolq(ext):
    """BoolQ accuracy of the compressed models against its majority-class rate, and the
    source ranked first by each accuracy summary (seed means)."""
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\textbf{BoolQ range} & \\textbf{Below 62.2\\%} & "
             "\\textbf{Best, 7 tasks} & \\textbf{Best, 6 tasks} & \\textbf{Best, chance-norm.} \\\\",
             "\\midrule"]
    for m in MAIN + ["llama-3.1-8b"]:
        for r in (0.3, 0.5):
            per = {s: seed_mean_metrics(ext, m, s, r) for s in SOURCES}
            per = {s: v for s, v in per.items() if v and "acc7" in v}
            if not per:
                continue
            rows = {s: _seed_rows(ext, m, s, r) for s in per}
            norm = {s: 100 * float(np.mean([np.mean([(x[f"final_zs_{t}_acc"] - CHANCE[t]) / (1 - CHANCE[t])
                                                     for t in TASKS]) for x in rows[s]])) for s in per}
            bq = [v["boolq"] for v in per.values()]
            below = sum(b < 62.17 for b in bq)
            b7 = max(per, key=lambda s: per[s]["acc7"])
            b6 = max(per, key=lambda s: per[s]["acc6"])
            bn = max(norm, key=norm.get)
            lines.append(f"{MODEL_LABEL[m]} & $-{int(r * 100)}\\%$ & {fmt(min(bq))}--{fmt(max(bq))} & "
                         f"{below} of {len(bq)} & {SOURCE_LABEL[b7]} & {SOURCE_LABEL[b6]} & {SOURCE_LABEL[bn]} \\\\")
    write("boolq.tex", "\n".join(lines) + "\n")


CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "hellaswag": 0.25,
          "winogrande": 0.5, "openbookqa": 0.25, "boolq": 0.5}


# ----------------------------------------------------------------------------
# controls: matched segments, Alpaca templates, mixture
# ----------------------------------------------------------------------------

def _ctrl_line(label, x, kl_cols=None):
    if x is None:
        return f"{label} & " + " & ".join(["n/a"] * (4 + len(kl_cols or []))) + " \\\\"
    mm = row_metrics(x)
    cells = [fmt(mm["wiki"]), fmt(mm["c4"]), fmt(mm["sp"]), fmt(mm.get("acc6", float("nan")))]
    for t in kl_cols or []:
        cells.append(fmt(mm.get(f"kl_{t}", float("nan")), 3))
    return f"{label} & " + " & ".join(cells) + " \\\\"


def table_format(ext):
    """Default packing against the matched-segment format, -50%, seed 0."""
    head = ["\\textbf{Model} & \\textbf{Calibration} & \\textbf{Format} & \\textit{Wiki} & \\textit{C4} & "
            "\\textit{SP} & \\textit{Acc} & \\textit{KL Wiki} & \\textit{KL C4} & \\textit{KL SP} & \\textit{KL Alpaca} \\\\",
            "\\midrule"]
    lines = list(head)
    for m in MAIN:
        for s, sseg in (("wikitext2", "wikitext2-art"), ("c4", "c4"), ("slimpajama", "slimpajama"),
                        ("alpaca", "alpaca"), ("selfdoc", "selfdoc")):
            for fmt_label, run in (("packed", f"proj-{s}-r50-s0"), ("segments", f"proj-{sseg}-r50-s0-seg127")):
                x = pick(ext, m, run)
                line = _ctrl_line(f"{MODEL_LABEL[m]} & {SOURCE_LABEL[s]} & {fmt_label}", x,
                                  ["wikitext2", "c4", "slimpajama", "alpaca"])
                lines.append(line)
        lines.append("\\midrule")
    write("format.tex", "\n".join(lines[:-1]) + "\n")


def table_sqout(ext):
    """Square attention-output layers projected on their input side (the main experiments)
    or on their output side, -50%, seed 0: untrained and trained."""
    lines = ["\\textbf{Model} & \\textbf{Calibration} & \\textbf{Side} & \\textit{Untrained Wiki} & "
             "\\textit{Wiki} & \\textit{C4} & \\textit{SP} & \\textit{Acc} \\\\", "\\midrule"]
    for m in ("llama-3.2-1b", "opt-1.3b"):
        for s in ("wikitext2", "c4", "selfdoc"):
            for side, run in (("input", f"proj-{s}-r50-s0"), ("output", f"proj-{s}-r50-s0-sqout")):
                x = pick(ext, m, run)
                if x is None:
                    continue
                mm = row_metrics(x)
                lines.append(f"{MODEL_LABEL[m]} & {SOURCE_LABEL[s]} & {side} & {fmt(x['init_ppl_wikitext2'])} & "
                             f"{fmt(mm['wiki'])} & {fmt(mm['c4'])} & {fmt(mm['sp'])} & {fmt(mm['acc6'])} \\\\")
        lines.append("\\midrule")
    write("sqout.tex", "\n".join(lines[:-1]) + "\n")


def table_alpaca(ext):
    """Alpaca with its template, without it, and its responses alone."""
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\textbf{Calibration} & \\textit{Wiki} & \\textit{C4} & "
             "\\textit{SP} & \\textit{Acc (6)} & \\textit{Acc (7)} \\\\", "\\midrule"]
    for m in MAIN:
        for r in (0.3, 0.5):
            for s in ("alpaca", "alpaca-plain", "alpaca-resp", "c4", "slimpajama"):
                x = pick(ext, m, f"proj-{s}-{rtag(r)}-s0")
                if x is None:
                    continue
                mm = row_metrics(x)
                lines.append(f"{MODEL_LABEL[m]} & $-{int(r * 100)}\\%$ & {SOURCE_LABEL[s]} & {fmt(mm['wiki'])} & "
                             f"{fmt(mm['c4'])} & {fmt(mm['sp'])} & {fmt(mm['acc6'])} & {fmt(mm['acc7'])} \\\\")
            lines.append("\\midrule")
    write("alpaca.tex", "\n".join(lines[:-1]) + "\n")


def table_mix(ext):
    """SlimPajama, self-generated documents, and their half-and-half mixture (seed 0),
    with the geometric-mean relative KL over the five held-out sets and the six new
    domains, relative to the best of the six main sources and the mixture."""
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\textbf{Calibration} & \\textit{Wiki} & \\textit{C4} & "
             "\\textit{SP} & \\textit{Acc} & \\textit{Rel. KL, 5 sets} & \\textit{Rel. KL, new} \\\\", "\\midrule"]
    for m in MAIN:
        for r in (0.3, 0.5):
            per = {s: pick(ext, m, f"proj-{s}-{rtag(r)}-s0") for s in SOURCES}
            per["mix"] = pick(ext, m, f"proj-mix-sp-selfdoc-{rtag(r)}-s0")
            met = {s: row_metrics(x) for s, x in per.items() if x is not None}
            g5, gn = geo_rel(met, KLSETS), geo_rel(met, NEWDOM)
            for s in ("slimpajama", "selfdoc", "mix"):
                if s not in met:
                    continue
                mm = met[s]
                lines.append(f"{MODEL_LABEL[m]} & $-{int(r * 100)}\\%$ & {SOURCE_LABEL[s]} & {fmt(mm['wiki'])} & "
                             f"{fmt(mm['c4'])} & {fmt(mm['sp'])} & {fmt(mm.get('acc6', float('nan')))} & "
                             f"{fmt(g5.get(s, (float('nan'),))[0], 2)} & {fmt(gn.get(s, (float('nan'),))[0], 2)} \\\\")
            lines.append("\\midrule")
    write("mix.tex", "\n".join(lines[:-1]) + "\n")


def table_qwen(ext):
    """Self-generated calibration of Qwen3-1.7B and Llama-3.2-1B at -30%: exact samples,
    exact samples restarted every 512 tokens, and documents of four length limits."""
    variants = [("selfgen", "proj-selfgen-r30-s0", "Exact, 2048 tokens"),
                ("selfpack", "proj-selfpack512-r30-s0", "Exact, restarted every 512"),
                ("M128", "proj-selfdoc-r30-s0-M128", "Documents, at most 128"),
                ("M512", "proj-selfdoc-r30-s0", "Documents, at most 512"),
                ("M1024", "proj-selfdoc-r30-s0-M1024", "Documents, at most 1024"),
                ("M2048", "proj-selfdoc-r30-s0-M2048", "Documents, at most 2048")]
    lines = ["\\textbf{Model} & \\textbf{Self-generated text} & \\textit{Wiki} & \\textit{C4} & \\textit{SP} & "
             "\\textit{Acc} & \\textit{KL Self} & \\textit{KL C4} \\\\", "\\midrule"]
    for m in ("qwen3-1.7b", "llama-3.2-1b"):
        for _, run, label in variants:
            x = pick(ext, m, run)
            if x is None:
                continue
            mm = row_metrics(x)
            lines.append(f"{MODEL_LABEL[m]} & {label} & {fmt(mm['wiki'])} & {fmt(mm['c4'])} & {fmt(mm['sp'])} & "
                         f"{fmt(mm.get('acc6', float('nan')))} & {fmt(mm.get('kl_selfgen', float('nan')), 3)} & "
                         f"{fmt(mm.get('kl_c4', float('nan')), 3)} \\\\")
        lines.append("\\midrule")
    write("qwen.tex", "\n".join(lines[:-1]) + "\n")


def table_lora(ext):
    """SVD-LLM with LoRA recovery against LSP, -30%, seed 0, per source."""
    lines = ["\\textbf{Calibration} & " + " & ".join(
        f"\\multicolumn{{4}}{{c}}{{\\textbf{{{MODEL_LABEL[m]}}}}}" for m in ("llama-3.2-1b", "opt-1.3b")) + " \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        " & " + " & ".join("\\textit{Wiki} & \\textit{C4} & \\textit{SP} & \\textit{Acc}" for _ in range(2)) + " \\\\",
        "\\midrule"]
    for method, prefix in (("SVD-LLM + LoRA", "lora"), ("LSP", "proj")):
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{method}}}}} \\\\")
        for s in SOURCES:
            cells = []
            for m in ("llama-3.2-1b", "opt-1.3b"):
                x = pick(ext, m, f"{prefix}-{s}-r30-s0")
                if x is None:
                    cells += ["n/a"] * 4
                    continue
                mm = row_metrics(x)
                cells += [fmt(mm["wiki"]), fmt(mm["c4"]), fmt(mm["sp"]), fmt(mm.get("acc6", float("nan")))]
            lines.append(f"{SOURCE_LABEL[s]} & " + " & ".join(cells) + " \\\\")
    write("lora.tex", "\n".join(lines) + "\n")


def table_scale(ext, dense_ext):
    """Llama-3.1-8B: -30% (mean over three seeds) and -50% (one seed); bold: best per column."""
    lines = ["\\textbf{Calibration} & \\multicolumn{5}{c}{\\textbf{$-30\\%$, three seeds}} & "
             "\\multicolumn{5}{c}{\\textbf{$-50\\%$, one seed}} \\\\",
             "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
             " & " + " & ".join("\\textit{Wiki} & \\textit{C4} & \\textit{SP} & \\textit{Acc} & \\textit{Rel. KL}"
                                for _ in range(2)) + " \\\\", "\\midrule"]
    m = "llama-3.1-8b"
    vals = {}   # (ratio, source) -> [wiki, c4, sp, acc6, rel KL]
    for r, seeds in ((0.3, (0, 1, 2)), (0.5, (0,))):
        per = {s: seed_mean_metrics(ext, m, s, r, seeds=seeds) for s in SOURCES}
        per = {s: v for s, v in per.items() if v}
        g = geo_rel(per, KLSETS)
        for s, v in per.items():
            vals[(r, s)] = [v["wiki"], v["c4"], v["sp"], v.get("acc6", float("nan")),
                            g.get(s, (float("nan"),))[0]]
    best = {}
    for r in (0.3, 0.5):
        col = [vals[(r, s)] for s in SOURCES if (r, s) in vals]
        for j, better in enumerate((min, min, min, max, min)):
            xs = [c[j] for c in col if not math.isnan(c[j])]
            best[(r, j)] = better(xs) if xs else None
    for s in SOURCES:
        cells = []
        for r in (0.3, 0.5):
            if (r, s) not in vals:
                cells += ["n/a"] * 5
                continue
            for j, (x, nd) in enumerate(zip(vals[(r, s)], (1, 1, 1, 1, 2))):
                cells.append(fmt(x, nd, bold=(best[(r, j)] is not None and fmt(x, nd) == fmt(best[(r, j)], nd))))
        lines.append(f"{SOURCE_LABEL[s]} & " + " & ".join(cells) + " \\\\")
    write("scale.tex", "\n".join(lines) + "\n")


def figure_domains_8b(ext, dense_ext):
    """Perplexity on the SlimPajama domains relative to the dense model, Llama-3.1-8B at
    -30%, all six sources, geometric mean over the three seeds (redraws domains_8b.pdf
    with the figure code of the original report)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    m, rows = "llama-3.1-8b", []
    for s in SOURCES:
        xs = [pick(ext, m, f"proj-{s}-r30-s{k}") for k in (0, 1, 2)]
        xs = [x for x in xs if x is not None]
        if len(xs) < 3:
            raise RuntimeError(f"{s}: {len(xs)} of 3 seeds")
        row = {"model": m, "run": f"proj-{s}-r30-s0", "done": True}
        for dom in MR.SP_DOMAINS:
            row[f"final_sp_{dom}"] = float(np.exp(np.mean([np.log(x[f"final_sp_{dom}"]) for x in xs])))
        rows.append(row)
    dense = dense_ext[dense_ext.model == m]
    MR.fig_domains(pd.DataFrame(rows), dense, os.path.join(FIG, "domains_8b.pdf"), ratio=0.3,
                   models=[m], sources=SOURCES, figsize=(3.6, 2.6))


def table_optim(ext, main_grid=None):
    """Llama-3.2-1B, -50%, seed 0: validation KL of each source at three learning rates
    (the main rate is 1e-4), and the trained model after 8 (main) and 24 epochs; then the
    crossover of initialization statistics and training windows at 8 and 24 epochs."""
    lines = ["\\textbf{Calibration} & \\multicolumn{3}{c}{\\textbf{Validation KL}} & "
             "\\multicolumn{4}{c}{\\textbf{8 epochs}} & \\multicolumn{4}{c}{\\textbf{24 epochs}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-12}",
             " & $3{\\times}10^{-5}$ & $10^{-4}$ & $3{\\times}10^{-4}$ & \\textit{Wiki} & \\textit{C4} & \\textit{SP} & "
             "\\textit{Acc} & \\textit{Wiki} & \\textit{C4} & \\textit{SP} & \\textit{Acc} \\\\", "\\midrule"]
    m = "llama-3.2-1b"

    def cells(x):
        if x is None:
            return ["n/a"] * 4
        mm = row_metrics(x)
        return [fmt(mm["wiki"]), fmt(mm["c4"]), fmt(mm["sp"]), fmt(mm.get("acc6", float("nan")))]

    for s in SOURCES:
        main0 = pick(ext, m, f"proj-{s}-r50-s0")
        vals = [pick(ext, m, f"proj-{s}-r50-s0-lr3e-5"), main0, pick(ext, m, f"proj-{s}-r50-s0-lr3e-4")]
        kls = [x["best_val_kl"] if x is not None else float("nan") for x in vals]
        best = min(range(3), key=lambda i: kls[i] if not math.isnan(kls[i]) else float("inf"))
        kcells = [fmt(k, 3, bold=(i == best)) for i, k in enumerate(kls)]
        lines.append(f"{SOURCE_LABEL[s]} & " + " & ".join(kcells + cells(main0) +
                                                         cells(pick(ext, m, f"proj-{s}-r50-s0-e24"))) + " \\\\")
    lines.append("\\midrule")
    for s, init, label in (("wikitext2", "slimpajama", "WikiText-2, SlimPajama statistics"),
                           ("slimpajama", "wikitext2", "SlimPajama, WikiText-2 statistics")):
        x8 = pick(main_grid, m, f"proj-{s}-r50-s0-init{init}") if main_grid is not None else None
        x24 = pick(ext, m, f"proj-{s}-r50-s0-init{init}-e24")
        lines.append(f"{label} & -- & -- & -- & " + " & ".join(cells(x8) + cells(x24)) + " \\\\")
    write("optim.tex", "\n".join(lines) + "\n")


def table_behavior(runs_dir=None):
    runs_dir = runs_dir or os.path.join(RECORDS, "runs")
    """Open-ended generation, long context and generative tasks, seed 0, -30%."""
    lines = ["\\textbf{Model} & \\textbf{Calibration} & \\textit{Gen. NLL} & \\textit{Rep. (greedy)} & "
             "\\textit{PPL 4k--8k} & \\textit{Passkey 8k} & \\textit{GSM8K} & \\textit{NQ} \\\\", "\\midrule"]
    for m in MAIN + ["llama-3.1-8b"]:
        for run, label in [("dense", "Dense")] + [(f"proj-{s}-r30-s0", SOURCE_LABEL[s]) for s in SOURCES]:
            p = os.path.join(runs_dir, m, run, "behave.json")
            if not os.path.exists(p):
                continue
            b = json.load(open(p))
            og = b.get("open_gen", {})
            lc = b.get("long_ctx", {}).get("ppl_by_position", {})
            pk = b.get("passkey", {})
            tk = b.get("tasks", {})
            # the last number of the answer: strict matching also scores the answer's
            # format, which compressed models rarely keep
            gsm = tk.get("gsm8k", {}).get("exact_match,flexible-extract")
            nq = tk.get("nq_open", {}).get("exact_match,remove_whitespace")
            far = [v for k, v in lc.items() if k.startswith("4096")]
            lines.append(f"{MODEL_LABEL[m]} & {label} & {fmt(og.get('sample', {}).get('teacher_nll', float('nan')), 2)} & "
                         f"{fmt(og.get('greedy', {}).get('rep4', float('nan')), 2)} & "
                         f"{fmt(far[0], 1) if far else 'n/a'} & "
                         # 40 trials: multiples of 2.5%, rounded half up (not to even)
                         f"{int(100 * pk['8000'] + 0.5) if '8000' in pk else 'n/a'} & "
                         f"{fmt(100 * gsm, 1) if gsm is not None else 'n/a'} & "
                         f"{fmt(100 * nq, 1) if nq is not None else 'n/a'} \\\\")
        lines.append("\\midrule")
    write("behavior.tex", "\n".join(lines[:-1]) + "\n")


def table_bench(analysis_dir=None):
    analysis_dir = analysis_dir or os.path.join(RECORDS, "analysis")
    lines = ["\\textbf{Model} & \\textbf{Version} & \\textit{Weights (GB)} & \\textit{Prefill, b1 (ms)} & "
             "\\textit{Prefill, b8 (ms)} & \\textit{Decode, b1 (tok/s)} & \\textit{Decode, b16 (tok/s)} & "
             "\\textit{Peak, b16 (GB)} \\\\", "\\midrule"]
    for m in MAIN + ["llama-3.1-8b"]:
        p = os.path.join(analysis_dir, f"bench-{m}.json")
        if not os.path.exists(p):
            continue
        b = json.load(open(p))
        for key, label in (("dense", "Dense"), ("r30", "$-30\\%$"), ("r50", "$-50\\%$")):
            if key not in b:
                continue
            x = b[key]
            lines.append(f"{MODEL_LABEL[m]} & {label} & {fmt(x['weights_gb'], 2)} & {fmt(x['prefill_ms_b1'], 1)} & "
                         f"{fmt(x['prefill_ms_b8'], 1)} & {fmt(x['decode_tok_s_b1'], 0)} & "
                         f"{fmt(x['decode_tok_s_b16'], 0)} & {fmt(x['peak_gb_b16'], 2)} \\\\")
        lines.append("\\midrule")
    write("bench.tex", "\n".join(lines[:-1]) + "\n")


def figure_curves(runs_dir=None):
    runs_dir = runs_dir or os.path.join(RECORDS, "runs")
    """Training and validation KL along the 512 steps, for 64 to 4096 distinct windows,
    C4 and self-generated documents (Llama-3.2-1B, -50%). Validation windows are each
    run's own (disjoint documents of its source), so the curves compare training with
    validation within a run; the third panel compares runs on the common held-out sets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    ns = [64, 256, 1024, 4096]
    allns = [64, 128, 256, 512, 1024, 4096]
    colors = {64: "#9ec5f4", 256: "#5598e7", 1024: "#1c5cab", 4096: "#0b0b0b"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, s, title in zip(axes[:2], ("c4", "selfdoc"), ("C4", "Self (doc.)")):
        for n in ns:
            run = f"proj-{s}-r50-s0" if n == 512 else f"proj-{s}-r50-s0-n{n}"
            p = os.path.join(runs_dir, "llama-3.2-1b", run, "train_log.jsonl")
            if not os.path.exists(p):
                continue
            recs = [json.loads(x) for x in open(p) if x.strip()]
            steps = [r.get("steps", 64 * (r["epoch"] + 1)) for r in recs]
            ax.plot(steps, [r["val_kl"] for r in recs], color=colors[n], marker="o", ms=2.5, lw=1.4,
                    label=f"{n}")
            # the first epoch ramps the projectors in (alpha 0 -> 1): its training KL is
            # not comparable, so training curves start at the second epoch
            ax.plot(steps[1:], [r["train_kl"] for r in recs][1:], color=colors[n], ls=":", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("KL (nats/token)")
    axes[0].legend(title="windows", fontsize=6.5, title_fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    # common held-out sets: mean KL over the five held-out sets of every source
    ax = axes[2]
    for s, label, mk in (("c4", "C4", "s"), ("selfdoc", "Self (doc.)", "P")):
        xs, ys = [], []
        for n in allns:
            run = f"proj-{s}-r50-s0" if n == 512 else f"proj-{s}-r50-s0-n{n}"
            p = os.path.join(runs_dir, "llama-3.2-1b", run, "metrics.json")
            if not os.path.exists(p):
                continue
            met = json.load(open(p))
            if not met.get("done") or "final" not in met:
                continue
            kl = met["final"].get("kl", {})
            if all(t in kl for t in KLSETS):
                xs.append(n)
                ys.append(float(np.exp(np.mean([np.log(kl[t]) for t in KLSETS]))))
        ax.plot(xs, ys, marker=mk, lw=1.4, label=label,
                color={"c4": "#eb6834", "selfdoc": "#008300"}[s])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("distinct windows")
    ax.set_ylabel("held-out KL (geo. mean)")
    ax.set_title("Common held-out sets")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "curves.pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {FIG}/curves.pdf")


def table_composition(path=None):
    path = path or os.path.join(RECORDS, "analysis", "selfgen-composition.json")
    """What the self-generated text is made of, by position (exact samples) and over the
    documents: repetition, compressibility, scripts, English function words, code."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    cols = [("rep4", "Rep. 4-grams", 100, 1), ("gzip_ratio", "gzip ratio", 1, 2),
            ("script_cjk", "CJK", 100, 1), ("script_cyrillic", "Cyrillic", 100, 1),
            ("english", "English words", 100, 1), ("code", "Code lines", 100, 1),
            ("doc_starts", "Doc. starts", 1, 1)]
    lines = ["\\textbf{Model} & \\textbf{Text} & " + " & ".join(f"\\textit{{{c[1]}}}" for c in cols) + " \\\\",
             "\\midrule"]
    for m in MAIN:
        sg = d.get(f"{m}:selfgen", {})
        for block in ("0-512", "512-1024", "1024-1536", "1536-2048"):
            if block not in sg:
                continue
            x = sg[block]
            lines.append(f"{MODEL_LABEL[m]} & exact, tokens {block.replace('-', '--')} & " +
                         " & ".join(fmt(sc * x[k], nd) for k, _, sc, nd in cols) + " \\\\")
        sd = d.get(f"{m}:selfdoc", {}).get("documents")
        if sd:
            lines.append(f"{MODEL_LABEL[m]} & documents & " +
                         " & ".join(fmt(sc * sd[k], nd) for k, _, sc, nd in cols) + " \\\\")
        lines.append("\\midrule")
    write("composition.tex", "\n".join(lines[:-1]) + "\n")


INST_SOURCES = [("c4", "C4"), ("slimpajama", "SlimPajama"), ("alpaca-chat", "Alpaca, chat format"),
                ("selfdoc", "Self (doc.)"), ("selfresp", "Self (responses)")]


def table_inst(ext, dense_ext):
    """Llama-3.2-1B-Instruct compressed by 30%: IFEval (prompt- and instruction-level,
    strict), mean accuracy, WikiText-2 perplexity and divergence on Alpaca and Self."""
    m = "llama-3.2-1b-it"
    cols = ["final_gen_ifeval_prompt_level_strict_acc", "final_gen_ifeval_inst_level_strict_acc"]
    lines = ["\\textbf{Calibration} & \\textit{IFEval, prompt} & \\textit{IFEval, instruction} & "
             "\\textit{Acc} & \\textit{Wiki} & \\textit{KL Alpaca} & \\textit{KL Self} \\\\", "\\midrule"]
    if dense_ext is not None:
        d = dense_ext[dense_ext.model == m]
        if len(d):
            d = d.iloc[0]
            a6 = 100 * float(np.mean([d[f"zs_{t}_acc"] for t in TASKS if t != "boolq"]))
            lines.append(f"Dense & {fmt(100 * d['gen_ifeval_prompt_level_strict_acc'])} & "
                         f"{fmt(100 * d['gen_ifeval_inst_level_strict_acc'])} & {fmt(a6)} & "
                         f"{fmt(d['ppl_wikitext2'], 2)} & 0 & 0 \\\\")
            lines.append("\\midrule")
    rows = {s: pick(ext, m, f"proj-{s}-r30-s0") for s, _ in INST_SOURCES}
    best = {}
    for c in cols:
        vals = [r[c] for r in rows.values() if r is not None and not pd.isna(r.get(c, float("nan")))]
        best[c] = max(vals) if vals else None
    for s, label in INST_SOURCES:
        x = rows[s]
        if x is None:
            continue
        mm = row_metrics(x)
        cells = [fmt(100 * x[c], 1, bold=(x[c] == best[c])) for c in cols]
        lines.append(f"{label} & " + " & ".join(cells) + f" & {fmt(mm.get('acc6', float('nan')))} & "
                     f"{fmt(mm['wiki'], 1)} & {fmt(mm.get('kl_alpaca', float('nan')), 3)} & "
                     f"{fmt(mm.get('kl_selfgen', float('nan')), 3)} \\\\")
    write("inst.tex", "\n".join(lines) + "\n")


def main(argv=None) -> int:
    global RECORDS, TAB, FIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="records/extended", help="runs/ and analysis/ of the extended evaluation")
    ap.add_argument("--main", default="results", help="run tables of the main grid (runs.csv, dense.csv)")
    ap.add_argument("--results", default="results/extended",
                    help="run tables and run_stats.json of the extended evaluation; output root")
    ap.add_argument("--only", default=None, help="comma-separated table names")
    args = ap.parse_args(argv)
    RECORDS = args.records
    TAB, FIG = os.path.join(args.results, "tables"), os.path.join(args.results, "figures")
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    main_grid = pd.read_csv(os.path.join(args.main, "runs.csv"))
    dense = pd.read_csv(os.path.join(args.main, "dense.csv"))
    ext = pd.read_csv(os.path.join(args.results, "runs.csv"))
    dense_ext = pd.read_csv(os.path.join(args.results, "dense.csv"))
    with open(os.path.join(args.results, "run_stats.json")) as f:
        stats = json.load(f)
    todo = {"bosall": lambda: table_bosall(main_grid, dense),
            "seeds": lambda: [table_seeds(ext, r) for r in (0.3, 0.5)],
            "initcmp": lambda: table_initcmp(ext),
            "uniform": lambda: table_uniform(ext, stats),
            "newdomains": lambda: [table_newdomains(ext, r) for r in (0.3, 0.5)],
            "paired": lambda: table_paired(stats),
            "boolq": lambda: table_boolq(ext),
            "format": lambda: table_format(ext),
            "alpaca": lambda: table_alpaca(ext),
            "sqout": lambda: table_sqout(ext),
            "mix": lambda: table_mix(ext),
            "qwen": lambda: table_qwen(ext),
            "lora": lambda: table_lora(ext),
            "scale": lambda: table_scale(ext, None),
            "optim": lambda: table_optim(ext, main_grid),
            "domains8b": lambda: figure_domains_8b(ext, dense_ext),
            "behavior": lambda: table_behavior(),
            "bench": lambda: table_bench(),
            "curves": lambda: figure_curves(),
            "composition": lambda: table_composition(),
            "inst": lambda: table_inst(ext, dense_ext)}
    for name, fn in todo.items():
        if args.only and name not in args.only.split(","):
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
