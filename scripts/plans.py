"""Experiment plans: every job behind the paper's results.

Two sets of plans. The main grid and its ablations (plans without a prefix) write
their runs to out/runs/, as published in records/runs/. The extended evaluation
(plans prefixed ``ext-``: the three-seed repetition of the grid, the controls,
Llama-3.1-8B at both ratios, the instruction-tuned model, behavior and cost) writes
to out/extended/, as published in records/extended/; its runs record every
evaluation item, measure KL on 128 held-out windows per source and on six
independently chosen domains, and evaluate the last epoch as well as the selected
one.

A plan is a list of jobs; ``scripts/submit.py PLAN [PLAN ...]`` runs them, on Slurm
or locally (``--local``). A job is a dict with

    name       unique job name (also the Slurm job name)
    script     the entry point, e.g. scripts/compress.py
    args       its command-line arguments
    run_dir    for compress.py jobs: out/runs/<model>/<run> (out/extended/runs/... for
               the extended evaluation); or
    output     for other jobs: the file the job produces
    requires   files that must exist before the job can start (optional)
    time       Slurm time limit
    partitions Slurm partitions (see below)

The arguments of every job are exactly those of the published run in
``records/runs/<model>/<run>/config.json`` (``records/extended/runs/...``);
``tests/test_plans.py`` checks it.

Slurm placement is read from the environment, so the plans carry no
cluster-specific names:

    FSC_PARTITIONS       partitions for most jobs: bfloat16 GPUs with >= 40 GB
    FSC_PARTITIONS_80G   partitions for Llama-3.1-8B: >= 80 GB (default: FSC_PARTITIONS)
    FSC_EXCLUDE          nodes to avoid (optional)

Empty values let Slurm use its default partition.
"""

from __future__ import annotations

import os

PARTITIONS = os.environ.get("FSC_PARTITIONS", "")
PARTITIONS_80G = os.environ.get("FSC_PARTITIONS_80G", PARTITIONS)
EXCLUDE = os.environ.get("FSC_EXCLUDE", "")

MODELS = {
    "opt-125m": "facebook/opt-125m",
    "opt-1.3b": "facebook/opt-1.3b",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.2-1b-it": "meta-llama/Llama-3.2-1B-Instruct",
}

SOURCES = ("wikitext2", "c4", "slimpajama", "alpaca", "selfgen", "selfdoc")
MAIN_MODELS = ("opt-1.3b", "llama-3.2-1b", "qwen3-1.7b")
RATIOS = (0.3, 0.5)
# Projector learning rate per model, selected on the validation KL of C4 calibration
# (plan pilot-lr at -50%, Appendix B; plan scale at -30% for Llama-3.1-8B, Section 4.8).
LR = {"opt-125m": 1e-3, "opt-1.3b": 1e-4, "llama-3.2-1b": 1e-4, "qwen3-1.7b": 1e-4,
      "llama-3.1-8b": 3e-5, "llama-3.2-1b-it": 1e-4}
# The training budget of the main experiments: 512 windows of 2048 tokens, batches of
# 8 windows, 8 epochs (512 steps).
COMMON = ["--n-train", "512", "--n-val", "32", "--batch-size", "8", "--micro-batch", "2",
          "--epochs", "8", "--seqlen", "2048"]


def _r(ratio: float) -> str:
    return f"r{int(round(ratio * 100))}"


def _lr(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-")


def run_dir(model: str, name: str) -> str:
    return f"out/runs/{model}/{name}"


def compress_job(model: str, name: str, args: list[str], time: str = "0-04:00:00",
                 partitions: str | None = None, requires: list[str] | None = None) -> dict:
    return {
        "name": f"fsc-{model}-{name}",
        "script": "scripts/compress.py",
        "args": ["--model", MODELS[model]] + args,
        "run_dir": run_dir(model, name),
        "partitions": PARTITIONS if partitions is None else partitions,
        "time": time,
        "requires": requires or [],
    }


# ----------------------------------------------------------------------------
# Self-generated calibration sets
# ----------------------------------------------------------------------------

GEN_BATCH = {"opt-125m": 128, "opt-1.3b": 64, "llama-3.2-1b": 128, "qwen3-1.7b": 64,
             "llama-3.1-8b": 64, "llama-3.2-1b-it": 128}
DOC_BATCH = {"opt-125m": 256, "opt-1.3b": 128, "llama-3.2-1b": 256, "qwen3-1.7b": 128,
             "llama-3.1-8b": 128, "llama-3.2-1b-it": 256}


def selfgen_path(model: str, temperature: float = 1.0, seed: int = 0, n: int = 1088,
                 seqlen: int = 2048) -> str:
    """Exact samples of the dense model, 2048 tokens long unless stated (1088 = 1024 + 64
    windows)."""
    tag = f"selfgen-T{temperature:.1f}-s{seed}" + ("" if n == 1088 else f"-n{n}")
    tag += "" if seqlen == 2048 else f"-L{seqlen}"
    return f"out/calib/{model}/{tag}.pt"


def selfgen_eval_path(model: str) -> str:
    """Exact samples reserved for measuring KL (seed 1000); never used for calibration."""
    return selfgen_path(model, 1.0, 1000, 64)


def selfdoc_path(model: str, seed: int = 0, max_len: int = 512, tag: str = "") -> str:
    """Independent self-generated documents of at most 512 tokens unless stated (2.3 M tokens)."""
    lt = "" if max_len == 512 else f"-M{max_len}"
    return f"out/calib/{model}/selfdoc-T1.0-s{seed}{lt}{tag}.pt"


def selfgen_job(model: str, temperature: float = 1.0, seed: int = 0, n: int = 1088,
                time: str = "0-03:00:00", partitions: str | None = None, seqlen: int = 2048) -> dict:
    out = selfgen_path(model, temperature, seed, n, seqlen)
    suffix = ("" if n == 1088 else f"-n{n}") + ("" if seqlen == 2048 else f"-L{seqlen}")
    return {
        "name": f"fsc-gen-{model}-T{temperature:.1f}-s{seed}" + suffix,
        "script": "scripts/generate_calib.py",
        "args": ["--model", MODELS[model], "--n", str(n), "--seqlen", str(seqlen),
                 "--temperature", str(temperature), "--seed", str(seed),
                 "--batch-size", str(GEN_BATCH[model] * (2 if seqlen <= 512 else 1)), "--out", out],
        "output": out,
        "partitions": PARTITIONS if partitions is None else partitions,
        "time": time,
    }


def selfdoc_job(model: str, seed: int = 0, n_tokens: int = 2_300_000, out: str | None = None,
                partitions: str | None = None, max_len: int = 512) -> dict:
    out = out or selfdoc_path(model, seed, max_len)
    lt = "" if max_len == 512 else f"-M{max_len}"
    # Longer documents need a smaller batch for the KV cache of the batch.
    batch = max(16, DOC_BATCH[model] * 512 // max(max_len, 512))
    return {
        "name": f"fsc-gendoc-{model}-s{seed}{lt}" + ("" if n_tokens == 2_300_000 else "-big"),
        "script": "scripts/generate_calib.py",
        "args": ["--model", MODELS[model], "--documents", "--n-tokens", str(n_tokens),
                 "--max-doc-len", str(max_len), "--temperature", "1.0", "--seed", str(seed),
                 "--batch-size", str(batch), "--out", out],
        "output": out,
        "partitions": PARTITIONS if partitions is None else partitions,
        "time": "0-04:00:00",
    }


def plan_selfgen() -> list[dict]:
    """Exact self-samples: evaluation pools, calibration pools, seeds and temperatures."""
    jobs = [selfgen_job(m, 1.0, 1000, n=64, time="0-01:00:00") for m in MAIN_MODELS]
    jobs += [selfgen_job(m) for m in ("opt-125m",) + MAIN_MODELS]
    jobs += [selfgen_job("llama-3.2-1b", 1.0, s) for s in (1, 2)]
    jobs += [selfgen_job("llama-3.2-1b", t, 0) for t in (0.7, 1.3)]
    return jobs


def plan_selfdoc() -> list[dict]:
    """Self-generated documents for every main model, and two more seeds on Llama-3.2-1B."""
    return [selfdoc_job(m) for m in MAIN_MODELS] + [selfdoc_job("llama-3.2-1b", s) for s in (1, 2)]


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------

def _source_args(model: str, source: str, seed: int = 0, temperature: float = 1.0):
    if source == "selfdoc":
        path = selfdoc_path(model, seed)
        return ["--source", "selfdoc", "--selfgen-path", path], [path]
    if source != "selfgen":
        return ["--source", source], []
    path = selfgen_path(model, temperature, seed)
    return ["--source", "selfgen", "--selfgen-path", path], [path]


def main_job(model, source, ratio, seed=0, mode="proj", extra=(), tag="", temperature=1.0,
             time="0-08:00:00", lr=None, partitions=None) -> dict:
    src, req = _source_args(model, source, seed, temperature)
    sname = source if temperature == 1.0 else f"{source}T{temperature:.1f}"
    name = f"{mode}-{sname}-{_r(ratio)}-s{seed}{tag}"
    kl_pool = selfgen_eval_path(model)
    args = src + ["--ratio", str(ratio), "--seed", str(seed), "--mode", mode,
                  "--kl-selfgen-path", kl_pool] + COMMON
    req = req + [kl_pool]
    if mode == "proj":
        args += ["--lr", str(lr or LR[model])]
    return compress_job(model, name, args + list(extra), time=time, requires=req,
                        partitions=partitions)


def plan_dense() -> list[dict]:
    """The uncompressed models (reference rows of every table)."""
    return [compress_job(m, "dense", ["--mode", "dense"], time="0-02:00:00")
            for m in ("opt-125m",) + MAIN_MODELS]


def plan_main() -> list[dict]:
    """Calibration source x model x ratio, learned projectors (Tables 3-4, Figures 2-4)."""
    return [main_job(m, s, r, extra=["--save-bases"])
            for r in RATIOS for m in MAIN_MODELS for s in SOURCES]


def plan_oblique() -> list[dict]:
    """The training-free oblique whitened truncation of SVD-LLM per source (Table 2)."""
    return [main_job(m, s, r, mode="none", extra=["--init", "oblique"], time="0-03:00:00")
            for r in RATIOS for m in MAIN_MODELS for s in SOURCES]


def plan_subspaces() -> list[dict]:
    """Overlaps of the removed subspaces across sources (Figure A4; needs plan main)."""
    jobs = []
    for m in MAIN_MODELS:
        for r in RATIOS:
            runs = [f"proj-{s}-{_r(r)}-s0" for s in SOURCES]
            out = f"out/subspaces/subspaces-{m}-{_r(r)}.csv"
            jobs.append({"name": f"fsc-subspaces-{m}-{_r(r)}", "script": "scripts/analyze_subspaces.py",
                         "args": ["--model", m, "--runs"] + runs + ["--out", out],
                         "output": out, "partitions": PARTITIONS, "time": "0-01:00:00",
                         "requires": [f"{run_dir(m, x)}/DONE" for x in runs]})
    return jobs


def plan_stats() -> list[dict]:
    """Documents per window, NLL and entropy of every source under each model (Table 1)."""
    jobs = []
    for m in MAIN_MODELS:
        out = f"out/stats/sources-{m}.json"
        sg, sd = selfgen_path(m), selfdoc_path(m)
        jobs.append({"name": f"fsc-stats-{m}", "script": "scripts/source_stats.py",
                     "args": ["--model", MODELS[m], "--selfgen-path", sg, "--selfdoc-path", sd,
                              "--out", out],
                     "output": out, "partitions": PARTITIONS, "time": "0-01:00:00",
                     "requires": [sg, sd]})
    return jobs


# ----------------------------------------------------------------------------
# Ablations (Llama-3.2-1B at -50% unless stated)
# ----------------------------------------------------------------------------

def plan_seeds() -> list[dict]:
    """Seeds 1 and 2 of every source (Table 9; seed 0 is plan main)."""
    return [main_job("llama-3.2-1b", s, 0.5, seed=sd) for sd in (1, 2) for s in SOURCES]


def plan_size() -> list[dict]:
    """Distinct calibration windows at a fixed budget of 512 steps (Figure 5)."""
    jobs = [main_job("llama-3.2-1b", s, 0.5, tag=f"-n{n}",
                     extra=["--n-train", str(n), "--epoch-size", "512"])
            for n in (64, 128, 256, 1024) for s in ("c4", "selfdoc")]
    # 4096 distinct windows, each seen once; the self-generated counterpart needs a
    # larger pool and is plan fresh.
    jobs.append(main_job("llama-3.2-1b", "c4", 0.5, tag="-n4096",
                         extra=["--n-train", "4096", "--epoch-size", "512"]))
    return jobs


def plan_fresh() -> list[dict]:
    """4096 windows of freshly sampled documents, each seen once (Figure 5)."""
    m = "llama-3.2-1b"
    pool = f"out/calib/{m}/selfdoc-T1.0-s0-big.pt"
    gen = selfdoc_job(m, n_tokens=8_500_000, out=pool)
    run = compress_job(m, "proj-selfdoc-r50-s0-fresh4096", [
        "--source", "selfdoc", "--selfgen-path", pool, "--ratio", "0.5", "--seed", "0",
        "--mode", "proj", "--lr", str(LR[m]), "--kl-selfgen-path", selfgen_eval_path(m)]
        + COMMON + ["--n-train", "4096", "--n-val", "32", "--epoch-size", "512"],
        requires=[pool, selfgen_eval_path(m)])
    return [gen, run]


def plan_temperature() -> list[dict]:
    """Exact self-samples drawn at temperatures 0.7 and 1.3 (Table 6)."""
    return [main_job("llama-3.2-1b", "selfgen", 0.5, temperature=t) for t in (0.7, 1.3)]


def plan_factors() -> list[dict]:
    """Free low-rank factors from the same initialization, two learning rates (Table 8)."""
    return [main_job(m, s, 0.5, mode="factors", tag=f"-{_lr(lr)}", extra=["--lr", str(lr)])
            for m in ("llama-3.2-1b", "opt-1.3b") for s in ("wikitext2", "c4", "selfgen")
            for lr in (1e-5, 1e-4)]


def plan_klalloc() -> list[dict]:
    """The measured-KL rank allocation of LSP, computed on each source (Table 7)."""
    return [main_job("llama-3.2-1b", s, 0.5, tag="-klalloc", extra=["--alloc", "kl"])
            for s in SOURCES]


def plan_domain() -> list[dict]:
    """Calibration on single SlimPajama domains (Table 5)."""
    return [main_job("llama-3.2-1b", s, 0.5) for s in ("sp-github", "sp-arxiv", "sp-book")]


def plan_crossover() -> list[dict]:
    """Initialization statistics from one source, training windows from another (Table 6)."""
    jobs = []
    for init_src, train_src in (("c4", "selfdoc"), ("selfdoc", "c4"),
                                ("wikitext2", "slimpajama"), ("slimpajama", "wikitext2")):
        extra = ["--init-source", init_src]
        if init_src == "selfdoc":
            extra += ["--init-selfgen-path", selfdoc_path("llama-3.2-1b")]
        j = main_job("llama-3.2-1b", train_src, 0.5, tag=f"-init{init_src}", extra=extra)
        if init_src == "selfdoc":
            j["requires"].append(selfdoc_path("llama-3.2-1b"))
        jobs.append(j)
    return jobs


def plan_bos() -> list[dict]:
    """Window format: C4 windows without BOS, and with a BOS at every document start
    (Table 6, Appendix C)."""
    jobs = [main_job("llama-3.2-1b", "c4", 0.5, tag="-nobos", extra=["--no-bos-windows"])]
    for r in RATIOS:
        jobs.append(main_job("llama-3.2-1b", "c4", r, tag="-docbos", extra=["--doc-bos"]))
        jobs.append(main_job("llama-3.2-1b", "c4", r, mode="none", tag="-docbos-oblique",
                             extra=["--doc-bos", "--init", "oblique"], time="0-03:00:00"))
    return jobs


# ----------------------------------------------------------------------------
# Learning rates, reproduction check and scale
# ----------------------------------------------------------------------------

def plan_pilot_lr() -> list[dict]:
    """Learning-rate grid on C4 at -50%, eight epochs (Table A2). The "-e8" suffix marks
    the runs added to complete the grid; every run uses the same eight-epoch budget."""
    jobs = []
    for m in ("opt-1.3b", "qwen3-1.7b"):
        for lr in (3e-5, 1e-4, 3e-4):
            jobs.append((m, f"pilot-proj-c4-r50-{_lr(lr)}", lr))
        jobs.append((m, f"pilot-proj-c4-r50-{_lr(1e-3)}-e8", 1e-3))
    for lr in (3e-5, 1e-4, 3e-4, 1e-3):
        jobs.append(("llama-3.2-1b", f"pilot-proj-c4-r50-{_lr(lr)}-e8", lr))
    return [compress_job(m, name, ["--source", "c4", "--ratio", "0.5", "--mode", "proj",
                                   "--lr", str(lr), "--no-zs", "--skip-init-eval"] + COMMON,
                         time="0-06:00:00") for m, name, lr in jobs]


def plan_pilot_opt125m() -> list[dict]:
    """OPT-125M in the reproduction setting of LSP, four learning rates (10 epochs)."""
    return [compress_job("opt-125m", f"pilot-proj-wikitext2-{_r(ratio)}-{_lr(lr)}", [
        "--source", "wikitext2", "--ratio", str(ratio), "--n-train", "1024",
        "--n-val", "64", "--batch-size", "32", "--micro-batch", "8", "--epochs", "10",
        "--lr", str(lr), "--no-zs"], time="0-02:00:00")
        for ratio in (0.3, 0.5, 0.7) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]


def plan_repro() -> list[dict]:
    """Reproduction of LSP with uniform allocation and WikiText-2 calibration (Table A1)."""
    jobs = []
    for model, epochs, lr, time in (("opt-125m", 40, LR["opt-125m"], "0-04:00:00"),
                                    ("opt-1.3b", 20, LR["opt-1.3b"], "0-12:00:00")):
        for ratio in (0.3, 0.5, 0.7):
            jobs.append(compress_job(model, f"repro-wikitext2-{_r(ratio)}", [
                "--source", "wikitext2", "--ratio", str(ratio), "--n-train", "1024",
                "--n-val", "64", "--batch-size", "32", "--micro-batch", "4",
                "--epochs", str(epochs), "--patience", "100", "--lr", str(lr), "--no-zs",
                "--kl-sources", ""], time=time))
    return jobs


def plan_diag_oblique() -> list[dict]:
    """The oblique truncation on OPT-125M in the reproduction setting (Table A1)."""
    return [compress_job("opt-125m", f"oblique-wikitext2-{_r(r)}", [
        "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
        "--mode", "none", "--init", "oblique", "--no-zs"], time="0-01:00:00")
        for r in (0.3, 0.5, 0.7)]


def plan_repro_init() -> list[dict]:
    """Training-free truncations in the reproduction setting: the oblique truncation on
    OPT-1.3B, and the whitened initialization on BOS-less windows (Table A1)."""
    jobs = []
    for r in (0.3, 0.5, 0.7):
        jobs.append(compress_job("opt-1.3b", f"oblique-wikitext2-{_r(r)}", [
            "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
            "--mode", "none", "--init", "oblique", "--no-zs", "--kl-sources", ""],
            time="0-01:00:00"))
        for m in ("opt-125m", "opt-1.3b"):
            jobs.append(compress_job(m, f"init-nobos-wikitext2-{_r(r)}", [
                "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
                "--mode", "none", "--no-bos-windows", "--no-zs", "--kl-sources", ""],
                time="0-01:00:00"))
    return jobs


SCALE = "llama-3.1-8b"
SCALE_ARGS = ["--gram-dtype", "fp32", "--micro-batch", "1", "--eval-batch", "2"]


def plan_scale() -> list[dict]:
    """Llama-3.1-8B at -30%: four learning rates on C4, then five sources (Table 10)."""
    big = PARTITIONS_80G
    jobs = [selfgen_job(SCALE, 1.0, 1000, n=64, time="0-02:00:00", partitions=big),
            selfdoc_job(SCALE, partitions=big),
            compress_job(SCALE, "dense", ["--mode", "dense", "--eval-batch", "2"],
                         time="0-04:00:00", partitions=big)]
    for lr in (3e-6, 1e-5, 3e-5, 1e-4):
        jobs.append(compress_job(SCALE, f"pilot-proj-c4-r30-{_lr(lr)}", [
            "--source", "c4", "--ratio", "0.3", "--mode", "proj", "--lr", str(lr), "--no-zs",
            "--skip-init-eval", "--kl-sources", ""] + COMMON + SCALE_ARGS,
            time="0-12:00:00", partitions=big))
    for src in ("wikitext2", "c4", "slimpajama", "alpaca", "selfdoc"):
        jobs.append(main_job(SCALE, src, 0.3, extra=SCALE_ARGS, time="0-16:00:00", partitions=big))
    return jobs


def plan_smoke() -> list[dict]:
    """A two-minute end-to-end check of the installation (OPT-125M); not in the paper."""
    return [compress_job("opt-125m", "smoke", [
        "--source", "wikitext2", "--ratio", "0.5", "--n-train", "64", "--n-val", "8",
        "--epochs", "2", "--zs-limit", "200"], time="0-00:30:00")]



# ============================================================================
# Extended evaluation (plans prefixed ext-)
# ============================================================================
#
# Every run below logs per-item results (each zero-shot item, each perplexity and
# KL window) for the paired statistics, measures KL on 128 held-out windows per
# source and on six domains chosen independently of the sources, and evaluates the
# last-epoch parameters when early stopping kept another epoch. The runs are written
# to out/extended/runs/, apart from the main grid's runs of the same names.

EXT = "out/extended"
EXT_DOMAINS = "dolly,gsm8k,pubmedqa,mbpp,opus,pg19"
EXT_EVAL_POOL_N = 256
INSTRUCT = "llama-3.2-1b-it"
LR50_8B = 3e-5   # Llama-3.1-8B at -50%: the lower validation KL of plan ext-scale's pilots


def ext_run_dir(model: str, name: str) -> str:
    return f"{EXT}/runs/{model}/{name}"


def ext_compress_job(model: str, name: str, args: list[str], time: str = "0-08:00:00",
                     partitions: str | None = None, requires: list[str] | None = None) -> dict:
    job = compress_job(model, name, args, time=time, partitions=partitions, requires=requires)
    job["name"] = f"fsc-ext-{model}-{name}"
    job["run_dir"] = ext_run_dir(model, name)
    return job


def ext_eval_pool(model: str) -> str:
    """Held-out exact samples for KL: 256 sequences (128 used), seed 1000."""
    return selfgen_path(model, 1.0, 1000, EXT_EVAL_POOL_N)


def ext_job(model, source, ratio, seed=0, mode="proj", extra=(), tag="", time=None, lr=None,
            keep_params=False, src_args=None, requires=(), name=None) -> dict:
    """One run of the extended evaluation: the protocol of main_job plus its evaluation.
    keep_params keeps the trained projectors, which plans ext-behave and ext-optim read."""
    if time is None:
        time = "0-16:00:00" if model == SCALE else "0-08:00:00"
    if src_args is None:
        src, req = _source_args(model, source, seed)
    else:
        src, req = list(src_args), []
    name = name or f"{mode}-{source}-{_r(ratio)}-s{seed}{tag}"
    pool = ext_eval_pool(model)
    args = src + ["--ratio", str(ratio), "--seed", str(seed), "--mode", mode,
                  "--kl-selfgen-path", pool] + COMMON
    if mode in ("proj", "factors"):
        args += ["--lr", str(lr or LR[model])]
    args += ["--log-items", "--kl-eval-windows", "128", "--kl-extra", EXT_DOMAINS,
             "--kl-extra-windows", "64", "--eval-last"]
    if not keep_params:
        args.append("--drop-params")
    if model == SCALE:
        args += SCALE_ARGS
    return ext_compress_job(model, name, args + list(extra), time=time,
                            partitions=PARTITIONS_80G if model == SCALE else None,
                            requires=list(req) + [pool] + list(requires))


def _first_set_outputs() -> set[str]:
    return {j["output"] for name in FIRST_SET for j in PLANS[name]() if j.get("output")}


def plan_ext_pools() -> list[dict]:
    """Self-generated pools of the extended evaluation that the main grid does not
    already produce: held-out pools of 256 samples, seeds 1 and 2 for every model,
    documents of other length limits and exact samples restarted every 512 tokens
    (Qwen3-1.7B and Llama-3.2-1B), and the large document pool of the data-size study."""
    jobs = []
    for m in MAIN_MODELS:
        jobs.append(selfgen_job(m, 1.0, 1000, n=EXT_EVAL_POOL_N, time="0-02:00:00"))
        for sd in (0, 1, 2):
            jobs.append(selfgen_job(m, 1.0, sd))
            jobs.append(selfdoc_job(m, sd))
    for m in ("qwen3-1.7b", "llama-3.2-1b"):
        for L in (128, 1024, 2048):
            jobs.append(selfdoc_job(m, 0, max_len=L))
        jobs.append(selfgen_job(m, 1.0, 0, n=2304, seqlen=512))
    jobs.append(selfdoc_job("llama-3.2-1b", 0, n_tokens=8_500_000,
                            out="out/calib/llama-3.2-1b/selfdoc-T1.0-s0-big.pt"))
    done = _first_set_outputs()
    return [j for j in jobs if j["output"] not in done]


def plan_ext_pools8b() -> list[dict]:
    """Llama-3.1-8B: held-out pool, and exact samples and documents for three seeds."""
    big = PARTITIONS_80G
    jobs = [selfgen_job(SCALE, 1.0, 1000, n=EXT_EVAL_POOL_N, time="0-03:00:00", partitions=big)]
    for sd in (0, 1, 2):
        jobs.append(selfgen_job(SCALE, 1.0, sd, time="0-06:00:00", partitions=big))
        jobs.append(selfdoc_job(SCALE, sd, partitions=big))
    done = _first_set_outputs()
    return [j for j in jobs if j["output"] not in done]


def plan_ext_inst_pools() -> list[dict]:
    """Llama-3.2-1B-Instruct: held-out pool, documents, and its own responses to Alpaca
    instructions outside the held-out set."""
    m = INSTRUCT
    out = f"out/calib/{m}/selfresp-T1.0-s0.pt"
    return [selfgen_job(m, 1.0, 1000, n=EXT_EVAL_POOL_N, time="0-02:00:00"),
            selfdoc_job(m, 0),
            {"name": f"fsc-genresp-{m}-s0", "script": "scripts/generate_calib.py",
             "args": ["--model", MODELS[m], "--responses", "--n-tokens", "2300000",
                      "--max-doc-len", "512", "--temperature", "1.0", "--seed", "0",
                      "--batch-size", "128", "--out", out],
             "output": out, "partitions": PARTITIONS, "time": "0-02:00:00"}]


def plan_ext_dense() -> list[dict]:
    """The uncompressed models under the extended evaluation."""
    jobs = []
    for m in MAIN_MODELS + (SCALE, INSTRUCT):
        args = ["--mode", "dense", "--log-items", "--kl-extra", EXT_DOMAINS,
                "--kl-eval-windows", "128", "--kl-selfgen-path", ext_eval_pool(m)]
        if m == SCALE:
            args += ["--eval-batch", "2"]
        if m == INSTRUCT:
            args += ["--gen-tasks", "ifeval", "--chat-template"]
        jobs.append(ext_compress_job(m, "dense", args, time="0-04:00:00",
                                     partitions=PARTITIONS_80G if m == SCALE else None,
                                     requires=[ext_eval_pool(m)]))
    return jobs


def plan_ext_main() -> list[dict]:
    """The full grid with three seeds (3 models x 2 ratios x 6 sources x 3 seeds;
    Section 4.9, Tables A3 and A4). Seed 0 keeps its projectors for plan ext-behave."""
    jobs = []
    for sd in (0, 1, 2):
        for r in RATIOS:
            for m in MAIN_MODELS:
                for s in SOURCES:
                    # Llama-3.2-1B at -50%, seed 0: initial bases kept for the subspace
                    # comparison with the longer schedule of plan ext-optim.
                    extra = ["--save-bases"] if (sd, r, m) == (0, 0.5, "llama-3.2-1b") else []
                    jobs.append(ext_job(m, s, r, seed=sd, keep_params=(sd == 0), extra=extra))
    return jobs


def plan_ext_scale() -> list[dict]:
    """Llama-3.1-8B: six sources and three seeds at -30%, and two learning rates on C4
    at -50% (Section 4.10, Table 18)."""
    jobs = [ext_job(SCALE, s, 0.3, seed=sd, keep_params=(sd == 0))
            for sd in (0, 1, 2) for s in SOURCES]
    for lr in (3e-5, 1e-4):
        jobs.append(ext_compress_job(SCALE, f"pilot-proj-c4-r50-{_lr(lr)}", [
            "--source", "c4", "--ratio", "0.5", "--mode", "proj", "--lr", str(lr), "--no-zs",
            "--skip-init-eval", "--kl-sources", "", "--drop-params"] + COMMON + SCALE_ARGS,
            time="0-12:00:00", partitions=PARTITIONS_80G))
    return jobs


def plan_ext_scale50() -> list[dict]:
    """Llama-3.1-8B at -50% with six sources, one seed (Table 18)."""
    return [ext_job(SCALE, s, 0.5, lr=LR50_8B, keep_params=True) for s in SOURCES]


def plan_ext_format() -> list[dict]:
    """Matched-segment format: 16 segments of 127 tokens per window, a single BOS
    (Section 4.4, Table 10)."""
    return [ext_job(m, s, 0.5, tag="-seg127", extra=["--segment-len", "127"])
            for m in MAIN_MODELS for s in ("wikitext2-art", "c4", "slimpajama", "alpaca", "selfdoc")]


def plan_ext_alpaca() -> list[dict]:
    """Alpaca without its template, and its responses alone (Section 4.2, Table 4)."""
    jobs = [ext_job(m, "alpaca-plain", r) for m in MAIN_MODELS for r in RATIOS]
    jobs += [ext_job(m, "alpaca-resp", 0.5) for m in MAIN_MODELS]
    return jobs


def plan_ext_optim() -> list[dict]:
    """Llama-3.2-1B at -50%: per-source learning rates, a three times longer schedule,
    and the crossover of initialization and training data under it (Section 4.8, Table 16)."""
    m, jobs = "llama-3.2-1b", []
    for s in SOURCES:
        for lr in (3e-5, 3e-4):
            jobs.append(ext_job(m, s, 0.5, lr=lr, tag=f"-{_lr(lr)}"))
        jobs.append(ext_job(m, s, 0.5, tag="-e24", time="0-16:00:00", keep_params=True,
                            extra=["--epochs", "24", "--save-bases"]))
    for init_src, train_src in (("slimpajama", "wikitext2"), ("wikitext2", "slimpajama")):
        jobs.append(ext_job(m, train_src, 0.5, tag=f"-init{init_src}-e24", time="0-16:00:00",
                            keep_params=True,
                            extra=["--epochs", "24", "--init-source", init_src, "--save-bases"]))
    return jobs



def plan_ext_subspaces() -> list[dict]:
    """Overlaps of the removed subspaces after 8 and 24 epochs, and in the crossover
    (Section 4.8); needs plans ext-main and ext-optim, which keep the bases."""
    m = "llama-3.2-1b"
    runs = ([f"proj-{s}-r50-s0" for s in SOURCES] + [f"proj-{s}-r50-s0-e24" for s in SOURCES]
            + ["proj-slimpajama-r50-s0-initwikitext2-e24", "proj-wikitext2-r50-s0-initslimpajama-e24"])
    out = f"{EXT}/analysis/subspaces-e24-{m}-r50.csv"
    return [{"name": "fsc-ext-subspaces-e24", "script": "scripts/analyze_subspaces.py",
             "args": ["--model", m, "--root", f"{EXT}/runs", "--runs"] + runs + ["--out", out],
             "output": out, "partitions": PARTITIONS, "time": "0-01:00:00",
             "requires": [f"{ext_run_dir(m, r)}/DONE" for r in runs]}]

def plan_ext_size() -> list[dict]:
    """Distinct windows at a fixed budget of 512 steps, with learning curves (Figure 6)."""
    m, jobs = "llama-3.2-1b", []
    for n in (64, 128, 256, 1024, 4096):
        for s in ("c4", "selfdoc"):
            extra = ["--n-train", str(n), "--epoch-size", "512"]
            if s == "selfdoc" and n == 4096:
                pool = selfdoc_path(m, 0, tag="-big")
                jobs.append(ext_job(m, s, 0.5, tag=f"-n{n}", requires=[pool], extra=extra,
                                    src_args=["--source", "selfdoc", "--selfgen-path", pool]))
            else:
                jobs.append(ext_job(m, s, 0.5, tag=f"-n{n}", extra=extra))
    return jobs


def plan_ext_mix() -> list[dict]:
    """Half SlimPajama, half self-generated documents, at the same budget (Table 7)."""
    jobs = []
    for m in MAIN_MODELS:
        for r in RATIOS:
            pool = selfdoc_path(m, 0)
            jobs.append(ext_job(m, "slimpajama", r, requires=[pool],
                                extra=["--mix-source", "selfdoc", "--mix-frac", "0.5",
                                       "--mix-selfgen-path", pool],
                                name=f"proj-mix-sp-selfdoc-{_r(r)}-s0"))
    return jobs


def plan_ext_qwen() -> list[dict]:
    """The failure of exact self-samples on Qwen3-1.7B: documents limited to 128, 1024 or
    2048 tokens, and exact samples restarted every 512 tokens, at -30%; Llama-3.2-1B for
    contrast (Section 4.6, Table 12)."""
    jobs = []
    for m in ("qwen3-1.7b", "llama-3.2-1b"):
        for L in (128, 1024, 2048):
            pool = selfdoc_path(m, 0, max_len=L)
            jobs.append(ext_job(m, "selfdoc", 0.3, requires=[pool],
                                src_args=["--source", "selfdoc", "--selfgen-path", pool],
                                name=f"proj-selfdoc-r30-s0-M{L}"))
        pool = selfgen_path(m, 1.0, 0, 2304, 512)
        jobs.append(ext_job(m, "selfpack", 0.3, requires=[pool],
                            src_args=["--source", "selfpack", "--selfgen-path", pool],
                            name="proj-selfpack512-r30-s0"))
    return jobs


def plan_ext_lora() -> list[dict]:
    """Another learned compressor: the truncation of SVD-LLM recovered with LoRA (rank 8,
    cross-entropy on the calibration windows; Section 4.7, Table 15)."""
    return [ext_job(m, s, 0.3, mode="lora", extra=["--lr", "1e-4"])
            for m in ("llama-3.2-1b", "opt-1.3b") for s in SOURCES]


def plan_ext_initcmp() -> list[dict]:
    """The untrained initialization under the choices documented by LSP, in its
    reproduction setting (Appendix B, Table A7)."""
    variants = {
        "base": [],
        "sqout": ["--square-side", "out"],
        "ridge": ["--out-ridge", "0.01"],
        "sqout-ridge": ["--square-side", "out", "--out-ridge", "0.01"],
        "fp32": ["--gram-dtype", "fp32"],
        "nobos": ["--no-bos-windows"],
    }
    jobs = []
    for m in ("opt-125m", "opt-1.3b"):
        for r in (0.3, 0.5, 0.7):
            for v, extra in variants.items():
                jobs.append(ext_compress_job(m, f"initcmp-{v}-{_r(r)}", [
                    "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024",
                    "--n-val", "64", "--mode", "none", "--no-zs", "--kl-sources", "",
                    "--eval-sp", ""] + extra, time="0-01:00:00"))
            jobs.append(ext_compress_job(m, f"initcmp-oblique-sqout-{_r(r)}", [
                "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
                "--mode", "none", "--init", "oblique", "--square-side", "out", "--no-zs",
                "--kl-sources", "", "--eval-sp", ""], time="0-01:00:00"))
    return jobs


def plan_ext_sqout() -> list[dict]:
    """Square attention-output projections on their output side, trained: the
    reproduction setting of LSP, and three sources in the setting of the main grid
    (Appendix B, Table A8)."""
    jobs = []
    for model, epochs, lr, time, ratios in (("opt-125m", 40, 1e-3, "0-04:00:00", (0.3, 0.5, 0.7)),
                                            ("opt-1.3b", 20, LR["opt-1.3b"], "0-12:00:00", (0.5, 0.7))):
        for ratio in ratios:
            jobs.append(ext_compress_job(model, f"repro-sqout-wikitext2-{_r(ratio)}", [
                "--source", "wikitext2", "--ratio", str(ratio), "--n-train", "1024",
                "--n-val", "64", "--batch-size", "32", "--micro-batch", "4",
                "--epochs", str(epochs), "--patience", "100", "--lr", str(lr), "--no-zs",
                "--kl-sources", "", "--eval-sp", "", "--square-side", "out", "--drop-params"],
                time=time))
    for m in ("llama-3.2-1b", "opt-1.3b"):
        for s in ("wikitext2", "c4", "selfdoc"):
            jobs.append(ext_job(m, s, 0.5, tag="-sqout", extra=["--square-side", "out"]))
    return jobs


def plan_ext_inst() -> list[dict]:
    """Instruction following: Llama-3.2-1B-Instruct at -30%, IFEval in its chat format
    (Section 4.11, Table 20)."""
    m, jobs = INSTRUCT, []
    chat = ["--gen-tasks", "ifeval", "--chat-template", "--doc-bos"]
    for s in ("c4", "slimpajama", "alpaca-chat", "selfdoc"):
        jobs.append(ext_job(m, s, 0.3, keep_params=True, extra=chat))
    pool = f"out/calib/{m}/selfresp-T1.0-s0.pt"
    jobs.append(ext_job(m, "selfresp", 0.3, keep_params=True, requires=[pool], extra=chat,
                        src_args=["--source", "selfresp", "--selfgen-path", pool]))
    return jobs


def plan_ext_behave() -> list[dict]:
    """Open-ended generation, long contexts, passkey retrieval and generative tasks of the
    seed-0 models at -30% and -50% (Llama-3.1-8B: -30%) and the dense models (Table 19)."""
    jobs = []
    for m in MAIN_MODELS + (SCALE,):
        ratios = (0.3,) if m == SCALE else RATIOS
        for run in ["dense"] + [f"proj-{s}-{_r(r)}-s0" for r in ratios for s in SOURCES]:
            d = ext_run_dir(m, run)
            args = ["--run-dir", d]
            if m in ("qwen3-1.7b", SCALE):
                args += ["--gen-tasks", "gsm8k,nq_open"]
            if m != "opt-1.3b":
                args += ["--long-context"]
            jobs.append({"name": f"fsc-ext-behave-{m}-{run}", "script": "scripts/behave.py",
                         "args": args, "output": f"{d}/behave.json",
                         "partitions": PARTITIONS_80G if m == SCALE else PARTITIONS,
                         "time": "0-08:00:00" if m == SCALE else "0-04:00:00",
                         "requires": [f"{d}/DONE"]})
    return jobs


def plan_ext_bench() -> list[dict]:
    """Memory and latency of the dense and the merged compressed models (Table 21)."""
    jobs = []
    for m in MAIN_MODELS + (SCALE,):
        out = f"{EXT}/analysis/bench-{m}.json"
        jobs.append({"name": f"fsc-ext-bench-{m}", "script": "scripts/bench.py",
                     "args": ["--model", MODELS[m], "--ratios", "0.3,0.5", "--out", out],
                     "output": out, "partitions": PARTITIONS_80G if m == SCALE else PARTITIONS,
                     "time": "0-02:00:00"})
    return jobs


def plan_ext_selfstats() -> list[dict]:
    """Repetition, script, language and kind of content of the self-generated pools
    (Section 4.6, Table 13); CPU only."""
    out = f"{EXT}/analysis/selfgen-composition.json"
    pools = []
    for m in MAIN_MODELS:
        pools += [f"{m}:selfgen:{selfgen_path(m)}", f"{m}:selfdoc:{selfdoc_path(m)}"]
    return [{"name": "fsc-ext-selfstats", "script": "scripts/analyze_selfgen.py",
             "args": ["--pools"] + pools + ["--out", out], "output": out,
             "partitions": PARTITIONS, "time": "0-02:00:00",
             "requires": [x.split(":", 2)[2] for x in pools]}]


def plan_ext_selfstats_seeds() -> list[dict]:
    """The composition of the exact-sample pool of each seed (Section 4.6); CPU only."""
    out = f"{EXT}/analysis/selfgen-composition-seeds.json"
    pools = [f"{m}:selfgen:s{k}:{selfgen_path(m, 1.0, k)}"
             for m in ("qwen3-1.7b", "llama-3.2-1b") for k in (0, 1, 2)]
    return [{"name": "fsc-ext-selfstats-seeds", "script": "scripts/analyze_selfgen.py",
             "args": ["--pools"] + pools + ["--out", out], "output": out,
             "partitions": PARTITIONS, "time": "0-03:00:00",
             "requires": [x.split(":", 3)[3] for x in pools]}]


PLANS = {
    "smoke": plan_smoke,
    "dense": plan_dense,
    "selfgen": plan_selfgen,
    "selfdoc": plan_selfdoc,
    "stats": plan_stats,
    "pilot-lr": plan_pilot_lr,
    "pilot-opt125m": plan_pilot_opt125m,
    "main": plan_main,
    "oblique": plan_oblique,
    "subspaces": plan_subspaces,
    "seeds": plan_seeds,
    "size": plan_size,
    "fresh": plan_fresh,
    "temperature": plan_temperature,
    "factors": plan_factors,
    "klalloc": plan_klalloc,
    "domain": plan_domain,
    "crossover": plan_crossover,
    "bos": plan_bos,
    "repro": plan_repro,
    "diag-oblique": plan_diag_oblique,
    "repro-init": plan_repro_init,
    "scale": plan_scale,
    "ext-pools": plan_ext_pools,
    "ext-pools8b": plan_ext_pools8b,
    "ext-inst-pools": plan_ext_inst_pools,
    "ext-dense": plan_ext_dense,
    "ext-main": plan_ext_main,
    "ext-scale": plan_ext_scale,
    "ext-scale50": plan_ext_scale50,
    "ext-format": plan_ext_format,
    "ext-alpaca": plan_ext_alpaca,
    "ext-optim": plan_ext_optim,
    "ext-subspaces": plan_ext_subspaces,
    "ext-size": plan_ext_size,
    "ext-mix": plan_ext_mix,
    "ext-qwen": plan_ext_qwen,
    "ext-lora": plan_ext_lora,
    "ext-initcmp": plan_ext_initcmp,
    "ext-sqout": plan_ext_sqout,
    "ext-inst": plan_ext_inst,
    "ext-behave": plan_ext_behave,
    "ext-bench": plan_ext_bench,
    "ext-selfstats": plan_ext_selfstats,
    "ext-selfstats-seeds": plan_ext_selfstats_seeds,
}

# Every plan behind the paper, in an order that satisfies the dependencies
# (self-generated pools before the runs that read them, plan main before subspaces,
# trained runs before their behavioral evaluation).
FIRST_SET = ["dense", "selfgen", "selfdoc", "stats", "pilot-lr", "pilot-opt125m", "main",
             "oblique", "subspaces", "seeds", "size", "fresh", "temperature", "factors",
             "klalloc", "domain", "crossover", "bos", "repro", "diag-oblique", "repro-init",
             "scale"]
EXTENDED = ["ext-pools", "ext-pools8b", "ext-inst-pools", "ext-dense", "ext-main", "ext-scale",
            "ext-scale50", "ext-format", "ext-alpaca", "ext-optim", "ext-subspaces", "ext-size", "ext-mix",
            "ext-qwen", "ext-lora", "ext-initcmp", "ext-sqout", "ext-inst", "ext-behave",
            "ext-bench", "ext-selfstats", "ext-selfstats-seeds"]
PAPER = FIRST_SET + EXTENDED


def get(name: str) -> list[dict]:
    if name == "all":
        return [job for plan in PAPER for job in PLANS[plan]()]
    if name not in PLANS:
        raise KeyError(f"unknown plan {name!r}; have {sorted(PLANS)} or 'all'")
    return PLANS[name]()
