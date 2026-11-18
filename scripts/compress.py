#!/usr/bin/env python3
"""One compression run, end to end.

    uv run --no-sync python scripts/compress.py --model meta-llama/Llama-3.2-1B \
        --source c4 --ratio 0.3 --run-dir out/runs/llama-3.2-1b/proj-c4-r30-s0

Phases: calibration windows -> Gram matrices -> whitened bases -> allocation ->
evaluation of the training-free initialization (merged) -> training (projectors,
free factors, or LoRA on factors) -> evaluation of the trained model (merged) ->
optionally, evaluation of the last epoch when another one was selected.

Every phase records its result in ``<run-dir>/metrics.json`` (or its own file)
and is skipped when that result exists, so a resubmitted job resumes where the
previous one stopped. Training itself checkpoints every epoch.

Modes:
    dense    evaluate the uncompressed model only
    none     training-free whitened truncation only (the initialization, "NoLSP")
    proj     learned subspace projections (weights frozen)
    factors  free low-rank factors trained from the same initialization
    lora     SVD-LLM's oblique truncation recovered by LoRA on both factors, trained
             with next-token cross-entropy on the calibration text (no teacher)

With --log-items, per-item results (every perplexity window, every KL window,
every benchmark item) go to ``items_<phase>.npz`` for paired comparisons.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fsc import alloc, arch, data, evaluate, hub, project, stats, train  # noqa: E402
from fsc import init as finit  # noqa: E402
from fsc.utils import Timer, env_info, log, read_json, write_json  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["dense", "none", "proj", "factors", "lora"], default="proj")
    ap.add_argument("--source", default="c4")
    ap.add_argument("--selfgen-path", default=None)
    ap.add_argument("--init-source", default=None,
                    help="source of the Gram statistics (initialization); default: --source")
    ap.add_argument("--init-selfgen-path", default=None)
    ap.add_argument("--doc-bos", action="store_true",
                    help="start every packed document with BOS, not only every window")
    ap.add_argument("--no-bos-windows", action="store_true",
                    help="calibrate on plain chunks without a leading BOS (window-format ablation)")
    ap.add_argument("--segment-len", type=int, default=0,
                    help="matched-segment format: windows of equal segments of this many tokens, "
                         "each the start of a distinct document (0 = pack whole documents)")
    ap.add_argument("--mix-source", default=None,
                    help="second calibration source, mixed with --source at --mix-frac")
    ap.add_argument("--mix-frac", type=float, default=0.5)
    ap.add_argument("--mix-selfgen-path", default=None)
    ap.add_argument("--ratio", type=float, default=0.3, help="fraction of compressible weights removed")
    ap.add_argument("--alloc", choices=["uniform", "kl"], default="uniform")
    ap.add_argument("--alloc-from", default=None, help="reuse alloc.json of another run")
    ap.add_argument("--init", choices=["recast", "oblique"], default="recast",
                    help="training-free evaluation: orthogonal recast (the LSP initialization) or "
                         "the oblique whitened truncation itself (SVD-LLM); oblique needs --mode none")
    ap.add_argument("--square-side", choices=["in", "out"], default="in",
                    help="side on which square single layers (attention output) are projected")
    ap.add_argument("--out-ridge", type=float, default=0.0,
                    help="ridge on output-side Grams, as a multiple of their mean diagonal")
    ap.add_argument("--untied", action="store_true", help="no Q/K/V or gate/up tying")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--gram-windows", type=int, default=None, help="default: all training windows")
    ap.add_argument("--gram-batch", type=int, default=4)
    ap.add_argument("--gram-dtype", choices=["fp64", "fp32"], default="fp64",
                    help="Gram accumulator precision; fp32 for 4096-wide models")
    ap.add_argument("--kl-windows", type=int, default=32)
    ap.add_argument("--kl-grid", type=int, default=8, help="removal fractions i/grid, i=1..grid-1")
    ap.add_argument("--kl-batch", type=int, default=4)
    # training
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--drop-p", type=float, default=0.05)
    ap.add_argument("--lam-ort", type=float, default=0.05)
    ap.add_argument("--alpha-epochs", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--epoch-size", type=int, default=None,
                    help="windows per (virtual) epoch; default: --n-train. With a fixed "
                         "epoch size the step budget no longer depends on --n-train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["kl", "ce"], default=None,
                    help="training objective; default kl, and ce for --mode lora")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--state-dir", default=None,
                    help="where train_state.pt (optimizer state, for resuming) is kept")
    # evaluation
    ap.add_argument("--eval-ppl", default="wikitext2,c4")
    ap.add_argument("--eval-sp", default="cc,c4,github,book,arxiv,wiki,stack")
    ap.add_argument("--sp-windows", type=int, default=32)
    ap.add_argument("--eval-seqlen", type=int, default=2048)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--zs-tasks", default=",".join(evaluate.ZS_TASKS))
    ap.add_argument("--zs-batch", type=int, default=16)
    ap.add_argument("--zs-limit", type=int, default=None)
    ap.add_argument("--no-zs", action="store_true")
    ap.add_argument("--gen-tasks", default="",
                    help="lm-eval generation tasks evaluated after zero-shot (e.g. ifeval)")
    ap.add_argument("--chat-template", action="store_true",
                    help="apply the tokenizer's chat template in --gen-tasks (instruct models)")
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--skip-init-eval", action="store_true")
    ap.add_argument("--save-bases", action="store_true", help="store the initial removed bases (fp16)")
    ap.add_argument("--kl-sources", default="wikitext2,c4,slimpajama,alpaca,selfgen",
                    help="held-out sets on which KL(dense || compressed) is measured")
    ap.add_argument("--kl-eval-windows", type=int, default=32)
    ap.add_argument("--kl-selfgen-path", default=None,
                    help="self-generated pool reserved for evaluation (never used for calibration)")
    ap.add_argument("--kl-extra", default="",
                    help="held-out domains outside every calibration source (data.EVAL_DOMAINS); "
                         "KL and perplexity are measured on them")
    ap.add_argument("--kl-extra-windows", type=int, default=64)
    ap.add_argument("--log-items", action="store_true",
                    help="store per-item results in items_<phase>.npz")
    ap.add_argument("--eval-last", action="store_true",
                    help="also evaluate the last epoch's parameters when another epoch was selected")
    ap.add_argument("--drop-params", action="store_true",
                    help="delete best_params.pt and last_params.pt once the run is done")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--reeval-zs", action="store_true",
                    help="re-run only the zero-shot evaluation of a finished run, with "
                         "--zs-batch; every other metric is kept (see reevaluate_zero_shot)")
    args = ap.parse_args(argv)
    if args.loss is None:
        args.loss = "ce" if args.mode == "lora" else "kl"
    return args


def _csv(s: str) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def load_model(name: str, dtype, device):
    tok = hub.load_tokenizer(name)
    model = hub.load_causal_lm(name, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def evaluate_all(model, tok, args, device, streams, sp_windows, items: dict | None = None) -> dict:
    """``streams[name] = (stream_without_bos, stream_gptq)``; see evaluate.perplexity.

    With ``items``, per-window NLLs and per-item benchmark results are added to it.
    """
    torch.cuda.empty_cache()
    out = {"ppl": {}, "ppl_stream": {}, "sp_ppl": {}}
    bos = tok.bos_token_id

    def pw(key):
        if items is None:
            return None
        items[key] = []
        return items[key]

    for name, (plain, gptq) in streams.items():
        out["ppl"][name] = evaluate.perplexity(model, plain, args.eval_seqlen, args.eval_batch,
                                               device, bos=bos, per_window=pw(f"ppl/{name}"))
        if bos is not None:
            out["ppl_stream"][name] = evaluate.perplexity(model, gptq, args.eval_seqlen,
                                                          args.eval_batch, device,
                                                          per_window=pw(f"pplstream/{name}"))
        else:
            out["ppl_stream"][name] = out["ppl"][name]
        log(f"    ppl {name}: {out['ppl'][name]:.3f} (stream {out['ppl_stream'][name]:.3f})")
    for sub, w in sp_windows.items():
        out["sp_ppl"][sub] = evaluate.window_perplexity(model, w, args.eval_batch, device,
                                                        per_window=pw(f"sp/{sub}"))
    if sp_windows:
        log("    sp ppl " + " ".join(f"{k}={v:.2f}" for k, v in out["sp_ppl"].items()))
    if not args.no_zs:
        out.update(zero_shot_metrics(model, tok, args, items))
    if _csv(args.gen_tasks):
        out["gen"] = gen_task_metrics(model, tok, args, items)
    return out


ZS_KEYS = ("zs", "zs_mean_acc", "zs_mean_mixed")


def zero_shot_metrics(model, tok, args, items: dict | None = None) -> dict:
    """Zero-shot accuracy on --zs-tasks: per task, and the mean (plain accuracy, and
    acc_norm where the task has it)."""
    tasks = _csv(args.zs_tasks)
    zi = {} if items is not None else None
    zs = evaluate.zero_shot(model, tok, tasks, args.zs_batch, args.zs_limit, items=zi)
    if items is not None:
        for task, d in zi.items():
            for k, v in d.items():
                items[f"zs/{task}/{k}"] = v
    mixed = [zs[t].get("acc_norm", zs[t]["acc"]) for t in tasks if t in zs]
    out = {"zs": zs, "zs_mean_acc": evaluate.zs_mean(zs, "acc", tasks),
           "zs_mean_mixed": float(sum(mixed) / len(mixed))}
    log(f"    zero-shot mean acc {out['zs_mean_acc']:.4f} (mixed {out['zs_mean_mixed']:.4f})")
    return out


def gen_task_metrics(model, tok, args, items: dict | None = None) -> dict:
    """lm-eval generation tasks (default few-shot settings of each task)."""
    tasks = _csv(args.gen_tasks)
    gi = {} if items is not None else None
    res = evaluate.zero_shot(model, tok, tasks, args.gen_batch, args.zs_limit, items=gi,
                             num_fewshot=None, chat_template=args.chat_template)
    if items is not None:
        for task, d in gi.items():
            for k, v in d.items():
                items[f"gen/{task}/{k}"] = v
    log("    generation tasks " + " ".join(
        f"{t}:{','.join(f'{k}={v:.4f}' for k, v in r.items() if 'stderr' not in k)}"
        for t, r in res.items()))
    return res


def save_items(run_dir: str, phase: str, items: dict | None) -> None:
    if items is None:
        return
    # Keys like "zs/boolq/acc" are stored as "zs|boolq|acc": npz members are files.
    arrays = {k.replace("/", "|"): np.asarray(v, dtype=np.int32 if k.endswith("_doc_id") else np.float32)
              for k, v in items.items()}
    tmp = os.path.join(run_dir, f"items_{phase}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(run_dir, f"items_{phase}.npz"))


@torch.no_grad()
def merge_check(model, bank, units, bases, windows, device, dtype) -> float:
    """KL between the projected model and its merged factors on a few windows."""
    x = windows[:2].to(device)
    bank.set_active(True)
    a = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
    bank.clear()
    bank.detach(model)
    try:
        with project.merged(model, units, bases, dtype):
            b = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
    finally:
        bank.attach(model)
    return float((a.exp() * (a - b)).sum(-1).mean())


def merge_parts(model, units, bases, dtype=None) -> list:
    """Exact low-rank factors of every unit with a non-empty removed basis, in ``dtype``."""
    parts = []
    for u in units:
        U = bases.get(u.name)
        if U is not None and U.shape[1] > 0:
            parts += project.merge_factors(model, u, U, dtype)[1]
    return parts


def calib_windows(args, tok, source, selfgen_path, n_train, n_val):
    """Calibration windows of one source in the run's format (segments, BOS, mixture)."""
    kw = dict(seed=args.seed, use_bos=not args.no_bos_windows, doc_bos=args.doc_bos,
              segment_len=args.segment_len)
    if args.mix_source and source == args.source:
        return data.mixed_windows(
            tok, [(source, selfgen_path, 1.0 - args.mix_frac),
                  (args.mix_source, args.mix_selfgen_path, args.mix_frac)],
            n_train, n_val, args.seqlen, **kw)
    return data.calibration_windows(tok, source, n_train, n_val, args.seqlen,
                                    selfgen_path=selfgen_path, **kw)


def calibration_and_bases(args, model, units, tok, device, timer):
    """Calibration windows, and the whitened bases of every unit from their Gram matrices."""
    with timer("calibration"):
        train_w, val_w = calib_windows(args, tok, args.source, args.selfgen_path,
                                       args.n_train, args.n_val)
    log(f"calibration {args.source}: train {tuple(train_w.shape)}, val {tuple(val_w.shape)}")
    if args.segment_len:
        log(f"segment format: {data.SEGMENT_STATS}")
    with timer("grams"):
        gw = train_w
        if args.init_source and args.init_source != args.source:
            gw, _ = data.calibration_windows(tok, args.init_source, args.n_train, 1, args.seqlen,
                                             seed=args.seed, selfgen_path=args.init_selfgen_path,
                                             use_bos=not args.no_bos_windows,
                                             segment_len=args.segment_len)
            log(f"initialization statistics from {args.init_source}")
        gw = gw if args.gram_windows is None else gw[:args.gram_windows]
        acc = torch.float64 if args.gram_dtype == "fp64" else torch.float32
        grams, n_tok = stats.collect_grams(model, units, gw, args.gram_batch, acc_dtype=acc,
                                           device=device)
        bases = finit.build_bases(model, units, grams, out_ridge=args.out_ridge)
        del grams
        torch.cuda.empty_cache()
    log(f"Grams on {n_tok} tokens; bases built")
    return train_w, val_w, bases


def oblique_parts(model, units, bases, ks, device, dtype) -> list:
    """Factors of the oblique whitened truncation (SVD-LLM) with the given ranks."""
    parts = []
    for u in units:
        k = ks[u.name]
        if k == 0:
            continue
        if u.side == "in":
            ws = [model.get_submodule(p).weight.detach() for p in u.members]
            A, Bs = bases[u.name].oblique_factors(ws, k, device=device)
            A = A.to(dtype)                  # cast per unit: see project.merge_factors
            parts += [(p, A, B.to(dtype), model.get_submodule(p).bias) for p, B in zip(u.members, Bs)]
        else:
            parts += project.merge_factors(model, u, bases[u.name].removed(k, device=device), dtype)[1]
    return parts


def recast_bases(units, bases, ks, device) -> dict:
    """Removed bases of the orthogonal recast, the initialization of the projectors."""
    return {u.name: bases[u.name].removed(ks[u.name], device=device).float()
            for u in units if ks[u.name] > 0}


@contextmanager
def _active(bank, flag: bool):
    bank.set_active(flag)
    try:
        yield
    finally:
        bank.set_active(True)


@torch.no_grad()
def kl_to_dense(model, kl_sets: dict, teacher, student, batch: int, device,
                items: dict | None = None) -> dict:
    """Token-averaged KL(dense || compressed) on each held-out set.

    ``teacher`` and ``student`` are zero-argument callables returning context
    managers under which the model computes the dense or the compressed function.
    The same passes give the student's and the teacher's next-token NLL, whose
    perplexities are returned under "xppl" and "xppl_dense". With ``items``, the
    per-window KL and NLL sums are stored (keys kl/, nlls/, nllt/; counts in ntok/).
    """
    # Large-vocabulary logits (Qwen3: 152k) need several GiB in one piece; after
    # training, the cached blocks can be too fragmented to provide it.
    torch.cuda.empty_cache()
    out, xppl, xppl_d = {}, {}, {}
    for name, w in kl_sets.items():
        total, n, s_nll, t_nll, cnt = 0.0, 0, 0.0, 0.0, 0
        per_kl, per_s, per_t = [], [], []
        for i in range(0, w.shape[0], batch):
            x = w[i:i + batch].to(device)
            with teacher():
                t = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
            with student():
                st = model(input_ids=x, use_cache=False).logits
            for b in range(x.shape[0]):
                kb = alloc.kl_sum(t[b:b + 1], st[b:b + 1])
                ls = st[b, :-1].float().log_softmax(-1)
                sn = -float(ls.gather(-1, x[b, 1:, None]).sum())
                tn = -float(t[b, :-1].gather(-1, x[b, 1:, None]).sum())
                per_kl.append(kb)
                per_s.append(sn)
                per_t.append(tn)
                total += kb
                s_nll += sn
                t_nll += tn
                cnt += x.shape[1] - 1
            n += x.numel()
            del t, st
        out[name] = total / n
        xppl[name] = math.exp(s_nll / cnt)
        xppl_d[name] = math.exp(t_nll / cnt)
        if items is not None:
            items[f"kl/{name}"] = per_kl
            items[f"nlls/{name}"] = per_s
            items[f"nllt/{name}"] = per_t
            items[f"ntok/{name}"] = [w.shape[1]] * w.shape[0]
    log("    KL to dense " + " ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return {"kl": out, "xppl": xppl, "xppl_dense": xppl_d}


def build_kl_sets(tok, args) -> dict:
    sets = {}
    for src in _csv(args.kl_sources):
        if src == "selfgen":
            if not args.kl_selfgen_path or not os.path.exists(args.kl_selfgen_path):
                log("    no self-generated evaluation pool; skipping its KL")
                continue
            sets[src] = data.kl_eval_windows(tok, src, args.kl_eval_windows, args.seqlen,
                                             selfgen_path=args.kl_selfgen_path)
        else:
            sets[src] = data.kl_eval_windows(tok, src, args.kl_eval_windows, args.seqlen)
    for src in _csv(args.kl_extra):
        sets[src] = data.kl_eval_windows(tok, src, args.kl_extra_windows, args.seqlen)
    log("held-out sets: " + " ".join(f"{k}={tuple(v.shape)}" for k, v in sets.items()))
    return sets


def attach_kl(metrics_phase: dict, res: dict) -> None:
    metrics_phase["kl"] = res["kl"]
    metrics_phase["xppl"] = res["xppl"]
    metrics_phase["xppl_dense"] = res["xppl_dense"]


def mark_done(run_dir: str) -> None:
    """Completion marker other jobs can depend on (metrics.json exists before a run ends)."""
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write("done\n")


def finish(args, metrics, save) -> int:
    metrics["done"] = True
    save()
    if args.drop_params:
        for f in ("best_params.pt", "last_params.pt"):
            p = os.path.join(args.run_dir, f)
            if os.path.exists(p):
                os.remove(p)
    mark_done(args.run_dir)
    log("done")
    return 0


def main(argv=None) -> int:
    args = parse(argv)
    if args.reeval_zs:
        return reevaluate_zero_shot(args)
    os.makedirs(args.run_dir, exist_ok=True)
    mpath = os.path.join(args.run_dir, "metrics.json")
    metrics = read_json(mpath, {})
    if metrics.get("done"):
        log(f"{args.run_dir}: already done")
        return 0
    write_json(os.path.join(args.run_dir, "config.json"), vars(args))
    envs = read_json(os.path.join(args.run_dir, "env.json"), [])
    envs.append(env_info())
    write_json(os.path.join(args.run_dir, "env.json"), envs)

    device = "cuda"
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    timer = Timer()
    timings = metrics.setdefault("timing", {})

    def save():
        timings.update(timer.t)
        write_json(mpath, metrics)

    def new_items():
        return {} if args.log_items else None

    log(f"loading {args.model}")
    model, tok = load_model(args.model, dtype, device)
    units = arch.discover_units(model, tie=not args.untied, square_side=args.square_side)
    dense_total = arch.total_dense(units)
    log(f"{len(units)} units, {dense_total / 1e6:.1f}M compressible weights")

    streams = {n: (data.eval_stream(tok, n, special=False), data.eval_stream(tok, n))
               for n in _csv(args.eval_ppl)}
    sp_windows = {s: data.eval_windows_sp(tok, s, args.sp_windows, args.eval_seqlen)
                  for s in _csv(args.eval_sp)}

    kl_sets = build_kl_sets(tok, args)

    if args.mode == "dense":
        if "dense" not in metrics:
            with timer("eval_dense"):
                items = new_items()
                metrics["dense"] = evaluate_all(model, tok, args, device, streams, sp_windows, items)
                # The dense model's own NLL on the held-out sets (its KL is zero).
                res = kl_to_dense(model, kl_sets, nullcontext, nullcontext, args.eval_batch,
                                  device, items)
                metrics["dense"]["xppl"] = res["xppl_dense"]
                save_items(args.run_dir, "dense", items)
        return finish(args, metrics, save)

    train_w, val_w, bases = calibration_and_bases(args, model, units, tok, device, timer)

    # -- allocation -------------------------------------------------------------
    apath = os.path.join(args.run_dir, "alloc.json")
    A = read_json(apath)
    if A is None:
        with timer("alloc"):
            if args.alloc_from:
                A = dict(read_json(args.alloc_from))
                A["copied_from"] = args.alloc_from
            elif args.alloc == "uniform":
                A = {"alloc": "uniform", "ks": alloc.uniform_alloc(units, args.ratio)}
            else:
                cpath = os.path.join(args.run_dir, "kl_curves.json")
                curves = read_json(cpath)
                if curves is None:
                    fracs = [i / args.kl_grid for i in range(1, args.kl_grid)]
                    kw = train_w[: args.kl_windows]
                    raw = alloc.measure_kl_curves(model, units, bases, kw, fracs, args.kl_batch,
                                                  dtype, device, log)
                    curves = {u: {str(k): v for k, v in d.items()} for u, d in raw.items()}
                    write_json(cpath, curves)
                curves_i = {u: {int(k): v for k, v in d.items()} for u, d in curves.items()}
                ks, pred = alloc.greedy_alloc(units, curves_i, args.ratio)
                A = {"alloc": "kl", "ks": ks, "predicted_kl": pred}
        A["realized_ratio"] = arch.realized_ratio(units, A["ks"])
        A["n_dense_units"] = sum(1 for u in units if A["ks"][u.name] == 0)
        write_json(apath, A)
    ks = {k: int(v) for k, v in A["ks"].items()}
    log(f"allocation {A.get('alloc')}: realized ratio {A['realized_ratio']:.4f}, "
        f"{A['n_dense_units']} units dense")
    metrics["realized_ratio"] = A["realized_ratio"]

    if args.init == "oblique" or args.mode == "lora":
        if args.init == "oblique" and args.mode != "none":
            raise SystemExit("--init oblique is a training-free baseline; use --mode none")
        parts = oblique_parts(model, units, bases, ks, device, dtype)
        del bases
        if "init" not in metrics and not args.skip_init_eval:
            log("evaluating the oblique whitened truncation")
            with timer("eval_init"):
                items = new_items()
                with project.swapped(model, parts, dtype):
                    metrics["params"] = {"dense": dense_total,
                                         "stored": project.stored_params(model, units)}
                    metrics["init"] = evaluate_all(model, tok, args, device, streams, sp_windows, items)
                attach_kl(metrics["init"], kl_to_dense(
                    model, kl_sets, nullcontext, lambda: project.swapped(model, parts, dtype),
                    args.eval_batch, device, items))
                save_items(args.run_dir, "init", items)
            save()
        if args.mode == "none":
            return finish(args, metrics, save)
        return train_lora(args, model, tok, units, parts, train_w, val_w, streams, sp_windows,
                          kl_sets, metrics, save, timer, device, dtype)

    init_bases = recast_bases(units, bases, ks, device)
    del bases
    if args.save_bases:
        p = os.path.join(args.run_dir, "bases_init.pt")
        if not os.path.exists(p):
            torch.save({k: v.half().cpu() for k, v in init_bases.items()}, p)

    # -- training-free initialization --------------------------------------------
    if "init" not in metrics and not args.skip_init_eval:
        log("evaluating the initialization (merged)")
        with timer("eval_init"):
            items = new_items()
            parts = merge_parts(model, units, init_bases, dtype)
            with project.swapped(model, parts, dtype):
                metrics["params"] = {"dense": dense_total,
                                     "stored": project.stored_params(model, units)}
                metrics["init"] = evaluate_all(model, tok, args, device, streams, sp_windows, items)
            attach_kl(metrics["init"], kl_to_dense(
                model, kl_sets, nullcontext, lambda: project.swapped(model, parts, dtype),
                args.eval_batch, device, items))
            save_items(args.run_dir, "init", items)
            del parts
        save()

    if args.mode == "none":
        return finish(args, metrics, save)

    # -- training ------------------------------------------------------------------
    cfg = train.TrainConfig(mode=args.mode, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, micro_batch=args.micro_batch,
                            drop_p=args.drop_p, lam_ort=args.lam_ort,
                            alpha_epochs=args.alpha_epochs, warmup_frac=args.warmup_frac,
                            patience=args.patience, seed=args.seed,
                            epoch_size=args.epoch_size, loss=args.loss)
    if args.mode == "proj":
        bank = project.ProjectorBank(units, init_bases).to(device)
    else:
        bank = project.FactorBank(model, units, init_bases).to(device)
    bank.attach(model)
    n_trainable = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    metrics["trainable_params"] = n_trainable
    log(f"training {args.mode}: {n_trainable / 1e6:.2f}M trainable parameters")

    best_path = os.path.join(args.run_dir, "best_params.pt")
    last_path = os.path.join(args.run_dir, "last_params.pt")
    if "train" not in metrics:
        with timer("train"):
            summary = train.train(model, bank, train_w, val_w, cfg, args.run_dir, device, dtype,
                                  log, state_dir=args.state_dir)
        metrics["train"] = summary
        save()
    else:
        bank.load_raw(torch.load(best_path, map_location="cpu"))
        train.eval_mode(bank, dtype)

    def evaluate_trained(phase: str):
        """Evaluate the bank's current parameters, merged, as metrics[phase]."""
        items = new_items()
        if args.mode == "proj":
            final_bases = {k: v.to(device) for k, v in bank.export().items()}
            metrics[f"merge_check_kl{'' if phase == 'final' else '_' + phase}"] = merge_check(
                model, bank, units, final_bases, val_w, device, dtype)
            bank.detach(model)
            parts = merge_parts(model, units, final_bases, dtype)
            with project.swapped(model, parts, dtype):
                metrics[phase] = evaluate_all(model, tok, args, device, streams, sp_windows, items)
            attach_kl(metrics[phase], kl_to_dense(
                model, kl_sets, nullcontext, lambda: project.swapped(model, parts, dtype),
                args.eval_batch, device, items))
            del parts
            bank.attach(model)
        else:
            with bank.exported(model, dtype):
                metrics["params_final"] = project.stored_params(model, units)
                metrics[phase] = evaluate_all(model, tok, args, device, streams, sp_windows, items)
            attach_kl(metrics[phase], kl_to_dense(
                model, kl_sets, lambda: _active(bank, False), lambda: _active(bank, True),
                args.eval_batch, device, items))
        save_items(args.run_dir, phase, items)

    # -- trained model, merged ------------------------------------------------------
    if "final" not in metrics:
        log("evaluating the trained model (merged)")
        with timer("eval_final"):
            evaluate_trained("final")
        log(f"    merge check KL {metrics.get('merge_check_kl', float('nan')):.2e}")
        save()

    # -- fixed-checkpoint control: the last epoch, when another was selected ---------
    tr = metrics["train"]
    last_epoch = tr["epochs_run"] - 1
    if args.eval_last and "last" not in metrics and "last_same" not in metrics:
        if tr["best_epoch"] == last_epoch:
            metrics["last_same"] = True
        elif os.path.exists(last_path):
            log(f"evaluating the last epoch ({last_epoch}; epoch {tr['best_epoch']} was selected)")
            bank.load_raw(torch.load(last_path, map_location="cpu"))
            train.eval_mode(bank, dtype)
            with timer("eval_last"):
                evaluate_trained("last")
        save()

    state = os.path.join(args.state_dir or args.run_dir, "train_state.pt")
    if os.path.exists(state):
        os.remove(state)
    return finish(args, metrics, save)


def train_lora(args, model, tok, units, parts, train_w, val_w, streams, sp_windows, kl_sets,
               metrics, save, timer, device, dtype) -> int:
    """SVD-LLM's truncation with LoRA recovery on both factors (cross-entropy by default)."""
    cfg = train.TrainConfig(mode="lora", epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, micro_batch=args.micro_batch,
                            drop_p=0.0, lam_ort=0.0, alpha_epochs=args.alpha_epochs,
                            warmup_frac=args.warmup_frac, patience=args.patience,
                            seed=args.seed, epoch_size=args.epoch_size, loss=args.loss)
    bank = project.LoraBank(parts, r=args.lora_rank, alpha=args.lora_alpha, dtype=dtype,
                            seed=args.seed).to(device)
    del parts
    bank.attach(model)
    metrics["trainable_params"] = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    log(f"training lora: {metrics['trainable_params'] / 1e6:.2f}M trainable parameters")
    best_path = os.path.join(args.run_dir, "best_params.pt")
    last_path = os.path.join(args.run_dir, "last_params.pt")
    if "train" not in metrics:
        with timer("train"):
            metrics["train"] = train.train(model, bank, train_w, val_w, cfg, args.run_dir,
                                           device, dtype, log, state_dir=args.state_dir)
        save()
    else:
        bank.load_raw(torch.load(best_path, map_location="cpu"))

    def evaluate_trained(phase):
        items = {} if args.log_items else None
        merged = bank.merged_parts(dtype)
        bank.detach(model)
        with project.swapped(model, merged, dtype):
            metrics[phase] = evaluate_all(model, tok, args, device, streams, sp_windows, items)
        attach_kl(metrics[phase], kl_to_dense(
            model, kl_sets, nullcontext, lambda: project.swapped(model, merged, dtype),
            args.eval_batch, device, items))
        bank.attach(model)
        save_items(args.run_dir, phase, items)

    if "final" not in metrics:
        with timer("eval_final"):
            evaluate_trained("final")
        save()
    tr = metrics["train"]
    if args.eval_last and "last" not in metrics and "last_same" not in metrics:
        if tr["best_epoch"] == tr["epochs_run"] - 1:
            metrics["last_same"] = True
        elif os.path.exists(last_path):
            bank.load_raw(torch.load(last_path, map_location="cpu"))
            with timer("eval_last"):
                evaluate_trained("last")
        save()
    state = os.path.join(args.state_dir or args.run_dir, "train_state.pt")
    if os.path.exists(state):
        os.remove(state)
    return finish(args, metrics, save)


def reevaluate_zero_shot(args) -> int:
    """Re-run only the zero-shot evaluation of a finished run, with --zs-batch.

    Rebuilds every model the run evaluated zero-shot -- the dense model, the
    training-free initialization (from the Gram statistics of the calibration
    windows, as in the run) and the trained model (from best_params.pt) -- and
    replaces their zero-shot results. Every other metric is kept. metrics.json keeps
    the previous results under ``zs_reevaluated``; config.json records the new batch.
    """
    mpath = os.path.join(args.run_dir, "metrics.json")
    cpath = os.path.join(args.run_dir, "config.json")
    metrics, config = read_json(mpath, {}), read_json(cpath, {})
    if not metrics.get("done"):
        raise SystemExit(f"{args.run_dir}: --reeval-zs needs a finished run")
    phases = [p for p in ("dense", "init", "final") if "zs" in (metrics.get(p) or {})]
    if not phases or config.get("zs_batch") == args.zs_batch:
        log(f"{args.run_dir}: nothing to re-evaluate (zero-shot batch {config.get('zs_batch')})")
        return 0
    envs = read_json(os.path.join(args.run_dir, "env.json"), [])
    envs.append(env_info())
    write_json(os.path.join(args.run_dir, "env.json"), envs)

    device, dtype = "cuda", DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    timer = Timer()
    log(f"re-evaluating zero-shot ({', '.join(phases)}) of {args.run_dir} with batch {args.zs_batch}")
    model, tok = load_model(args.model, dtype, device)
    new = {}
    if "dense" in phases:
        with timer("zs_dense"):
            new["dense"] = zero_shot_metrics(model, tok, args)
    else:
        units = arch.discover_units(model, tie=not args.untied, square_side=args.square_side)
        _, _, bases = calibration_and_bases(args, model, units, tok, device, timer)
        ks = {k: int(v) for k, v in read_json(os.path.join(args.run_dir, "alloc.json"))["ks"].items()}
        if args.init == "oblique":
            parts = oblique_parts(model, units, bases, ks, device, dtype)
        else:
            init_bases = recast_bases(units, bases, ks, device)
            parts = merge_parts(model, units, init_bases, dtype)
        del bases
        if "init" in phases:
            log("zero-shot of the initialization (merged)")
            with timer("zs_init"), project.swapped(model, parts, dtype):
                new["init"] = zero_shot_metrics(model, tok, args)
        del parts
        if "final" in phases:
            log("zero-shot of the trained model (merged)")
            bank = (project.ProjectorBank(units, init_bases) if args.mode == "proj"
                    else project.FactorBank(model, units, init_bases)).to(device)
            bank.attach(model)
            bank.load_raw(torch.load(os.path.join(args.run_dir, "best_params.pt"), map_location="cpu"))
            train.eval_mode(bank, dtype)
            with timer("zs_final"):
                if args.mode == "proj":
                    final_bases = {k: v.to(device) for k, v in bank.export().items()}
                    bank.detach(model)
                    with project.swapped(model, merge_parts(model, units, final_bases, dtype), dtype):
                        new["final"] = zero_shot_metrics(model, tok, args)
                else:
                    with bank.exported(model, dtype):
                        new["final"] = zero_shot_metrics(model, tok, args)

    previous = {p: {k: metrics[p][k] for k in ZS_KEYS if k in metrics[p]} for p in new}
    for p, values in new.items():
        metrics[p].update(values)
    metrics["zs_reevaluated"] = {"zs_batch": args.zs_batch, "previous_zs_batch": config.get("zs_batch"),
                                 "previous": previous, "timing": timer.t,
                                 "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(mpath, metrics)
    config["zs_batch"] = args.zs_batch
    write_json(cpath, config)
    log("re-evaluated: " + ", ".join(f"{p} {previous[p]['zs_mean_acc']:.4f} -> {new[p]['zs_mean_acc']:.4f}"
                                     for p in new))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
