"""Perplexity and zero-shot accuracy of a (possibly merged) model.

Every function can also return per-item results (the NLL of each window, the
correctness of each benchmark item), which the extended evaluation uses for paired
comparisons between calibration sources.
"""

from __future__ import annotations

import math

import numpy as np
import torch

ZS_TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa", "boolq"]


@torch.no_grad()
def nll_windows(model, windows: torch.Tensor, batch_size: int, device,
                per_window: list | None = None) -> tuple[float, int]:
    """Summed next-token NLL over a batch of windows, and the number of predicted tokens.

    With ``per_window``, the summed NLL of each window is appended to it.
    """
    total, count = 0.0, 0
    for i in range(0, windows.shape[0], batch_size):
        x = windows[i:i + batch_size].to(device)
        logits = model(input_ids=x, use_cache=False).logits
        for b in range(x.shape[0]):
            lp = logits[b, :-1].float().log_softmax(-1)
            s = -float(lp.gather(-1, x[b, 1:, None]).sum())
            total += s
            count += x.shape[1] - 1
            if per_window is not None:
                per_window.append(s)
    return total, count


@torch.no_grad()
def perplexity(model, stream: torch.Tensor, seqlen: int, batch_size: int, device,
               max_windows: int | None = None, bos: int | None = None,
               per_window: list | None = None) -> float:
    """Perplexity over non-overlapping windows of a 1-D token stream.

    With ``bos``, every window is BOS followed by seqlen - 1 stream tokens, the
    format the model sees at a document start (and in calibration). Without it,
    windows are plain seqlen-token chunks (the GPTQ/SparseGPT convention, where
    only the first window begins with BOS).
    """
    body = seqlen - 1 if bos is not None else seqlen
    n = stream.numel() // body
    if max_windows is not None:
        n = min(n, max_windows)
    w = stream[: n * body].view(n, body)
    if bos is not None:
        w = torch.cat([torch.full((n, 1), bos, dtype=w.dtype), w], 1)
    total, count = nll_windows(model, w, batch_size, device, per_window)
    return math.exp(total / count)


@torch.no_grad()
def window_perplexity(model, windows: torch.Tensor, batch_size: int, device,
                      per_window: list | None = None) -> float:
    total, count = nll_windows(model, windows, batch_size, device, per_window)
    return math.exp(total / count)


def _per_item(samples: list[dict]) -> dict[str, np.ndarray]:
    """Per-item metric arrays of one lm-eval task, ordered by doc_id.

    Scalar metrics give one value per item; list-valued ones (IFEval's
    instruction-level accuracy) are kept as the per-item mean, with the count.
    Generation tasks score every filter separately: keys are ``metric,filter``.
    """
    by_key: dict[str, dict[int, float]] = {}
    for s in samples:
        doc = int(s.get("doc_id", len(by_key)))
        filt = s.get("filter", "none")
        for k, v in s.items():
            if k in ("doc_id", "doc", "target", "arguments", "resps", "filtered_resps",
                     "filter", "metrics", "doc_hash", "prompt_hash", "target_hash"):
                continue
            key = k if filt == "none" else f"{k},{filt}"
            if isinstance(v, bool) or isinstance(v, (int, float)):
                by_key.setdefault(key, {})[doc] = float(v)
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (bool, int, float)) for x in v):
                by_key.setdefault(key, {})[doc] = float(np.mean([float(x) for x in v]))
                by_key.setdefault(key + "#n", {})[doc] = float(len(v))
    out = {}
    for key, d in by_key.items():
        ids = sorted(d)
        out[key] = np.array([d[i] for i in ids], dtype=np.float32)
        out.setdefault("_doc_id", np.array(ids, dtype=np.int32))
    return out


def zero_shot(model, tokenizer, tasks=ZS_TASKS, batch_size: int = 32,
              limit: int | None = None, items: dict | None = None,
              num_fewshot: int | None = 0, chat_template: bool = False,
              gen_kwargs: str | None = None) -> dict:
    """lm-eval-harness; returns {task: {metric: value}} with stderr.

    With ``items``, the per-item results of every task are stored in it as
    ``items[task] = {metric: array}`` (see _per_item).
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model.eval()
    while True:
        # lm-eval takes a float32 log-softmax over (batch, length, vocab); with a
        # 150k vocabulary and long BoolQ/HellaSwag contexts, batch 32 needs ~12 GB.
        # Halve the batch on OOM rather than lose a finished run.
        torch.cuda.empty_cache()
        try:
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
            kw = dict(model=lm, tasks=list(tasks), num_fewshot=num_fewshot, limit=limit,
                      log_samples=items is not None, bootstrap_iters=1000,
                      verbosity="WARNING")
            if chat_template:
                kw["apply_chat_template"] = True
            if gen_kwargs:
                kw["gen_kwargs"] = gen_kwargs
            out = simple_evaluate(**kw)
            break
        except torch.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size //= 2
            print(f"zero-shot OOM; retrying with batch size {batch_size}", flush=True)
    res = {}
    for task, r in out["results"].items():
        res[task] = {k.split(",")[0] if k.endswith(",none") else k: v for k, v in r.items()
                     if "," in k and isinstance(v, (int, float))}
    if items is not None:
        for task, samples in (out.get("samples") or {}).items():
            items[task] = _per_item(samples)
    return res


def zs_mean(res: dict, metric: str = "acc", tasks=ZS_TASKS) -> float:
    vals = [res[t][metric] for t in tasks if t in res and metric in res[t]]
    return float(sum(vals) / len(vals)) if vals else float("nan")
