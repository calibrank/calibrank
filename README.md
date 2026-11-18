# When Calibration Data Become Training Data

Code and run records for the paper

> Van Tien Pham. **When Calibration Data Become Training Data: Specialization and
> Self-Generated Text in Learned Low-Rank Compression of Large Language Models.**
> *Informatics*, 13(11), Article 201 (2026).
> [https://doi.org/10.3390/informatics13110201](https://doi.org/10.3390/informatics13110201)

Low-rank compression of a language model is guided by a small calibration set. When
the compression is learned by distillation, as in learned subspace projections (LSP;
[Bini et al., 2026](https://arxiv.org/abs/2609.40127)), the calibration set becomes
the training set. The paper compresses OPT-1.3B, Llama-3.2-1B and Qwen3-1.7B by 30%
and 50%, and Llama-3.1-8B, with LSP calibrated on six sources at a matched budget
(WikiText-2, C4, SlimPajama, Alpaca, and two kinds of text sampled from the dense
model), and measures what each source preserves. In short:

- Before training, the calibration source changes perplexity by up to 600-fold.
  Training removes most of this gap but turns it into specialization: each source
  best preserves its own text, WikiText-2, the customary choice, gives the least
  general models, and zero-shot accuracy slightly favors instruction data, through
  its content rather than its template.
- Calibrating on exact samples of the dense model makes the expected distillation
  objective the sequence-level divergence between the two models, yet these samples
  fail on Qwen3-1.7B. Independent self-generated documents give the most uniformly
  faithful Llama models without any external data, and mixed half-and-half with
  SlimPajama they come close to the most faithful source on every model.
- The domain of the initialization survives training, more distinct calibration
  windows keep helping at a fixed number of steps, and compressed Llama models
  collapse on inputs that lack the beginning-of-sequence token they were calibrated
  with.
- The comparison holds over three seeds, under a window format matched across
  sources, per-source learning rates and a three times longer schedule, with a second
  learned compressor (SVD-LLM with LoRA recovery), and on Llama-3.1-8B. Fidelity is
  not utility: with every source, the compressed models lose most of their
  instruction following and multi-step reasoning.

This repository contains the reimplementation of LSP, the definition of every job
behind the paper, the records of all 459 runs, and the scripts that turn the records
into every table and figure of the paper and the statistics its text quotes.

## Contents

| Path | What it holds |
|---|---|
| `src/fsc/` | the library: compressible units, calibration data, Gram matrices, whitened initialization, rank allocation, projectors and their exact merge into low-rank factors, LoRA on factors, distillation, evaluation, self-generated text, pinned Hub revisions |
| `scripts/` | entry points: one compression run (`compress.py`), self-generated calibration (`generate_calib.py`), behavioral evaluation (`behave.py`), memory and latency (`bench.py`), the experiment plans and their runner (`plans.py`, `submit.py`), reporting (`aggregate.py`, `make_report.py`, `make_report_ext.py`, `run_stats.py`, `claims.py`) |
| `records/` | the raw measurements: one directory per run, plus the per-source statistics, the subspace overlaps and the analyses |
| `results/` | everything derived from `records/`: run tables, the LaTeX tables and PDF figures of the paper, its statistics and the numbers its text quotes |
| `slurm/` | environment setup and the Slurm job runner |
| `tests/` | CPU tests: exact merging, initialization identities, gradients, window formats and objectives, plans against records, results against records |

The runs come in two sets, which the paper describes in Section 3.5.4. The **main
grid** and its ablations are single runs evaluated on 32 held-out windows per
source (`records/runs/`, `results/`). The **extended evaluation**, which comprises
the three-seed repetition of the grid, the controls of Section 3.4, the Llama-3.1-8B
experiments at both ratios and the instruction-tuned model, uses 128 held-out
windows per source and six independently chosen domains, and records the result of
every evaluation item (`records/extended/`, `results/extended/`).

## Installation

Requirements: Linux, [uv](https://docs.astral.sh/uv/), and for GPU runs an NVIDIA GPU
with bfloat16 support and a driver >= 525.

```bash
git clone https://github.com/calibrank/calibrank.git
cd calibrank
make setup          # uv sync from uv.lock, then the CPU tests
```

`uv.lock` pins every package, including torch 2.6.0+cu124, transformers 4.57.6 and
lm-eval 0.4.9.2, the versions the experiments ran with. Commands below use
`uv run --no-sync`, which runs in that environment without re-resolving it.

## Reproducing the paper from the records (CPU)

```bash
make results        # records/ -> results/ (about a minute)
make check          # results/ is byte-identical to what records/ produce
make stats          # recompute results/extended/run_stats.json (about ten minutes)
```

`make results` runs `aggregate.py` (run tables), `make_report.py` (tables and figures
of the main grid), `claims.py` (the numbers the Results text states about the main
grid, written to `results/claims.txt`) and `make_report_ext.py` (tables and figures
of the extended evaluation). `run_stats.py` computes the seed statistics, the 95%
paired-bootstrap intervals over evaluation items and the ranking probabilities that
the paper quotes, in `results/extended/run_stats.json`; `make check` reads the
committed file, and `pytest -m slow` recomputes it as well. The files map to the
paper as follows.

| Paper | File in `results/` | Paper | File in `results/` |
|---|---|---|---|
| Table 1 | `tables/sources.tex` | Table 17 | `extended/tables/paired.tex` |
| Table 2 | `tables/oblique.tex` | Table 18 | `extended/tables/scale.tex` |
| Table 3 | `tables/main.tex` | Table 19 | `extended/tables/behavior.tex` |
| Table 4 | `extended/tables/alpaca.tex` | Table 20 | `extended/tables/inst.tex` |
| Table 5 | `tables/klsummary.tex` | Table 21 | `extended/tables/bench.tex` |
| Table 6 | `extended/tables/uniform.tex` | Table 22 | `extended/tables/bosall.tex` |
| Table 7 | `extended/tables/mix.tex` | Table A1 | `tables/tasks.tex` |
| Table 8 | `tables/domain.tex` | Table A2 | `extended/tables/boolq.tex` |
| Table 9 | `tables/ablations.tex` | Tables A3, A4 | `extended/tables/seeds_r30.tex`, `seeds_r50.tex` |
| Table 10 | `extended/tables/format.tex` | Table A5 | `extended/tables/newdomains_r50.tex` |
| Table 11 | `tables/alloc.tex` | Table A6 | `tables/repro.tex` |
| Table 12 | `extended/tables/qwen.tex` | Table A7 | `extended/tables/initcmp.tex` |
| Table 13 | `extended/tables/composition.tex` | Table A8 | `extended/tables/sqout.tex` |
| Table 14 | `tables/factors.tex` | Table A9 | `tables/lr.tex` |
| Table 15 | `extended/tables/lora.tex` | Table A10 | `tables/bos.tex` |
| Table 16 | `extended/tables/optim.tex` | | |

| Paper | File in `results/` | Paper | File in `results/` |
|---|---|---|---|
| Figure 1 | `figures/entropy.pdf` | Figure 7 | `extended/figures/domains_8b.pdf` |
| Figure 2 | `figures/dumbbell_r30.pdf` | Figure A1 | `figures/dumbbell_r50.pdf` |
| Figure 3 | `figures/kl_matrix_r50.pdf` | Figure A2 | `figures/kl_matrix_r30.pdf` |
| Figure 4 | `figures/domains_r30.pdf` | Figure A3 | `figures/domains_r50.pdf` |
| Figure 5 | `figures/size.pdf` | Figure A4 | `figures/subspaces_r50.pdf` |
| Figure 6 | `extended/figures/curves.pdf` | | |

The tables are the LaTeX bodies inserted into the paper's `tabular` environments.
`results/runs.csv` and `results/extended/runs.csv` have one row per run with its
configuration and every metric (`init_*`: training-free initialization, `final_*`:
trained model, `last_*`: the last epoch when another one was selected; `ppl_*`:
perplexity on BOS-prefixed windows, `pplstream_*`: GPTQ-style windows, `sp_*`:
SlimPajama domains, `zs_*`: zero-shot accuracy, `kl_*`: divergence from the dense
model on held-out sets and on the six independent domains, `gen_*`: generative tasks).

## Re-running the experiments (GPU)

The 182 runs of the main grid took about 80 GPU-hours on NVIDIA A100 (40 and 80 GB),
H100 and H200 GPUs, and the 277 runs of the extended evaluation about 245 GPU-hours on
A100 GPUs (40 and 80 GB), behavioral evaluation included, plus a few hours to sample
the self-generated calibration sets. Every job needs one GPU with bfloat16 and at
least 40 GB of memory; the Llama-3.1-8B jobs need 80 GB.

**1. Data and models.** Accept the license of the gated
[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B),
[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and
[Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) repositories, log in
(`uv run --no-sync huggingface-cli login`), then, on a machine with internet access:

```bash
uv run --no-sync python scripts/prefetch.py    # models, corpora and benchmarks, ~30 GB
```

Models and the four calibration corpora are loaded at the exact Hub commits the
experiments used, pinned in `src/fsc/hub.py`, so later changes to those repositories
do not affect a re-run; the other repositories are loaded from their default branch.

**2. Check the GPU and the pipeline.**

```bash
uv run --no-sync python scripts/verify_gpu.py              # the torch build runs kernels here
uv run --no-sync python scripts/submit.py smoke --local    # a 2-minute end-to-end run
```

**3. Run the plans.** `scripts/plans.py` defines every job of the paper as named plans;
`all` runs them in dependency order. On a Slurm cluster:

```bash
export FSC_PARTITIONS=<partitions with >= 40 GB GPUs>     # optional; Slurm's default otherwise
export FSC_PARTITIONS_80G=<partitions with >= 80 GB GPUs> # for Llama-3.1-8B
uv run --no-sync python scripts/submit.py all --list      # what is done, blocked, to run
uv run --no-sync python scripts/submit.py all             # submit; re-run to top up the queue
```

`submit.py` keeps at most `--max-jobs` (default 12) of your jobs queued, skips finished
jobs, and holds a job until the files it needs exist (for example a self-generated
calibration set), so running it again is always safe. Jobs run through
`slurm/run.slurm` with `HF_HUB_OFFLINE=1`, for compute nodes without internet access.
Without Slurm, add `--local` to run the jobs one after another on this machine.
`--only` selects jobs by name; a blocked job names the file it is waiting for. For
example, one run of the main grid, after the held-out self-samples it is evaluated on:

```bash
uv run --no-sync python scripts/submit.py selfgen --only llama-3.2-1b-T1.0-s1000 --local
uv run --no-sync python scripts/submit.py main --only llama-3.2-1b-proj-c4-r50 --local
```

**4. Compare.** Each run writes `out/runs/<model>/<run>/` (main grid) or
`out/extended/runs/<model>/<run>/` (extended evaluation), in the layout of
`records/`; source statistics go to `out/stats/`, subspace overlaps to
`out/subspaces/`, and the analyses of the extended evaluation to
`out/extended/analysis/`. Compare your runs with the records, and rebuild the
paper's tables and figures from them:

```bash
uv run --no-sync python scripts/compare_records.py                                    # main grid
uv run --no-sync python scripts/compare_records.py --mine out/extended --records records/extended
make results RECORDS=out RESULTS=out/results                                        # tables, figures, claims
```

A run that stops (time limit, preemption) resumes from its last completed phase when
resubmitted; training checkpoints every epoch.

| Plan | Jobs | Paper | Plan | Jobs | Paper |
|---|---|---|---|---|---|
| `dense` | 4 | reference rows | `ext-pools`, `ext-pools8b`, `ext-inst-pools` | 19 + 6 + 3 | self-generated sets |
| `selfgen`, `selfdoc` | 11 + 5 | self-generated sets | `ext-dense` | 5 | reference rows |
| `stats` | 3 | Table 1, Figure 1 | `ext-main` | 108 | Tables 6, 17, A2-A5 |
| `pilot-lr` | 12 | Table A9 | `ext-scale`, `ext-scale50` | 20 + 6 | Table 18, Figure 7 |
| `pilot-opt125m` | 12 | Appendices B-C | `ext-format` | 15 | Table 10 |
| `main` | 36 | Tables 3, 5, Figures 2-4 | `ext-alpaca` | 9 | Table 4 |
| `oblique` | 36 | Table 2 | `ext-optim`, `ext-subspaces` | 20 + 1 | Table 16, Section 4.8 |
| `subspaces` | 6 | Figure A4 | `ext-size` | 10 | Figure 6 |
| `seeds` | 12 | seeds 1 and 2 of Llama-3.2-1B at -50% | `ext-mix` | 6 | Table 7 |
| `size`, `fresh` | 9 + 2 | Figure 5 | `ext-qwen`, `ext-selfstats*` | 8 + 2 | Tables 12, 13 |
| `temperature` | 2 | Table 9 | `ext-lora` | 12 | Table 15 |
| `factors` | 12 | Table 14 | `ext-initcmp`, `ext-sqout` | 42 + 11 | Tables A7, A8 |
| `klalloc` | 6 | Table 11 | `ext-inst` | 5 | Table 20 |
| `domain` | 3 | Table 8 | `ext-behave` | 46 | Table 19 |
| `crossover`, `bos` | 4 + 5 | Table 9, Table A10 | `ext-bench` | 4 | Table 21 |
| `repro`, `diag-oblique`, `repro-init` | 6 + 3 + 9 | Table A6 | `smoke` | 1 | installation check |
| `scale` | 12 | Table 22 | | | |

### What to expect from a re-run

`tests/test_plans.py` checks that the plans define exactly the published runs, with the
configuration recorded in each `config.json`. Re-running five records of the main grid
from a fresh clone gave these differences from the records
(`scripts/compare_records.py` prints the same comparison for your own runs):

| Run | WikiText-2 perplexity | C4 perplexity | SlimPajama perplexity | Mean accuracy |
|---|---|---|---|---|
| `opt-125m/dense` | +0.00% | +0.00% | +0.00% | +0.07 points |
| `opt-1.3b/dense` | +0.00% | +0.00% | +0.00% | -0.11 points |
| `llama-3.2-1b/dense` | +0.00% | +0.01% | +0.00% | -0.04 points |
| `qwen3-1.7b/dense` | +0.00% | +0.00% | +0.00% | +0.00 points |
| `llama-3.2-1b/proj-c4-r50-s0` | +4.33% | -0.22% | +0.88% | -0.14 points |

Evaluation reproduces exactly on the same GPU architecture, and within about 0.1
accuracy points across architectures, with perplexities equal to 0.01%. Training is
not bit-for-bit reproducible on GPUs, whose backward kernels are not deterministic:
on the same GPU type as its record, the trained run reached a best validation KL
within 0.2% of the record and matches it on its calibration domain (C4), while its
WikiText-2 perplexity differs by 4.3%, of the order of the seed-to-seed variation
reported in the paper. The first seed of the extended evaluation repeats the main
grid on different GPUs; its perplexities differ from the main grid's by a median of
0.9% and its mean accuracy by a median of 0.16 points (Section 4.9).

## Records

`records/runs/<model>/<run>/` (main grid) and `records/extended/runs/<model>/<run>/`
(extended evaluation) hold, for every run:

| File | Content |
|---|---|
| `config.json` | every argument of `scripts/compress.py` |
| `metrics.json` | `init` and `final` metrics (perplexities, SlimPajama domains, zero-shot accuracy, KL to the dense model), the training summary and history, `merge_check_kl` (projected vs merged model), parameter counts, time and peak memory per phase; for the extended evaluation also the last-epoch metrics, the six independent domains and the generative tasks |
| `alloc.json` | the number of directions removed from every unit, and the realized ratio |
| `train_log.jsonl` | train and validation KL after every epoch |
| `kl_curves.json` | measured-KL allocation only: the KL of every unit at every removal fraction |
| `items_init.npz`, `items_final.npz`, `items_last.npz` | extended evaluation: the result of every evaluation item (each zero-shot question, each perplexity and KL window), for the paired comparisons |
| `behave.json` | extended evaluation, seed 0: open-ended generation, long-context perplexity, passkey retrieval and generative tasks |
| `env.json` | software versions, GPU, driver |

The zero-shot results of 33 runs of the main grid, first evaluated with an lm-eval
batch size of 32, were re-evaluated at the final size of 16
(`scripts/compress.py --reeval-zs`, which rebuilds the initialization and the trained
model and reruns only the zero-shot tasks), so that every record uses the same
evaluation setting; their `metrics.json` keeps the earlier results under
`zs_reevaluated`.

Run names read `<mode>-<source>-r<ratio>-s<seed>[-<variant>]`: `proj` learned
projectors, `none` training-free truncation, `factors` free low-rank factors, `lora`
the truncation of SVD-LLM recovered with LoRA, `dense` the uncompressed model.
`records/stats/` holds the per-source statistics of Table 1 and Figure 1,
`records/subspaces/` the overlaps of Figure A4, and `records/extended/analysis/` the
memory and latency measurements (Table 21), the composition of the self-generated text
(Table 13) and the subspace overlaps of the longer schedule (Section 4.8). The trained
projectors (`best_params.pt`) and initial bases are not included: they are regenerated
by the runs, and every reported number is in the records.

## Code

| Module | Role |
|---|---|
| `fsc.arch` | which linear layers form units (tied query/key/value and gate/up groups), and their parameter costs |
| `fsc.data` | calibration windows for every source and window format, evaluation streams, held-out KL sets and independent domains |
| `fsc.stats` | input and output Gram matrices of every unit |
| `fsc.init` | whitened bases: the orthogonal recast that initializes LSP, and the oblique truncation of SVD-LLM |
| `fsc.alloc` | uniform and measured-KL rank allocation |
| `fsc.project` | projectors, the projector bank, the exact merge into low-rank factors, free factors, LoRA on factors |
| `fsc.train` | distillation of the projectors against the dense model, and the cross-entropy objective of LoRA recovery |
| `fsc.evaluate` | perplexity, zero-shot accuracy and generative tasks (lm-eval), with per-item results |
| `fsc.selfgen` | text sampled from the dense model: exact sequences, independent documents, responses to instructions |
| `fsc.hub` | pinned Hugging Face revisions of every model and corpus |

`scripts/compress.py --help` documents a single run. `scripts/status.py` prints the
validation-KL trajectory of runs in progress, `scripts/compare_records.py` compares
finished runs with the records, and `scripts/passkey_probe.py` prints what a trained
model writes after the passkey prompt (the diagnostic of Section 4.11).

## Citation

If you use this code or these records, please cite the paper:

```bibtex
@article{pham2026calibration,
  author          = {Pham, Van Tien},
  title           = {When Calibration Data Become Training Data: Specialization and
                     Self-Generated Text in Learned Low-Rank Compression of Large
                     Language Models},
  journal         = {Informatics},
  volume          = {13},
  year            = {2026},
  number          = {11},
  article-number  = {201},
  url             = {https://www.mdpi.com/2227-9709/13/11/201},
  issn            = {2227-9709},
  doi             = {10.3390/informatics13110201}
}
```

The compression method is learned subspace projections, reimplemented here:

```bibtex
@misc{bini2026lsp,
  author        = {Bini, Massimo and Christensen, Anders and Alaniz, Stephan and
                   Goldfeder, Judah and Winther, Ole and LeCun, Yann and
                   Shwartz-Ziv, Ravid and Akata, Zeynep},
  title         = {Learning Functional Subspaces for Neural Network Compression},
  year          = {2026},
  eprint        = {2609.40127},
  archivePrefix = {arXiv},
}
```

## License

The code and the records are released under the [Apache License 2.0](LICENSE). The
models and datasets used by the experiments are not redistributed here and keep their
own licenses (in particular the Llama 3.1 and 3.2 community licenses).
