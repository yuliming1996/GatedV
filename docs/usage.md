# Detailed usage

## Quick start

Python 3.10+ on Linux or macOS:

```sh
git clone https://github.com/yuliming1996/GatedV.git
cd GatedV
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -B tools/reproduce.py --study smoke
```

The last command runs a tiny synthetic CPU check. For a full study, install a
CUDA-compatible PyTorch build and choose one of:

```sh
python -B tools/reproduce.py --study small --download
python -B tools/reproduce.py --study medium --download
```

These commands prepare the data, train all paired seeds, evaluate the cohort,
and compute statistics: 10 runs for small or 6 for medium. Downloads require
`curl`. Add `--dry-run` to inspect the plan. Commands below run from the repository root.

## Recipes and outputs

| Study | Width / layers / heads | Paired seeds | Updates per run | Peak LR |
|---|---|---|---:|---:|
| Small | 128 / 6 / 4 | 20261201–20261205 | 32,768 | 0.0048 |
| Medium | 384 / 12 / 6 | 20270211–20270213 | 327,680 | 0.0006 |

Both use batch 8, context 256, an 8,192-token vocabulary, FP32, math SDPA and
TF32 disabled. The complete recipes are in [configs](../configs/). `v_silu`
is the code name for GatedV. These commands reproduce the selected confirmation
recipes, not the earlier searches or mechanism controls.

Default locations are `data/raw/`, `data/paper_small/` or `data/paper_medium/`,
and `outputs/reproduction_small/` or `outputs/reproduction_medium/`.
Override them with `--raw`, `--data-root`, and `--output`:

```sh
python -B tools/reproduce.py --study small --download \
  --raw /path/to/cache --data-root /path/to/prepared \
  --output /path/to/results
```

The full studies default to `--device cuda:0`; smoke defaults to CPU.
A successful run writes `complete.json`; its `comparison` field points to the
statistics file. Per-document loss arrays are retained under `evaluation/`.

## Resume

Repeat the same command with `--resume`. Code, recipe, data and environment must
match the saved identity. Completed runs are checked before being skipped;
incomplete runs resume from their checkpoints. SIGINT/SIGTERM requests saving
a checkpoint after the current update (exit code 75). Unknown errors stop the
workflow for diagnosis, and incomplete cohorts are not evaluated.

## Data preparation only

```sh
python -B tools/prepare_data.py --study small --download --output data/paper_small
python -B tools/prepare_data.py --study medium --download --output data/paper_medium
```

The tool uses pinned FineWeb-Edu shards, the bundled tokenizer, and frozen
accepted rows from [data_plan.json](../reproduction/data_plan.json). Downloaded
shards and reconstructed arrays must match their recorded hashes. This replays
the original selections; it does not repeat the historical deduplication audit.
The published holdouts should not be treated as newly blind tests for later tuning.

Small needs about 2.69 GB of downloaded shards and 0.10 GB of prepared files;
medium needs about 6.46 GB and 1.06 GB. Add space for checkpoints and dependencies.
Omit `--download` to use verified cached shards; `--raw` selects their location.
Interrupted downloads can resume. Failed encoding leaves an `OUTPUT.partial`
directory; after diagnosing the failure, use a new output path or explicitly
remove that partial directory. Encoding itself restarts rather than resumes.

Successful preparation creates `STUDY_corpus/` and `STUDY_holdout/`. Training
arrays are `tokens_train.npy` / `tokens_dev.npy` (`uint16`) and
`docs_train.npy` / `docs_dev.npy` (`int64` half-open document boundaries).
The tokenizer uses `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`; preparation appends
one EOS per document without adding BOS. The evaluator uses separate holdout
arrays bound by [holdout descriptors](../configs/).

To recheck prepared data without encoding:

```sh
python -B tools/prepare_data.py --study small --output data/paper_small --verify-only
```

Historical paths in provenance JSON identify original evidence; they are not
runtime dependencies. Replacing the tokenizer or corpus defines a new experiment.
The original affine-control calibration coefficients are not bundled, so those
controls require separately prepared calibration rather than the quick-start commands.

## Individual runs and GLU controls

For a single small-model run (not the complete paired comparison):

```sh
python -B code/train.py --config configs/small.json --arm v_silu \
  --data data/paper_small/small_corpus --output runs/small_single
```

GLU uses `(X W_C) * SiLU(X W_G)` with separate content and gate projections.
Its [four-arm plan](../experiments/glu_attention/configs/scientific_plan.json)
tests `{6e-4, 3e-4, 1e-4, 1e-5}` at seed 20270201, 327,680 updates per run.
The full GLU has 27,927,936 parameters; reduced GLU has 26,186,112 versus
26,158,464 for baseline/GatedV. Reduced value width 258 preserves six heads and
is approximately, not exactly, parameter matched.

```sh
python -B experiments/glu_attention/inspect_controls.py
python -B experiments/glu_attention/code/train.py \
  --config experiments/glu_attention/configs/lr_controls_20261013/medium_v_glu_full_lr6e-4.json \
  --data data/paper_medium/medium_corpus --output runs/glu_full_lr6e-4
```

These are individual run commands, not an automatic 16-run queue. GLU checkpoints
use `glu_controls_v1`; the root runtime uses `portable_v1`. Use the matching
runtime's evaluation and benchmark tools; checkpoints are not interchangeable.
The current GLU grid supports development-set LR selection, not a new holdout
or multi-seed confirmation. See each program's `--help` for other options.

## Reading the evidence

[results/verification.json](../results/verification.json) records numerical
reanalysis of 133 historical endpoints, 77 summary cells, medium holdout arrays,
42 timing slots, learning curves, and the separate five-endpoint GLU snapshot.
This checks retained numbers and file integrity, not the original training execution.

Study folders A/B contain the initial screen and unsuccessful depth-reduction
experiments; C/E/F contain branch, learning-rate and mechanism explorations;
D/M contain the small/medium confirmations. Learning curves are development
PPL, not held-out PPL. Figure means average per-seed PPL; paired differences
are GatedV minus baseline, so negative values favor GatedV. The paper contains
the interpretation and uncertainty estimates. Timing uses synchronized elapsed
measurements; training duration is not a separate forward-latency benchmark.

## Paper and figures

Compile the paper with the included figures (requires LaTeX):

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

To regenerate figures, run from the repository root:

```sh
python -m pip install -r requirements-figures.txt
python -B paper/make_paper_figures.py
```
