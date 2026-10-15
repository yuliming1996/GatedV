# Running the reference implementation

The release contains two separate runtimes: `code/` for the original GatedV
experiments, and `experiments/glu_attention/code/` for the GLU comparison.
They share numerical conventions but use different checkpoint identities.
Neither accepts checkpoints from the other runtime automatically.

The model, data sampling, training, evaluation and statistical source files are
preserved byte for byte from the research package. Their hashes and adaptation
history are in [`code/source_manifest.json`](../code/source_manifest.json).
Original machine-specific launchers, temporary checks and execution logs are
excluded. Packaging does not rerun the scientific experiments.

## One command per study

Install `requirements-reproduce.txt` in an isolated environment. Start with:

```sh
python -B tools/reproduce.py --study smoke
python -B tools/reproduce.py --study small --dry-run
```

The first command exercises training and holdout evaluation using synthetic data
and a tiny model on CPU. It is only an installation check. The second prints the
complete small-model plan without writing files, downloading or running models.

For a full paper confirmation, choose one:

```sh
python -B tools/reproduce.py --study small --download
python -B tools/reproduce.py --study medium --download
```

Each command prepares data, declares a paired cohort, runs every full-budget
trajectory in order, verifies every endpoint before evaluation, and computes the
original study's statistics. Small means ten runs at five paired seeds; medium
means six runs at three paired seeds. Neither command repeats the original
learning-rate search. The GLU follow-up remains a separate experiment.

Defaults are resolved from the script's repository, independently of the current
working directory: prepared data in `data/paper_STUDY/`, raw shards in
`data/raw/`, and run/evaluation outputs under `outputs/reproduction_STUDY/`.
Use `--data-root`, `--raw` and `--output` to choose other locations. Relative
values explicitly supplied to these options are relative to your current
working directory. The same entrypoint works after moving the entire checkout.
For example, reuse a previously verified cache and choose a new result folder:

```sh
python -B tools/reproduce.py --study small --download \
  --raw data/raw --output outputs/small_replication
```

The full presets default to `--device cuda:0`; CPU may be selected explicitly but
is impractical for these budgets. The smoke preset defaults to CPU. Software and
device identities are bound into each run, so changing device during a resume is
not supported. Use a PyTorch build appropriate to your CUDA platform.

Add `--resume` to the same command after an interruption. A completed endpoint is
verified before it is skipped. An incomplete run resumes its checkpoint with
matching recipe, data, code and environment; exit code 75 stops the coordinator
without evaluating an incomplete cohort. Unknown errors stop for diagnosis.
Partial evaluation attempts are retained and a retry writes a new attempt.
Data preparation failures retain a `.partial` directory; follow the
[preparation recovery instructions](../reproduction/README.md) before retrying.

A successful run writes `complete.json` with the cohort, result and evaluation
hashes. The `comparison` field names the statistics file (normally
`analysis/comparison_001.json`): it includes per-seed NLL/PPL, mean PPL, paired
changes and the original uncertainty calculations. `evaluation/attempt_001/`
retains per-document arrays. Failed or interrupted attempts remain available
for diagnosis. Report the generated comparison even if it differs from the
published outcome.

The commands below are the lower-level interfaces used by the coordinator. They
are useful for separate experiments; a new corpus or changed recipe should not
be described as an exact reproduction of the published data and settings.

## Environment

Use Python 3.10 or newer and install `requirements-reproduce.txt` (or just `requirements.txt` for already-prepared data) in an isolated
environment. PyTorch and NumPy are required by both runtimes; `psutil` is also
required by the GLU trainer. CUDA requires a compatible PyTorch installation.
The runners use Unix file locking and support Linux/macOS with CPU or CUDA;
Windows and MPS are not supported. CPU is suitable for small technical checks.
The full supplied recipes are substantial training jobs.

All commands below run from the repository root. Set
`PYTHONDONTWRITEBYTECODE=1` and use `python -B` to avoid bytecode files:

```sh
export PYTHONDONTWRITEBYTECODE=1
python -B code/train.py --help
python -B code/evaluate.py --help
python -B code/analyze.py --help
python -B code/benchmark.py --help
python -B experiments/glu_attention/inspect_controls.py
```

The scripts use FP32, mathematical SDPA, deterministic algorithms, no autocast,
no TF32 and no compilation. AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8`,
matrix weight decay `0.1`, zero decay on vectors and `foreach=False`.
Global gradient norm is clipped at `1.0`. Learning rates warm up linearly and
decay with a cosine schedule to 10% of the peak. These settings describe the
reported implementation, not recommended defaults for every model. The main paper comparisons use
name-seeded fan-in Gaussian initialization, including paired initial weights.
A historical fixed-0.02 initialization mentioned in an inherited source comment
belongs to an earlier experiment and is not used by these recipes.

## Original GatedV recipes

The supplied confirmation presets are:

| Preset | Width / layers / heads / FFN | Batch / context | Updates | Training targets | Peak LR / warmup |
|---|---|---|---:|---:|---|
| `configs/small.json` | 128 / 6 / 4 / 384 | 8 / 256 | 32,768 | 67,108,864 | 0.0048 / 512 |
| `configs/medium.json` | 384 / 12 / 6 / 1152 | 8 / 256 | 327,680 | 671,088,640 | 0.0006 / 5,120 |

Both use an 8,192-token vocabulary. The larger model's original LR search
selected the lowest tested point, 6e-4; the lower-LR follow-up is separate and
still in progress in this release. The presets should not be read as globally
optimal learning rates.

First reconstruct the main paper datasets using the [preparation guide](../reproduction/README.md), or provide compatible prepared arrays described in [Data](data.md). For example,
with the appropriate corpus under `data/paper_medium/medium_corpus/`:

```sh
python -B code/train.py --config configs/medium.json \
  --data data/paper_medium/medium_corpus --output runs/medium/baseline_s20270211
python -B code/train.py --config configs/medium.json \
  --data data/paper_medium/medium_corpus --arm v_silu --output runs/medium/v_silu_s20270211
```

For the declared medium cohort, repeat for seeds `20270212` and `20270213`
using `--seed`. The small cohort uses seeds `20261201` through `20261205`,
`configs/small.json` and output paths `runs/small/ARM_sSEED`.
`v_silu` is the implementation name for GatedV. Other supported projection
controls remain in the model, including those with negative experimental
outcomes. `configs/historical_recipes.json` records historical recipes and
budgets; it is descriptive metadata, not a direct training CLI configuration.

The `v_affine_match` and `v_silu_standardized` controls require an externally
prepared `--calibration FILE`. It must contain one row per layer with
`layer`, `a`, `b`, `c`, `d`, or a manifest containing
`by_seed[SEED].constants` in that form. The layer indices start at zero.
The coefficients must be frozen from train-only baseline initialization;
the release does not generate them or include the historical calibration
payload. These controls cannot be reproduced by omitting the coefficients.

## Checkpoints and resuming

The root runtime writes schema `portable_v1`. Its identity binds the complete
recipe, initial parameter fingerprint, data hashes, executable source hashes,
calibration and software/device environment. Each run retains its sampled
training/development schedules, per-window development losses, model and
optimizer states, RNG states and update ledger. `result.json` is committed only
after the final update, development evaluation and data hash checks.

To resume a run, repeat its original command with `--resume`. The recipe,
data, code, environment and initialization must still match. SIGINT/SIGTERM
requests checkpointing after the current update; an incomplete run exits with
code 75. A hard interruption may lose updates after the latest checkpoint.
The portable runner does not include the research machine's automatic queue
or system capacity supervisor. Only load trusted checkpoints, since optimizer
and RNG state use PyTorch's pickle-based format.

## Evaluation and analysis

`configs/small_cohort.json` and `configs/medium_cohort.json` declare the complete
paired confirmation cohorts. Run paths are relative to the cohort JSON.
If settings or locations change, declare a new cohort with each run's exact
`identity.config` before evaluating it. The supplied holdout descriptors bind
the original data hashes, not arbitrary replacement data.

After all endpoints in the declared cohort finish, and with the matching
holdout payload supplied separately:

```sh
python -B code/evaluate.py --cohort configs/medium_cohort.json \
  --holdout data/paper_medium/medium_holdout --seal configs/medium_holdout.json \
  --output outputs/medium_holdout
python -B code/analyze.py --evaluation outputs/medium_holdout --study medium \
  --domains data/paper_medium/medium_holdout/documents_holdout.jsonl \
  --output outputs/medium_statistics.json
```

Use `--study small` for the five-seed small-model cohort. The evaluator checks
all declared endpoints before opening holdout files, then saves loss sums,
target counts, document IDs and offsets. Analysis recomputes the summaries
from those arrays. Three/five-seed intervals and conditional cluster bootstrap
serve different purposes; the bootstrap is conditional on the observed seeds.
For one document per domain, document and domain partitions coincide.

## Timing

```sh
python -B code/benchmark.py --cohort configs/medium_cohort.json \
  --output outputs/medium_benchmark
```

The default benchmark uses seven interleaved rounds, alternating model order,
rotating seed order, three warmups and ten timed repetitions per mode. It
measures full-context forward passes and optimizer steps on synthetic tokens,
with device synchronization around wall-clock timings. Training steps modify
disposable copies, while endpoint checkpoint hashes are checked before and
afterwards. These are measured durations, not GPU-throughput-normalized values
or autoregressive decoding speeds. Temperature, clocks and shared-host load
can affect them. Raw observations should accompany any performance claim.

The root portable benchmark records CUDA peak allocation but omits historical
process/RSS sampling; its output is not a replacement for the archived memory
measurements. See the [GLU comparison](../experiments/glu_attention/README.md)
for its separate runtime and current study boundary.
