# GLU attention comparison

This follow-up compares GatedV with a separately projected GLU value branch
in the same decoder backbone. It tests the operator described by
[Wang et al.](https://arxiv.org/abs/2507.00022v2), with
[the authors' implementation](https://github.com/WangZehaoAI/GLU-Attention)
inspected at commit `e6986495b96d484894b1f780ac22b6fc2f67cebd`.
The code here independently implements that operator; it does not vendor
the external source or reproduce that paper's complete training setup.

The 16-run development-set LR comparison is **in progress** as of 2026-10-15.
The [results snapshot](../../results/glu_lr_controls_20261015.json) distinguishes
completed endpoints from unfinished jobs. It is not a held-out or multi-seed
confirmation, and it does not yet establish each arm's best learning rate.

## Operators and parameter counts

For input states `X`, GatedV aggregates `SiLU(X W_V)`. The GLU controls aggregate
`(X W_C) * SiLU(X W_G)`, using separate content and gate projections.
The attention-weighting rule and the common backbone are unchanged. Linear layers
are bias-free throughout this comparison.

| Arm | Value width / head | Parameters | Difference from GatedV |
|---|---:|---:|---:|
| `baseline` | 384 / 64 | 26,158,464 | 0 |
| `v_silu` (GatedV) | 384 / 64 | 26,158,464 | 0 |
| `v_glu_full` | 384 / 64 | 27,927,936 | +1,769,472 (+6.764%) |
| `v_glu_reduced` | 258 / 43 | 26,186,112 | +27,648 (+0.106%) |

The reduced control is **near parameter matched**, not exactly matched.
The original V/output pair has `2 d²` weights per layer; GLU has `3 d m`.
Exact equality requires `m=2d/3=256`, which cannot be split across six heads.
The declared `nearest` policy therefore uses 258. No compensating FFN change
or unused parameter padding is applied. Other width policies are implemented
for explicit use, but are not additional runs in this study.

Shared-shape matrices use paired name-based initialization. Full GLU's
content projection matches the original V initialization, while its gate has
a separate initialization. Reduced GLU changes the V/output shapes and
output fan-in; its entire initial state is not identical to GatedV.

## Scientific plan

[`configs/scientific_plan.json`](configs/scientific_plan.json) exports the
actual selection order, recipe hashes, data hashes and initial parameter
fingerprints. It is a portable description, not an automatic queue launcher.
The 16 recipe files are unchanged from the executed study. Their `note` fields
retain the historical preparation wording; they are ignored by the recipe
loader and do not report current execution status.

Each arm uses peak LRs `6e-4`, `3e-4`, `1e-4` and `1e-5`, paired seed
`20270201`, 327,680 updates, batch 8, context 256, and 671,088,640 training
targets. The execution order is LR blocks in that order, with baseline,
GatedV, full GLU and reduced GLU within each block. All runs use the same
token schedule, FP32/math SDPA policy, 5,120-step warmup and cosine decay to
10% of peak LR. Total planned cost is 5,242,880 updates and 10,737,418,240
targets, excluding technical checks.

After all runs finish, each arm's LR is selected by minimum final full-dev
NLL; exact ties choose the lower LR. A boundary winner is reported as such.
The original medium grid was `6e-4, 1.2e-3, 2.4e-3`; both models preferred its
lower boundary. The present follow-up addresses that unresolved range and
must be distinguished from the original experiment. No holdout evaluation,
additional confirmation seed or automatic grid extension belongs to this
plan.

## Running the code

Install the root dependencies first. From this directory, inspect shapes and
counts without data, training or CUDA:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B inspect_controls.py
```

For a new, explicitly selected endpoint with your prepared train/dev arrays:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B code/train.py \
  --config configs/lr_controls_20261013/medium_v_glu_full_lr6e-4.json \
  --data ../../data/medium_corpus \
  --output runs/v_glu_full_s20270201_lr6e-4
```

The same interface is available for all 16 supplied recipes. Data and
checkpoint payloads are not included; see [Data](../../docs/data.md).
`--resume` requires exactly matching run identity. The runtime retains
finite model/optimizer checks, atomic checkpointing, resource sampling and
a 12 GiB per-process CUDA allocator ceiling. Machine-specific scheduling and
system capacity supervision are not included in this release.

Checkpoints use **`glu_controls_v1`**, distinct from the root runtime's
`portable_v1` and historical research checkpoints. Use this directory's
evaluation, analysis and benchmark programs with these checkpoints. Source
hashes are part of identity; moving files is permitted, changing executable
content is not compatible with automatic same-identity resume.

The generic evaluation and analysis tools are retained for a separately
declared future cohort. The current 16-run LR grid is not such a cohort: it
contains several LRs for each arm/seed. `analyze.py --study pilot` produces a
descriptive single-pair comparison without seed-level uncertainty intervals;
the name denotes the analysis mode, not an extra pilot experiment. `medium`
requires three paired seeds and `small` requires five. `--reference` and
`--candidate` choose the compared arms. The benchmark measures synchronized
wall-clock forward/optimizer-step durations on disposable checkpoint copies;
training elapsed time is not a substitute for that benchmark.

The preserved implementation includes all nine `code/*.py` files. Source
fingerprints are listed in the root
[source manifest](../../code/source_manifest.json). Publication packaging
does not rerun these experiments or change their numerical path.
