# Reconstruct the paper's small and medium datasets

The original tokenizer and frozen accepted-row selections are now included. The preparation tool downloads pinned public FineWeb-Edu shards, verifies each shard's size and SHA-256, takes the exact selected rows in their original order, encodes the original text without normalization/BOS, appends one EOS (ID 3), and checks the historical hashes of all token/boundary arrays and holdout metadata. No raw document text is stored in the selection files.

This is **reconstruction of the original accepted datasets**, not a fresh execution or independent audit of the historical exclusion/deduplication decisions. Historical exclusion registries remain outside this release. These selections cover the main small and medium confirmations, and their reused training/development data, not every earlier study's separate holdout or the missing mechanism-calibration payloads.

## Complete reproduction

The repository-level entrypoint includes preparation, all paired training runs,
holdout evaluation and statistics:

```sh
python -m pip install -r requirements-reproduce.txt
python -B tools/reproduce.py --study small --download
# Or: python -B tools/reproduce.py --study medium --download
```

Use `--study smoke` first for a quick synthetic CPU check, or `--dry-run` to
inspect a formal plan without executing it. See [usage](../docs/usage.md) for
resume behavior and output locations. The following interface prepares only the
data and is also usable independently.

## Prepare

Run from the repository root, in an isolated Python environment:

```sh
python -m pip install -r reproduction/requirements.txt
python -B tools/prepare_data.py --study small --download --output data/paper_small
```

For the medium study:

```sh
python -B tools/prepare_data.py --study medium --download --output data/paper_medium
```

Use `--study all --output data/paper_all` to reconstruct both in one root. `curl` must be installed for downloading. Omit `--download` to use files already in `data/raw/`; use `--raw /path/to/cache` to select another cache. File names must be `013_00000.parquet`, etc. Existing cached shards are verified, never silently replaced. Downloads use HTTPS with certificate verification enabled. Interrupted transfers retain a `.download` cache and resume on retry; each completed shard must still match its pinned size and SHA. Nonblocking file locks prevent two preparations from writing the same output or download simultaneously. Network access, upstream availability and disk space are required.

Small requires shards 013 and 003 (~2.69 GB downloaded, ~0.10 GB prepared); medium requires 001, 002 and 004 (~6.46 GB downloaded, ~1.06 GB prepared). Both together require ~9.15 GB of cached shards plus ~1.16 GB of prepared files; allow additional space for dependencies and training checkpoints. The medium corpus contains 512M tokens, so preparation is a substantial CPU job. The default tokenizer worker count is bounded to two.

The output directory must be new. Work is staged in `OUTPUT.partial`; failures preserve this directory for inspection and no completed output is committed. After diagnosis, choose a new output path or explicitly remove only the failed partial directory. Preparation currently restarts rather than resuming partial encoding.

For the small command, successful outputs are:

```text
data/paper_small/
├── small_corpus/
│   ├── tokens_train.npy       # uint16, 40,001,310 tokens
│   ├── docs_train.npy         # int64, 27,830 [start,end) rows
│   ├── tokens_dev.npy         # uint16, 501,144 tokens
│   ├── docs_dev.npy           # int64, 352 rows
│   └── tokenizer.json
├── small_holdout/
│   ├── tokens_holdout.npy
│   ├── docs_holdout.npy
│   ├── documents_holdout.jsonl
│   ├── eval_offsets.npy
│   └── eval_document_ids.npy
└── reconstruction.json
```

The medium command creates analogous `medium_corpus/` and `medium_holdout/` directories. Its training arrays contain 512,000,068 tokens / 212,052 documents; its dev contains 2,861,010 stored tokens / 2,048 documents. Stored token counts differ from the 524,288 targets used per medium evaluation.

Recheck prepared payloads without downloading or encoding:

```sh
python -B tools/prepare_data.py --study small --output data/paper_small --verify-only
```

## Train and evaluate

Install the root training dependencies with a suitable CUDA-enabled PyTorch build. Train the complete declared cohort from scratch. The first small pair is:

```sh
python -B code/train.py --config configs/small.json --data data/paper_small/small_corpus --output runs/small/baseline_s20261201
python -B code/train.py --config configs/small.json --arm v_silu --data data/paper_small/small_corpus --output runs/small/v_silu_s20261201
```

Repeat both arms with `--seed 20261202`, `20261203`, `20261204`, `20261205`, matching the output names in `configs/small_cohort.json`. Only after all ten runs finish:

```sh
python -B code/evaluate.py --cohort configs/small_cohort.json --holdout data/paper_small/small_holdout --seal configs/small_holdout.json --output outputs/small_holdout
python -B code/analyze.py --evaluation outputs/small_holdout --study small --domains data/paper_small/small_holdout/documents_holdout.jsonl --output outputs/small_statistics.json
```

For medium, use `configs/medium.json`, seeds `20270211`, `20270212`, `20270213`, paths `runs/medium/ARM_sSEED`, and `data/paper_medium/medium_corpus`. After all six runs:

```sh
python -B code/evaluate.py --cohort configs/medium_cohort.json --holdout data/paper_medium/medium_holdout --seal configs/medium_holdout.json --output outputs/medium_holdout
python -B code/analyze.py --evaluation outputs/medium_holdout --study medium --domains data/paper_medium/medium_holdout/documents_holdout.jsonl --output outputs/medium_statistics.json
```

These commands repeat the confirmation recipes at their selected learning rates. They do not rerun the earlier LR searches, exploratory branches or GLU grid. The medium corpus can also be passed to the separate GLU runtime as described in its README.

Exact data hashes do not guarantee bitwise-identical training outcomes across software/hardware. The portable runtime has a distinct identity from the historical runs; this preparation tool deliberately does not recreate old machine-specific `manifest.json` files. Both arms of a new cohort must use the same prepared directory. CPU is available for small technical checks; the supplied full training budgets target CUDA and are expensive. No training is launched by preparation.

## Sources, attribution and selection provenance

Source: Hugging Face FineData, **FineWeb-Edu**, fixed revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, `sample/10BT`. [Dataset card and source terms](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md). Preserve upstream attribution/terms; the repository MIT code license does not replace them.

`data_plan.json` contains direct HTTPS download URLs, source byte counts and SHA-256 values, selection-file hashes, historical ledger hashes and final payload hashes. Source file sizes and LFS SHA-256 values were checked against the official fixed-revision Hugging Face tree on 15 October 2026. All five local source shards matched those hashes.

| Shard | Used for | Direct download |
|---|---|---|
| 013 | Small train/dev | [013_00000.parquet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/013_00000.parquet) |
| 003 | Small confirmation holdout | [003_00000.parquet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/003_00000.parquet) |
| 001, 002 | Medium train | [001_00000.parquet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/001_00000.parquet), [002_00000.parquet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/002_00000.parquet) |
| 004 | Medium fresh dev/holdout | [004_00000.parquet](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/004_00000.parquet) |

The accepted-row lists were exported from the original document ledgers, preserving order and per-document token lengths. The small train/dev manifest and medium reused-train manifest matched the hashes retained in `results/datasets.json`. The medium dev ledger was retrieved read-only from the original experiment host and matched its historical manifest. Holdout templates retain original non-text metadata and a null text placeholder; reconstruction fills that placeholder from the pinned shard, reproducing the original JSONL bytes. Final holdout hashes bind to the unchanged `configs/*_holdout.json` descriptors.

The selection files reveal which published documents belong to the historical holdouts. These sets can reproduce the published evaluation, but should not be described as newly blind tests for future model selection.

## Validation performed

On 15 October 2026, the tool actually reconstructed both studies from the verified local copies of all five public shards. All 18 historical payload hashes matched, including both holdout JSONL files. Five input/download-integrity tests passed. The official source tree was fetched over HTTPS; multi-GB downloads were not repeated. Environment versions and each checked hash are recorded in [`validation.json`](validation.json). This check did not train models.
