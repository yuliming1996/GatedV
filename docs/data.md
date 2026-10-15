# Data requirements

The original trained tokenizer is bundled in [`../tokenizer/`](../tokenizer/README.md), with its verified historical SHA-256 and loading instructions.
This repository does not distribute raw documents, token arrays, model checkpoints or historical calibration coefficients. It
contains model/training code, recipes, hash descriptors and result summaries.
The [reconstruction guide](../reproduction/README.md) now provides a download/preparation script for the main small and medium confirmation datasets, using the original accepted-row selections and tokenizer. It checks reconstructed payloads against historical hashes. Substituting new text or training a new tokenizer constitutes a new experiment, even if array shapes match.

For the complete data-to-statistics workflow, use
`python -B tools/reproduce.py --study small --download` (or `--study medium`)
after installing `requirements-reproduce.txt`. No historical workspace folder is
required. The commands below document the arrays consumed by the underlying
runtime; ordinary reproduction does not require preparing them manually.

## Training and development arrays

Both trainers accept a directory containing:

| File | Required representation |
|---|---|
| `tokens_train.npy`, `tokens_dev.npy` | One-dimensional integer token IDs; original storage is `uint16` |
| `docs_train.npy`, `docs_dev.npy` | Integer arrays of shape `[number_of_documents, 2]`; original storage is `int64` |
| `tokenizer.json` | Copy from `tokenizer/tokenizer.json`; if supplied, bound into the run's hash identity |
| `manifest.json` | Corpus metadata, if supplied; also bound into the run's hash identity |

Each document row is a half-open `[start, end)` span in its corresponding
token array. Spans must be nonempty and contained in that array. Token IDs
must be in `[0, vocab)`. The reported recipes use `vocab=8192`; model/tokenizer
vocabulary and special-token conventions must agree. NumPy arrays are loaded
with `allow_pickle=False`.

Documents longer than the context length are eligible for training. A sample
uses `context+1` contiguous tokens; inputs and targets are offset by one token.
The sampler chooses from all valid document-contained windows with a fixed
PCG64 schedule, seeded by `202609210000 + training_seed`. No window crosses a
document boundary. Development evaluation uses a fixed traversal and the
declared target budget. Input hashes are checked before training and again
before a completed result is committed.

The medium study and GLU follow-up use the prepared 512,000,068-token training
corpus with 212,052 documents. The medium development set contains 2,048
documents and 524,288 evaluated targets. The GLU
[scientific plan](../experiments/glu_attention/configs/scientific_plan.json)
records the original train/dev/tokenizer/manifest SHA-256 values without
machine-specific locations. Matching these hashes verifies bytes, not the
validity or independence of a newly prepared corpus.

## Holdout payloads

The original small and medium holdouts are separate datasets. Each contains
2,048 documents and evaluates the first 256 targets per document, for 524,288
targets. Neither is part of the current GLU LR-selection experiment.

For a separately declared completed cohort, the evaluator expects:

| File | Format |
|---|---|
| `tokens_holdout.npy` | One-dimensional `uint16` token IDs |
| `docs_holdout.npy` | `int64` document boundaries, shape `[documents, 2]` |
| `eval_offsets.npy` | `int64`, exactly the first token offset of every document |
| `eval_document_ids.npy` | `int64`, exactly `0, ..., documents-1` |
| `documents_holdout.jsonl` | Metadata; `doc_id` and `domain_group` are used for domain-cluster analysis |

Documents must be contiguous in the token array, longer than the context,
and terminated by the declared EOS token (historically ID 3). The seal JSON
contains `context`, `documents`, `eval_tokens`, optional `eos`, and a
`files_sha256` mapping. Every file in that mapping must be provided and match.
The included `configs/small_holdout.json` and `configs/medium_holdout.json`
are descriptors for the original payloads. They cannot validate unrelated
data. The evaluator validates every declared training endpoint before opening
the holdout, and rechecks holdout hashes before committing completion.

## Reproducibility limits

Original corpus preparation relied on a fixed tokenizer, source-document
inventory, historical exclusion registries and prior accepted corpora. Those
historical exclusion registries are not reconstructed by the training runtime or the new preparation tool; the tool replays the frozen accepted rows instead. The selection and
filtering process used domain, identity, text, approximate duplicate and long
span checks; approximate filtering does not guarantee semantic non-overlap.
The shared corpus family, first-window evaluation and coarse domain grouping
also limit generalization. One document per domain means document/domain
bootstrap partitions contain the same information.

The release supports exact payload reconstruction for the main small/medium confirmations, new experiments on compatible data, and inspection of the reported methodology. Accepted-row replay does not replace the independent historical data audit, reproduce all exploratory holdouts, or regenerate missing calibration coefficients. Upstream data terms continue to apply.
