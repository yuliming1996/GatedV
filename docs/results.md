# Results and evidence

The completed paper experiments and the subsequent GLU/learning-rate study are separate evidence sets. The latter is an incomplete development-set snapshot, frozen here at 15 October 2026, 08:40 China Standard Time. It does not replace the paper's held-out results or establish the final learning-rate ranking.

## Completed paper experiments

Perplexity (PPL) is computed as `exp(NLL)` for each model. The means below are arithmetic means of per-seed PPL; they are not `exp(mean NLL)`. Lower values are better.

| Model size | Paired seeds | Baseline mean held-out PPL | GatedV mean held-out PPL | Relative reduction | Improving pairs |
|---|---:|---:|---:|---:|---:|
| 2.33M | 5 | 59.118187 | 57.915667 | 2.0341% | 5/5 |
| 26.16M | 3 | 21.383592 | 20.599922 | 3.6648% | 3/3 |

The small confirmation used a fixed peak learning rate of 0.0048. The medium confirmation followed a single-seed, equal-budget search over 0.0006, 0.0012, and 0.0024; both arms selected the lower boundary, 0.0006. These are finite-budget comparisons, not claims that either architecture was optimally tuned. The subsequent lower-learning-rate study addresses that limitation.

The paired NLL mean differences (GatedV minus baseline) were −0.0205913 and −0.0373168 for the small and medium confirmations. Their approximate 95% Student-t intervals across training seeds were [−0.0364062, −0.00477635] and [−0.0545543, −0.0200793], respectively. These estimates involve only five and three seed pairs. The separate document/domain bootstrap is conditional on those trained models; each accepted domain contributes one document, so the two partitions yield the same resampling result rather than independent evidence.

The short, interleaved DGX Spark benchmark measured mean-of-seed-median time ratios of **1.007003** for full-context forward passes and **1.011433** for training steps. These are synchronized wall-clock measurements, not GPU-throughput-normalized values. Ambient temperature, clock frequency, and power were not controlled; the approximately 0.7% forward difference should not be interpreted as a hardware-independent cost or an autoregressive decoding benchmark. All 42 timing slots and their raw samples are retained.

## Exploratory and negative results

The archive retains all 133 completed scientific trajectories, including earlier depth-reduction experiments omitted from the current paper text. The two selected four-layer nonlinear-projection candidates did not match the six-layer baseline: their held-out mean PPL increased by approximately 7.04% and 6.82%. Same-depth GatedV gains do not overturn that result.

In the reused-development-set branch comparison, Q-only SiLU increased mean PPL by approximately 1.61%; all-QKV SiLU improved it by only 0.21%, with mixed seed directions and a conditional interval crossing zero. No K-only arm was tested. The small-model, four-learning-rate exploration selected 0.0024 for baseline and GatedV and found a 3.08% development PPL reduction. Its affine and standardized-SiLU controls did not recover the full GatedV improvement at their tested rates. These controls reused three seeds and the development set, searched fewer learning rates than the core arms, and matched initial moments only; they do not isolate a unique causal mechanism.

The study identifiers in `results/studies/` are A (initial screen), B (depth confirmation), C (branch/LR exploration), D (small GatedV confirmation), E/F (mechanism and supplementary LR exploration), and M (medium selection and confirmation). The 133 trajectories are not 133 independent statistical replications.

## GLU and lower-learning-rate study: 5 of 16 endpoints

All rows below use seed 20270201, 327,680 optimizer updates, and 671,088,640 training-target exposures. They were evaluated on the same development set, not a new holdout. Each completed endpoint passed the original endpoint-audit kernel, including its learning-rate and exposure schedule, finite model/Adam state, and 17 full development evaluations. The baseline 3e−4 audit was finalized at 08:41:15, shortly after the 08:40 status observation.

| Model | Parameters | Peak LR | Dev NLL | Dev PPL |
|---|---:|---:|---:|---:|
| Baseline | 26,158,464 | 6e−4 | 3.070762576 | 21.558336268 |
| GatedV | 26,158,464 | 6e−4 | 3.044748173 | 21.004740973 |
| GLU, full value width | 27,927,936 | 6e−4 | 3.053871186 | 21.197244261 |
| GLU, reduced value width | 26,186,112 | 6e−4 | 3.069079256 | 21.522077224 |
| Baseline | 26,158,464 | 3e−4 | 3.023266655 | 20.558339116 |

At 6e−4, GatedV PPL is 2.5679% below baseline, 0.9082% below full-width GLU, and 2.4037% below reduced-width GLU. The reduced GLU uses value width 258 to retain six heads; its 27,648 additional parameters make it **near matched, not exactly parameter matched**. GLU uses an independently projected content branch and gate, whereas GatedV applies SiLU directly to the existing value projection.

Baseline improves by 4.6386% when its peak learning rate decreases from 6e−4 to 3e−4. This is evidence that the original lower search boundary matters. GatedV at 3e−4 was still training at this snapshot; the remaining 11 endpoints have no final result here. The planned grid is 6e−4, 3e−4, 1e−4, and 1e−5 for each of four arms. Comparing each method at its selected learning rate must wait for the complete grid. These single-seed development results cannot establish a multi-seed or held-out GLU/GatedV ranking.

Full-precision values, all 85 saved development summaries, source audit hashes, and result/checkpoint/array hashes are retained in [the snapshot JSON](../results/glu_lr_controls_20261015.json); [the CSV](../results/glu_lr_controls_20261015.csv) contains the five endpoint rows. Training duration in those files is an elapsed training cost, not an independent forward benchmark.

## What can be verified from this repository

From the repository root, with Python 3.10+ and NumPy installed:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B results/verify_results.py
```

This checks the packaged file hashes, all 133 original endpoint rows and 77 summary cells, held-out seed statistics, medium per-document loss arrays and conditional bootstrap, the 42 raw cost slots, 374 saved learning-curve points, and arithmetic/bindings in the five-endpoint GLU snapshot. It needs neither a GPU nor training data. `results/verification.json` records the packaging-time verification.

`source_ledger.json` distinguishes original source hashes from packaged extract hashes. The numerical results retain their original values. A few provenance-only path labels were converted to explicit historical identifiers; original source hashes are preserved alongside the updated packaged hashes. `packaging_provenance.json` records copy or adaptation status and identifies the extended verifier and new GLU extract. Historical identifiers are not runtime file dependencies. A source hash for an omitted file provides traceability, not independent verification of that file. Original checkpoints, corpus text, full execution logs, and the new GLU per-document arrays are not distributed here. Consequently, the portable check is numerical reanalysis and file-integrity verification, not a rerun or independent replication of the original training audits.

The released `learning_curves.csv` contains diagnostic/selection development evaluations. Its final points must not be substituted for the fresh-heldout table values. The medium held-out NPZ files contain numerical losses, counts, document indices, and offsets, with no text or token IDs.

Both completed confirmations use the same FineWeb-Edu family and first-window evaluation, with conservative domain grouping and approximate deduplication. These choices limit distributional coverage and do not guarantee absence of semantic overlap. Improved NLL/PPL alone does not demonstrate general capability gains or successful depth reduction.
