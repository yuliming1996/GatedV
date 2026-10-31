# Linear and SiLU values at inference

Appendix B evaluates six final medium-model checkpoints (three paired training
seeds, peak learning rate 0.0003). Each checkpoint uses the same weights for
both inference modes on the original 2,048 held-out windows. Values below are
the mean and sample standard deviation of per-seed perplexity.

| Training | Linear at inference | SiLU at inference |
|---|---:|---:|
| Linear (baseline) | 20.233 ± 0.068 | 102.565 ± 31.053 |
| SiLU (GatedV) | 4,300.232 ± 899.037 | 19.748 ± 0.047 |

Recompute all four cells with NumPy from the repository root:

```sh
python -B results/verify_cross_eval.py
```

Each NPZ contains loss sums, counts, document IDs and offsets. The
[protocol](protocol.json) specifies the inference intervention. The
[evaluation record](complete.json) binds each array to its checkpoint and
parameter digests. [Per-seed results and paired contrasts](summary.json) are
available at full precision.

To repeat the forward passes after completing the
[`medium-lr3e4` reproduction](../../README.md#usage), use its six final
checkpoints and prepared holdout:

```sh
python -B experiments/cross_eval/run.py --output outputs/cross_eval
```

The evaluator works with checkpoints produced by the repository's training
recipe. It checks all six native inference results against the evaluation
saved alongside those checkpoints. The archived matrix above can be verified
without GPU access or model weights.
