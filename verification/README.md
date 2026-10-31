# Verification

Run from the repository root after [installing dependencies](../README.md#usage):

```sh
python -B tools/verify_release.py
python -B results/verify_results.py
python -B tools/reproduce.py --study smoke
```

The first two commands check packaged files and recompute saved statistics.
The last runs a tiny synthetic CPU test without downloading data. These checks
do not repeat the original training runs.

For the 0.0003 medium-model result in Table 3, run
`python -B results/verify_medium_lr3e4.py` with NumPy installed. It checks the
six saved array digests and window alignment, computes the mean PPL and sample
SD across seeds, and repeats the 2,000 document-bootstrap draws. It also
counts windows where GatedV has lower NLL after averaging across the three seed
pairs. For the six full training recipes, use
`python -B tools/reproduce.py --study medium-lr3e4 --dry-run`.

Two further checks use NumPy and the published loss arrays:

```sh
python -B results/verify_value_gate_grid.py
python -B results/verify_cross_eval.py
```

The first checks all 19 single-seed development endpoints and prints the three
shared learning rates in Table 7. The second recomputes the four cells in
Appendix B and checks that each checkpoint's native inference mode matches
its original per-window losses exactly. [Checkpoint evaluation instructions](../results/cross_eval/README.md)

To recheck prepared data:

```sh
python -B tools/prepare_data.py --study small --output data/paper_small --verify-only
```

Replace both occurrences of `small` with `medium` for the larger study.
