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

To recheck prepared data:

```sh
python -B tools/prepare_data.py --study small --output data/paper_small --verify-only
```

Replace both occurrences of `small` with `medium` for the larger study.
The saved GLU snapshot contains partial, single-seed development results;
it is separate from the completed held-out comparisons in the README.
