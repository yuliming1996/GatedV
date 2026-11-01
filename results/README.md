# Results used in the paper

The main held-out comparison (Table 3) uses peak learning rates 0.0048 for
the small model and 0.0003 for the medium model. PPL changes are ratios of
the arithmetic mean PPL across seeds, computed before rounding: −2.03% and
−2.40%, respectively. These are reductions of 2.03% and 2.40%.

| Paper result | Saved evidence | Check |
|---|---|---|
| Small held-out comparison, Table 3 | [Five paired seeds](studies/D/results.json) | `verify_results.py` |
| Medium held-out comparison at 0.0003, Table 3 | [Six evaluations](medium_lr3e4/) | `verify_medium_lr3e4.py` |
| Development curves, Figures 2–3 | [Per-seed checkpoints](learning_curves_lr3e4.csv) | `verify_paper_consistency.py` |
| Small activation and moment controls, Tables 4–5 | [Per-run endpoints](all_endpoints.csv) | `verify_results.py` |
| Medium timing and memory, Table 6 | [42 timing slots](medium_cost/) | `verify_results.py` |
| Medium learning-rate and value-gate grids, Tables 2 and 7 | [19 development endpoints](value_gate_grid/) | `verify_value_gate_grid.py` |
| Original medium confirmation at 0.0006, Appendix A | [Original results](studies/M/results.json) and [six evaluations](medium_holdout/) | `verify_results.py` |
| Train/inference activation comparison, Appendix B | [Twelve evaluations](cross_eval/) | `verify_cross_eval.py` |

Run checks from the repository root, for example
`python -B results/verify_paper_consistency.py`.
This also compares the displayed paper/README values and the confirmation
recipes. See [verification instructions](../verification/README.md).

For training, `tools/reproduce.py --study small` reproduces the small-model
recipe, `--study medium-lr3e4` the main medium-model recipe, and
`--study medium` the original 0.0006 recipe in Appendix A. Both medium studies
use the same held-out set.

## Historical records

`all_endpoints.csv`, `all_summary_cells.csv`, `budget.json`, and
`verification.json` retain the original A–F/M bundle of 133 trajectories.
Their medium confirmation uses 0.0006; their totals exclude the later value-gate
grid and 0.0003 confirmation. The dated `glu_lr_controls_20261015.*` files
record the first five completed grid runs. The complete grid is in
`value_gate_grid/`. Dated validation reports in `docs/` describe the snapshots
they checked; [paper_consistency.json](../docs/paper_consistency.json) records
the current paper/README check.
