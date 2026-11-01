# Headwise G2 comparison

These three runs add headwise G2 to the single-seed development comparison in
Table 7. Each uses seed 20270201, 327,680 updates, batch size 8 and context
length 256: 671,088,640 prediction targets per run. The training windows and
2,048 development windows match the [original value-gate grid](../value_gate_grid/).
No held-out evaluation was performed for G2.

| Peak LR | Baseline PPL | GatedV PPL | Headwise G2 PPL |
|---|---:|---:|---:|
| 0.0006 | 21.558 | 21.005 | 21.568 |
| 0.0003 | 20.558 | 20.062 | 20.630 |
| 0.0001 | 20.938 | 20.632 | 21.189 |

G2 has its lowest development PPL at 0.0003. Its PPL is above baseline at all
three rates. At 0.0003, GatedV is 2.76% below G2. These are single-seed results.

The [manifest](manifest.json) binds the original run identities, results,
per-window arrays, CUDA preflight and training audit by SHA-256. The
[independent verification](independent_verification.json) records checks of
983,040 progress entries, 51 development arrays, final checkpoints, optimizer
and RNG states, and process/lock exit. Full checkpoints and progress logs
remain in the research archive; the files here support endpoint recomputation.

```sh
PYTHONDONTWRITEBYTECODE=1 python -B results/verify_g2_headwise.py
```

The [training runtime and recipes](../../experiments/g2_headwise/) are exact
copies of the frozen files used for these runs.
