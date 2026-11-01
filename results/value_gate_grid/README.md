# Medium-model value-gate comparison

[Endpoint results](endpoints.csv) lists final development NLL and PPL for 19 runs.
Each row links a training recipe to a per-window loss array through its run ID
and SHA-256 digest. All variants used seed 20270201, 327,680 training updates
and the same 2,048 development windows.

The baseline, GatedV and two GLU variants each completed four peak learning rates
(0.0006, 0.0003, 0.0001 and 0.00001). The tanh control completed the first three.
Its 0.00001 run was stopped before the full budget. Table 7 compares these five
variants and [headwise G2](../g2_headwise/) at the three shared rates.

Run `python -B results/verify_value_gate_grid.py` from the repository root to
recompute all 19 endpoints and print the table. The archived GLU and combined
three-rate comparison reports have SHA-256 digests
`a846defd3113cf3c4792d736905c016312db6bbedff5eb1b474ab0f9b081194f` and
`ae9162061dbbe2924762098100639f5576d5e1a40cba00b24f1546c46e399e55`,
respectively. The five-run snapshot dated 2026-10-15 records an earlier stage
of the GLU grid.

[GLU implementation](../../experiments/glu_attention/code/model.py) ·
[Tanh implementation](../../experiments/tanh_self_gate/code/model.py) ·
[Tanh recipes](../../experiments/tanh_self_gate/configs)

To train the tanh control at 0.0003 using the prepared medium corpus:

```sh
python -B experiments/tanh_self_gate/code/train.py \
  --config experiments/tanh_self_gate/configs/medium_v_tanh_gate_lr3e-4.json \
  --data data/paper_medium/medium_corpus --output outputs/tanh_lr3e4
```
