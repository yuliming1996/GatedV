# Headwise G2 training recipe

Each attention layer projects its RMS-normalized input from width 384 to six
scalar gates, applies sigmoid, and multiplies each value head by its gate
before attention aggregation. The projection has no bias and adds 27,648
parameters across 12 layers, for a total of 26,186,112. This implements the
headwise G2 value gate from Gated Attention (Table 1, row 11 of the cited paper).

The five files in `code/` and three recipes in `configs/` retain the exact
training-source hashes recorded in [the protocol](../../results/g2_headwise/protocol.json).
Shared backbone matrices use the same name-based initialization as the
baseline/GatedV value-gate grid. The runtime includes the original resource
guard. Its GLU-related comments and metadata reflect the runtime it extends.

After preparing the medium corpus with `tools/prepare_data.py`, a single run is:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B experiments/g2_headwise/code/train.py \
  --config experiments/g2_headwise/configs/g2_lr3e-4.json \
  --data data/paper_medium/medium_corpus --output outputs/g2_lr3e4
```

The other two recipes use peak learning rates 0.0006 and 0.0001. Each runs
327,680 updates in FP32 with math SDPA and TF32 disabled. Use a separate output
directory for each rate. See [saved results and checks](../../results/g2_headwise/).
