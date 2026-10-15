# GatedV: Self-Gated Values for Transformer Language Models

GatedV applies SiLU to projected values before attention aggregation, without
adding trainable parameters:

$$
\widetilde V = V \odot \sigma(V).
$$

[Paper](paper/main.tex) · [Code](code/model.py) · [Usage](docs/usage.md)

![GatedV architecture](paper/figures/architecture_gatedv.png)

## Results

Mean held-out PPL under the paper's training settings:

| Parameters | Paired seeds | Baseline | GatedV | PPL reduction |
|---|---:|---:|---:|---:|
| 2.33M | 5 | 59.12 | 57.92 | 2.03% |
| 26.16M | 3 | 21.38 | 20.60 | 3.66% |

GatedV improved PPL in every seed pair. A short DGX Spark benchmark measured
about 0.7% higher full-context forward latency for the 26.16M model.

## Ongoing work

| Experiment | Status |
|---|---|
| GLU comparison and lower learning rates | In progress |
| Tanh self-gating ablation | Prepared; awaiting GLU completion |

By [Li-Ming Yu](https://orcid.org/0000-0001-7552-8895).
[Citation](CITATION.cff) · [MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
