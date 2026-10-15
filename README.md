# GatedV: Self-Gated Values for Transformer Language Models

GatedV applies SiLU to projected values before attention aggregation, preserving
the standard Transformer's projections, value width, and parameter count:

$$
\widetilde V = V \odot \sigma(V), \qquad
Y = \mathrm{softmax}(QK^\top/\sqrt{d_h}+M)\widetilde V W_O.
$$

[Paper](paper/main.tex) · [Model code](code/model.py) · [Detailed usage](docs/usage.md)

![GatedV architecture](paper/figures/architecture_gatedv.png)

## Results

Held-out perplexity, reported as the arithmetic mean ± sample standard deviation
across paired seeds. Each pair uses matched parameters and training-token budgets.

| Parameters | Seed pairs | Baseline PPL | GatedV PPL | Reduction |
|---|---:|---:|---:|---:|
| 2.33M | 5 | 59.1182 ± 0.5298 | 57.9157 ± 0.7816 | 2.03% |
| 26.16M | 3 | 21.3836 ± 0.1658 | 20.5999 ± 0.0184 | 3.66% |

All pairs improve. The medium study searched `{6e-4, 1.2e-3, 2.4e-3}`; both arms
selected the lower boundary. A short DGX Spark benchmark measured about 0.7%
higher full-context forward latency, without controlling temperature or clocks.
The [paper](paper/main.tex) discusses prior work, ablations, and these limitations.

**GLU / lower-LR follow-up:** the [saved snapshot](results/glu_lr_controls_20261015.csv)
contains 5 of 16 endpoints as of 15 October 2026, 08:40 Beijing time. These are
single-seed development results, not a completed LR selection or held-out
comparison. This follow-up is not yet incorporated into the manuscript.

## Quick start

Python 3.10+ on Linux or macOS:

```sh
git clone https://github.com/yuliming1996/GatedV.git
cd GatedV
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -B tools/reproduce.py --study smoke
```

This runs a tiny synthetic CPU check. To reproduce either full confirmation
study, use a CUDA-compatible PyTorch installation and run one of:

```sh
python -B tools/reproduce.py --study small --download
python -B tools/reproduce.py --study medium --download
```

Each command reconstructs the original data, trains all paired seeds, evaluates
the completed cohort, and computes statistics. These are full training jobs:
10 runs for small and 6 for medium. They use the reported confirmation learning
rates rather than repeating the LR search. Raw downloads require `curl` and about
2.69 GB (small) or 6.46 GB (medium), plus space for prepared data and checkpoints.
Hardware and software differences can affect training results.

Add `--dry-run` to inspect a plan or `--resume` to continue an interrupted run.
Outputs go to `outputs/reproduction_small/` or `outputs/reproduction_medium/`.
See [detailed usage](docs/usage.md) for custom paths, data preparation and GLU runs.

## Results, figures and paper

Verify the saved numbers without training:

```sh
python -B tools/verify_release.py
python -B results/verify_results.py
```

Compile the paper using the included figures (requires a LaTeX installation):

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Optionally regenerate figures from the repository root:

```sh
python -m pip install -r requirements-figures.txt
python -B paper/make_paper_figures.py
```

[Configurations](configs/), [all historical endpoints](results/all_endpoints.csv),
and [learning curves](results/learning_curves.csv) are included. The numerical
records retain exploratory and negative results. Raw corpora and checkpoints
are not bundled; the original tokenizer and accepted-document selections are.

## Citation and license

By **Li-Ming Yu** ([ORCID](https://orcid.org/0000-0001-7552-8895)).
See [CITATION.cff](CITATION.cff) for citation metadata.
Code is licensed under [MIT](LICENSE); dataset and dependency terms are listed
in [third-party notices](THIRD_PARTY_NOTICES.md).
