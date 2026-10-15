# Manuscript

[`main.tex`](main.tex) is the current manuscript, **GatedV: Self-Gated Values for Transformer Language Models**, by Li-Ming Yu. It is based on the manuscript last edited on 2026-10-13, with the public [repository URL](https://github.com/yuliming1996/GatedV) added to the introduction on 2026-10-15. The scientific text, author details, tables, bibliography, and three included figures are otherwise unchanged. This directory contains the files needed to build that version, rather than earlier drafts or compiled manuscript PDFs.

The manuscript reports the completed small- and medium-model studies. It does **not** incorporate the ongoing four-arm GLU/low-learning-rate comparison. In particular, its medium learning-rate grid and discussion of untested GLU baselines describe the earlier study; those passages need author review before submitting a revised manuscript.

## Compile

With a LaTeX distribution providing pdfLaTeX, BibTeX, and latexmk:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The bibliography is loaded from [`references.bib`](references.bib). The committed figure PDFs are sufficient for compilation; Python is needed only to regenerate them. A clean temporary build with TeX Live 2025 and latexmk 4.86a produced a 13-page PDF with no missing citations or cross-references. The repository-link addition was compiled again on 2026-10-15 with the same outcome. One underfull bibliography line remains; no manuscript text was changed to adjust its formatting.

## Regenerate figures

From the repository root, using Python with NumPy and Matplotlib:

```sh
python -m pip install -r paper/requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -B paper/make_paper_figures.py
```

The generator reads [`../results/learning_curves.csv`](../results/learning_curves.csv), verifies its SHA-256, and writes the three PDF/PNG figures and their numerical exports. It does not train models or open the underlying evaluation datasets. See [`figures/README.md`](figures/README.md) for definitions of PPL, paired differences, and display choices.

The committed figures and [`paper_figure_manifest.json`](figures/paper_figure_manifest.json) are preserved together. They were originally generated with Matplotlib 3.11.1 and NumPy 2.5.2. A temporary regeneration using Matplotlib 3.11.2 and NumPy 2.5.3 reproduced all three numerical exports byte-for-byte and all three PNG images pixel-for-pixel. Image file hashes differed because the graphics-library metadata changed; the committed artwork was not replaced. Other versions or fonts may alter rendering without changing the values. The generated manifest records the actual versions and hashes used on each run.

Paths to original experiment summaries, arrays, and audits in the CSV exports are provenance identifiers. Those archive files are not required to draw these figures and are not revalidated by the plotting script.

## Included files

- `main.tex` and `references.bib`: current paper source and external bibliography.
- `make_paper_figures.py`: the Python generator actually used by this manuscript.
- `figures/architecture_gatedv.{pdf,png}`: model architecture.
- `figures/development_delta_ppl.{pdf,png}`: paired development-PPL differences.
- `figures/development_ppl.{pdf,png}`: development-PPL learning curves.
- `figures/paper_ppl_{points,summary}.csv`, `paper_ppl_data.json`, and `paper_figure_manifest.json`: numerical exports and figure provenance.

Older manuscript copies, unused figures, preview alternatives, PGFPlots/TikZ generators, and LaTeX build intermediates are omitted.
