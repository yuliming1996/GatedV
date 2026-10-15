# Manuscript figures

The current manuscript is `../main.tex`. Its figures are generated externally by `../make_paper_figures.py` and inserted with `\includegraphics`; no TikZ or PGFPlots code is embedded in the manuscript.

| Figure | PDF / PNG | Content |
|---|---|---|
| 1 | [architecture_gatedv.pdf](architecture_gatedv.pdf) / [PNG](architecture_gatedv.png) | Transformer block and the alternative baseline / GatedV value transforms |
| 2 | [development_delta_ppl.pdf](development_delta_ppl.pdf) / [PNG](development_delta_ppl.png) | Paired development PPL difference versus the fraction of scheduled optimizer updates, small and medium panels |
| 3 | [development_ppl.pdf](development_ppl.pdf) / [PNG](development_ppl.png) | Baseline and GatedV development PPL versus optimizer updates, small and medium panels |

## Data and definitions

The source is [learning_curves.csv](../../results/learning_curves.csv), bound by [learning_curves_provenance.json](../../results/learning_curves_provenance.json). Only `small_confirmation` and `medium_confirmation` enter these figures: five and three paired seeds, respectively. The six medium selection trajectories are not pooled with confirmation runs.

PPL is `exp(dev_nll)` for each saved run/checkpoint. Means are arithmetic means of per-seed PPL, not exponentials of mean NLL. Paired differences are `PPL_GatedV - PPL_baseline` in absolute PPL units; negative values favor GatedV. Percentage differences in the manuscript tables are a separate quantity. These are development trajectories, not held-out trajectories or training-minibatch losses.

Thin lines represent individual seeds or pairs. Thick black lines represent their arithmetic means. PPL plots use blue for baseline seeds and orange for GatedV seeds; baseline means have solid lines and filled triangles, and GatedV means have dashed lines and filled circles. No per-seed legend or inferential bands are used.

The main PPL panels display the 16 post-initialization observations on linear axes, while their insets preserve all 17 observations on logarithmic PPL axes. The paired-difference panels exclude initialization from the display because its much larger magnitude obscures training differences. Numerical plot exports retain initialization. There is no smoothing, additional evaluation, or combination with held-out endpoints.

## Regeneration

From the package root, using Python with NumPy and Matplotlib:

```sh
python paper/make_paper_figures.py
```

The script writes vector PDF figures, PNG previews, numerical plot data, and provenance information. Recompile `paper/main.tex` with the generated figures and external `paper/references.bib` available. The script does not modify the manuscript or experimental records.

Only figures used by the current manuscript are included in this release. Earlier `.tex` fragments, preview alternatives, and superseded generators have been omitted. The manuscript and its committed figures are preserved unchanged; see the [paper README](../README.md) for the packaging and regeneration checks.
