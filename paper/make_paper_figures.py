"""Generate the approved confirmation figures from retained evaluations.

Figure 1 is preserved. Figures 2 and 3 are standalone PGFPlots sources,
vector PDFs, and PNG previews. No manuscript or experimental records are edited.
Python uses only the standard library; rendering requires TeX/PGFPlots and Poppler.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdflatex", default="pdflatex")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--tex-only", action="store_true", help="Export figure sources and data only")
    args = parser.parse_args()

    PAPER = Path(__file__).resolve().parent
    SOURCE = PAPER.parent / "results/learning_curves_lr3e4.csv"
    SOURCE_SHA = "e7a887e80e91d15846a6a3045b4bae94b722dbe2cefa3975ea7ae949e492200b"
    FIGURES = PAPER / "figures"
    FIGURES.mkdir(exist_ok=True)


    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()


    assert sha(SOURCE) == SOURCE_SHA
    rows = list(csv.DictReader(SOURCE.open(newline="")))
    designs = {
        "small_confirmation": ("Small (2.33M), five paired seeds", list(range(20261201, 20261206)), 32768, 0.0048),
        "medium_confirmation": ("Medium (26.16M), three paired seeds", list(range(20270211, 20270214)), 327680, 0.0003),
    }
    data = {}
    for phase, (title, seeds, total, lr) in designs.items():
        selected = [r for r in rows if r["phase"] == phase]
        assert len(selected) == len(seeds) * 2 * 17
        assert sorted({int(r["seed"]) for r in selected}) == seeds
        assert {float(r["lr"]) for r in selected} == {lr}
        assert {int(r["total_steps"]) for r in selected} == {total}
        assert {r["arm"] for r in selected} == {"baseline", "v_silu"}
        steps = list(range(0, total + 1, total // 16))
        runs = {}
        for arm in ("baseline", "v_silu"):
            runs[arm] = {}
            for seed in seeds:
                rr = sorted(
                    (r for r in selected if r["arm"] == arm and int(r["seed"]) == seed),
                    key=lambda r: int(r["step"]),
                )
                assert [int(r["step"]) for r in rr] == steps
                values = [math.exp(float(r["dev_nll"])) for r in rr]
                assert all(math.isclose(v, float(r["ppl"]), rel_tol=1e-12) for v, r in zip(values, rr))
                assert all(int(r["training_targets"]) == int(r["step"]) * 2048 for r in rr)
                runs[arm][str(seed)] = values
        means = {
            arm: [statistics.mean(runs[arm][str(seed)][i] for seed in seeds) for i in range(17)]
            for arm in runs
        }
        paired = {
            str(seed): [
                100 * (g / b - 1)
                for g, b in zip(runs["v_silu"][str(seed)], runs["baseline"][str(seed)])
            ]
            for seed in seeds
        }
        relative_mean = [100 * (g / b - 1) for g, b in zip(means["v_silu"], means["baseline"])]
        assert all(-5.2 < x < 5.2 for values in paired.values() for x in values)
        data[phase] = {
            "title": title, "seeds": seeds, "steps": steps, "total_steps": total, "peak_lr": lr,
            "per_seed_ppl": runs, "mean_ppl": means, "paired_relative_ppl_percent": paired,
            "relative_mean_ppl_percent": relative_mean,
        }


    def coords(x, y):
        return " ".join(f"({a:.12g},{b:.12g})" for a, b in zip(x, y))


    tex = [
        r"\documentclass[border=3pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern,amsmath,pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{MeanBlue}{HTML}{245D80}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"group style={group size=2 by 1,horizontal sep=9mm},",
        r"scale only axis,width=65mm,height=48mm,",
        r"axis lines*=left,axis line style={black!55,line width=.45pt},",
        r"tick style={black!55,line width=.45pt},tick align=outside,",
        r"ticklabel style={font=\fontsize{8}{10}\selectfont},",
        r"label style={font=\fontsize{9}{11}\selectfont},",
        r"title style={font=\fontsize{9}{11}\selectfont,anchor=south west,at={(0,1.025)}},",
        r"ymin=-5.2,ymax=5.2,ytick={-4,-2,0,2,4},",
        r"yticklabels={$-4$,$-2$,$0$,$+2$,$+4$},",
        r"ymajorgrids=true,grid style={black!9,line width=.35pt},",
        r"scaled ticks=false,xlabel={Optimizer updates ($10^3$)},",
        r"legend style={draw=none,fill=none,font=\fontsize{8}{10}\selectfont,",
        r"at={(1.075,1.23)},anchor=south,legend columns=2,column sep=4mm},",
        r"legend cell align=left,clip=true]",
    ]
    for panel, d in enumerate(data.values()):
        x = [s / 1000 for s in d["steps"]]
        total = d["total_steps"] / 1000
        tick = [0, 8, 16, 24, total] if panel == 0 else [0, 80, 160, 240, total]
        tick_labels = ["0", "8", "16", "24", "32.8"] if panel == 0 else ["0", "80", "160", "240", "327.7"]
        options = [
            "title={" + f"({chr(97+panel)}) {d['title']}" + "}", f"xmin={-total*.015:.6f}",
            f"xmax={total*1.035:.6f}", "xtick={" + ",".join(map(str, tick)) + "}",
            "xticklabels={" + ",".join(tick_labels) + "}",
        ]
        if panel == 0:
            options.append(r"ylabel={Relative development PPL (\%)}")
        else:
            options.append("yticklabels={}")
        tex.append(r"\nextgroupplot[" + ",".join(options) + "]")
        tex.append(r"\addplot[black!45,densely dashed,line width=.5pt,forget plot] coordinates {"
                   + f"({-total*.015},0) ({total*1.035},0)" + "};")
        for i, values in enumerate(d["paired_relative_ppl_percent"].values()):
            forget = "" if panel == 0 and i == 0 else ",forget plot"
            tex.append(r"\addplot[black!34,line width=.55pt,no marks" + forget + "] coordinates {"
                       + coords(x, values) + "};")
            if panel == 0 and i == 0:
                tex.append(r"\addlegendentry{Individual seed pairs}")
        forget = "" if panel == 0 else ",forget plot"
        tex.append(r"\addplot[MeanBlue,line width=1.5pt,mark=*,mark size=1.3pt,"
                   r"mark options={solid,fill=MeanBlue,draw=MeanBlue}" + forget + "] coordinates {"
                   + coords(x, d["relative_mean_ppl_percent"]) + "};")
        if panel == 0:
            tex.append(r"\addlegendentry{Relative change in mean PPL}")
        end = d["relative_mean_ppl_percent"][-1]
        tex.append(
            r"\node[anchor=north east,text=MeanBlue,font=\fontsize{8}{10}\selectfont,"
            r"align=right] at (rel axis cs: .97,.96) {Final: $" + f"{end:.2f}" + r"\%$};"
        )
    tex += [r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}"]
    (FIGURES / "development_relative_ppl.tex").write_text("\n".join(tex) + "\n")

    # A complementary absolute-PPL view of the final quarter, using the five
    # existing observations at 75, 81.25, 87.5, 93.75, and 100 percent progress.
    late_tex = [
        r"\documentclass[border=3pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern,amsmath,pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{GatedBlue}{HTML}{245D80}",
        r"\definecolor{BaseCopper}{HTML}{BE7947}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"group style={group size=2 by 1,horizontal sep=13mm},",
        r"scale only axis,width=63mm,height=45mm,",
        r"axis lines*=left,axis line style={black!55,line width=.45pt},",
        r"tick style={black!55,line width=.45pt},tick align=outside,",
        r"ticklabel style={font=\fontsize{8}{10}\selectfont},",
        r"label style={font=\fontsize{9}{11}\selectfont},",
        r"title style={font=\fontsize{9}{11}\selectfont,anchor=south west,at={(0,1.025)}},",
        r"xmin=74.5,xmax=100.8,xtick={75,80,85,90,95,100},",
        r"ymajorgrids=true,grid style={black!9,line width=.35pt},",
        r"scaled ticks=false,xlabel={Training progress (\%)},",
        r"legend style={draw=none,fill=none,font=\fontsize{8}{10}\selectfont,",
        r"at={(1.1,1.23)},anchor=south,legend columns=2,column sep=6mm},",
        r"legend cell align=left,clip=true]",
    ]
    late_ranges = [(55.8, 68.6, "56,58,60,62,64,66,68"), (19.8, 23.2, "20,21,22,23")]
    for panel, d in enumerate(data.values()):
        indices = [i for i, step in enumerate(d["steps"]) if step >= 0.75 * d["total_steps"]]
        assert indices == list(range(12, 17))
        x = [100 * d["steps"][i] / d["total_steps"] for i in indices]
        assert x == [75, 81.25, 87.5, 93.75, 100]
        low, high, ticks = late_ranges[panel]
        late_tex.append(r"\nextgroupplot[title={" + f"({chr(97 + panel)}) {d['title']}"
                        + "}," + f"ymin={low},ymax={high},ytick={{{ticks}}}"
                        + (r",ylabel={Development PPL}" if panel == 0 else "") + "]")
        for arm, color, marker, label in (
            ("baseline", "BaseCopper", "square*", "Baseline"),
            ("v_silu", "GatedBlue", "*", "GatedV"),
        ):
            for values in d["per_seed_ppl"][arm].values():
                y = [values[i] for i in indices]
                assert all(low < value < high for value in y)
                late_tex.append(r"\addplot[" + color + "!35!white,line width=.55pt,no marks,forget plot] coordinates {"
                                + coords(x, y) + "};")
        for arm, color, marker, label in (
            ("baseline", "BaseCopper", "square*", "Baseline"),
            ("v_silu", "GatedBlue", "*", "GatedV"),
        ):
            y = [d["mean_ppl"][arm][i] for i in indices]
            late_tex.append(r"\addplot[" + color + ",line width=1.5pt,mark=" + marker
                            + ",mark size=1.5pt,mark options={solid}"
                            + (",forget plot" if panel else "") + "] coordinates {" + coords(x, y) + "};")
            if panel == 0:
                late_tex.append(r"\addlegendentry{" + label + "}")
        d["late_training_view"] = {
            "steps": [d["steps"][i] for i in indices], "training_progress_percent": x,
            "y_axis_range": [low, high], "saved_times_per_run": 5,
        }
    late_tex += [r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}"]
    (FIGURES / "development_ppl_late_training.tex").write_text("\n".join(late_tex) + "\n")
    (FIGURES / "paper_ppl_data.json").write_text(json.dumps(data, indent=2) + "\n")

    stems = ("development_relative_ppl", "development_ppl_late_training")
    if not args.tex_only:
        for stem in stems:
            with tempfile.TemporaryDirectory(prefix="gatedv_figure_") as temp:
                result = subprocess.run(
                    [args.pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                     "-output-directory", temp, str(FIGURES / (stem + ".tex"))],
                    cwd=FIGURES, capture_output=True, text=True,
                )
                if result.returncode:
                    raise RuntimeError(result.stdout[-6000:] + result.stderr[-2000:])
                (FIGURES / (stem + ".pdf")).write_bytes((Path(temp) / (stem + ".pdf")).read_bytes())
            subprocess.run(
                [args.pdftoppm, "-singlefile", "-r", "210", "-png",
                 str(FIGURES / (stem + ".pdf")), str(FIGURES / stem)], check=True,
            )
    outputs = [FIGURES / (stem + extension) for stem in stems
               for extension in ((".tex",) if args.tex_only else (".tex", ".pdf", ".png"))]
    outputs.append(FIGURES / "paper_ppl_data.json")
    manifest = {
        "status": "STAGED_AUDITED_LR3E4_FIGURES_20261030",
        "source": "../results/learning_curves_lr3e4.csv", "source_sha256": SOURCE_SHA,
        "generator_sha256": sha(Path(__file__)), "figure1_regenerated": False,
        "figure2": {
            "stem": stems[0], "saved_times_per_run": 17, "raw_ppl_observations": 272,
            "individual_lines": "100 * (paired GatedV PPL / baseline PPL - 1)",
            "mean_line": "100 * (arithmetic mean GatedV PPL / arithmetic mean baseline PPL - 1)",
            "initialization_included": True, "mean_of_seed_ratios_used": False,
            "x_axis": "optimizer updates in thousands", "y_range_percent": [-5.2, 5.2],
        },
        "figure3": {
            "stem": stems[1], "saved_times_per_run": 5, "raw_ppl_observations": 80,
            "progress_percent": [75, 81.25, 87.5, 93.75, 100],
            "lines": "Per-seed PPL and arithmetic means by arm", "independent_y_scales": True,
        },
        "smoothing": False, "inferential_bands": False, "new_evaluations": False,
        "outputs_sha256": {str(f.relative_to(PAPER)): sha(f) for f in outputs},
    }
    (FIGURES / "paper_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "rendered": not args.tex_only}))


if __name__ == "__main__":
    main()
