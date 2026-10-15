"""Reproduce manuscript figures from the retained development evaluations.

Run with Python 3, NumPy and Matplotlib: ``python paper/make_paper_figures.py``.
The script does not train or evaluate models and never modifies experiment data.
PPL is exponentiated per run before arithmetic averaging across paired seeds.
Connecting segments join observations; no smoothing or resampling is applied.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np


PAPER = Path(__file__).resolve().parent
OUT = PAPER / "figures"
SOURCE = PAPER.parent / "results" / "learning_curves.csv"
EXPECTED_SOURCE_SHA256 = "645b4c6d218c438c012d70b2c047c59dcba9281f5a1a457f6c9afbbc0faed21a"
STUDIES = {
    "small_confirmation": dict(title="Small (2.33M)", seeds=list(range(20261201, 20261206)),
                               total=32768, lr=0.0048, step=2048),
    "medium_confirmation": dict(title="Medium (26.16M)", seeds=list(range(20270211, 20270214)),
                                total=327680, lr=0.0006, step=20480),
}
BLUE = "#286B93"
ORANGE = "#C16A32"
ARMS = {"baseline": dict(label="Baseline", color=BLUE, marker="^", ls="-"),
        "v_silu": dict(label="GatedV", color=ORANGE, marker="o", ls="--")}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.65, "xtick.major.width": 0.65,
    "ytick.major.width": 0.65, "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035,
                metadata={"Title": stem.replace("_", " "), "CreationDate": None,
                          "ModDate": None})
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def load_data():
    assert sha(SOURCE) == EXPECTED_SOURCE_SHA256, "The source changed; review before replotting."
    rows = []
    with SOURCE.open(newline="") as handle:
        for original in csv.DictReader(handle):
            if original["phase"] not in STUDIES:
                continue
            r = dict(original)
            for key in ("seed", "step", "total_steps", "training_targets", "targets"):
                r[key] = int(r[key])
            for key in ("lr", "dev_nll", "ppl"):
                r[key] = float(r[key])
            r["dev_ppl"] = math.exp(r["dev_nll"])
            assert math.isclose(r["ppl"], r["dev_ppl"], rel_tol=1e-12)
            assert math.isfinite(r["dev_ppl"]) and r["dev_ppl"] > 0
            r["update_fraction"] = r["step"] / r["total_steps"]
            assert r["training_targets"] == r["step"] * 2048
            rows.append(r)
    assert len(rows) == 272
    data = {}
    for phase, design in STUDIES.items():
        rr = [r for r in rows if r["phase"] == phase]
        assert sorted({r["seed"] for r in rr}) == design["seeds"]
        assert {r["lr"] for r in rr} == {design["lr"]}
        assert {r["total_steps"] for r in rr} == {design["total"]}
        assert {r["arm"] for r in rr} == set(ARMS)
        steps = list(range(0, design["total"] + 1, design["step"]))
        runs, means = {}, {}
        for arm in ARMS:
            runs[arm] = {}
            for seed in design["seeds"]:
                run = sorted((r for r in rr if r["arm"] == arm and r["seed"] == seed),
                             key=lambda r: r["step"])
                assert [r["step"] for r in run] == steps
                runs[arm][str(seed)] = [r["dev_ppl"] for r in run]
            means[arm] = [statistics.mean(v[i] for v in runs[arm].values()) for i in range(17)]
        delta = {str(seed): [g - b for g, b in zip(runs["v_silu"][str(seed)],
                                                     runs["baseline"][str(seed)])]
                 for seed in design["seeds"]}
        delta_mean = [statistics.mean(v[i] for v in delta.values()) for i in range(17)]
        assert all(math.isclose(d, g - b, abs_tol=1e-10)
                   for d, g, b in zip(delta_mean, means["v_silu"], means["baseline"]))
        data[phase] = dict(**design, steps=steps,
                           update_fraction=[s / design["total"] for s in steps],
                           per_seed_ppl=runs, mean_ppl=means, paired_delta_ppl=delta,
                           mean_paired_delta_ppl=delta_mean)
    return rows, data


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, pad=3)


def draw_ppl_lines(ax, d, start, inset=False):
    xx = np.asarray(d["steps"][start:]) / 1000
    for arm, style in ARMS.items():
        for vv in d["per_seed_ppl"][arm].values():
            ax.plot(xx, vv[start:], color=style["color"], linestyle=style["ls"],
                    linewidth=0.55 if inset else 0.85, alpha=0.55 if inset else 0.62,
                    marker=None if inset else style["marker"], markersize=2.9,
                    markeredgewidth=0, markerfacecolor=style["color"], zorder=2)
        ax.plot(xx, d["mean_ppl"][arm][start:], color="black", linestyle=style["ls"],
                linewidth=1.0 if inset else 1.65, marker=None if inset else style["marker"],
                markersize=3.8, markeredgewidth=0, markerfacecolor="black", zorder=3)


def plot_ppl(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.30))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.79, wspace=0.25)
    for index, (ax, d) in enumerate(zip(axes, data.values())):
        style_axes(ax)
        draw_ppl_lines(ax, d, start=1)
        ax.set_title(f"({chr(97 + index)}) {d['title']}", loc="left", pad=8)
        ax.set_xlabel(r"Optimizer updates ($10^3$)")
        ax.set_ylabel("Development perplexity")
        ax.set_xlim(0, d["total"] / 1000 * 1.025)
        ax.set_xticks(np.arange(5) * (8 if index == 0 else 80))
        ax.set_ylim((52, 154) if index == 0 else (19, 45))
        ax.set_yticks([60, 80, 100, 120, 140] if index == 0 else [20, 25, 30, 35, 40, 45])
        inset = ax.inset_axes([0.54, 0.57, 0.425, 0.35])
        draw_ppl_lines(inset, d, start=0, inset=True)
        inset.set_yscale("log")
        inset.set_title("Including initialization", fontsize=7, pad=2)
        inset.set_xlim(0, d["total"] / 1000)
        inset.set_ylim(10, 30000)
        inset.set_yticks([1e2, 1e4])
        inset.set_xticks([0, d["total"] / 1000])
        inset.set_xticklabels(["0", "32.8" if index == 0 else "327.7"])
        inset.yaxis.set_minor_locator(NullLocator())
        inset.tick_params(labelsize=6.5, length=2, pad=1.5)
        for spine in inset.spines.values():
            spine.set_color("0.65")
            spine.set_linewidth(0.5)
    handles = [Line2D([], [], color=BLUE, lw=0.9, marker="^", ms=3, mew=0, label="Baseline seeds"),
               Line2D([], [], color=ORANGE, lw=0.9, ls="--", marker="o", ms=3, mew=0,
                      label="GatedV seeds"),
               Line2D([], [], color="black", lw=1.65, marker="^", ms=4, mew=0,
                      label="Baseline mean"),
               Line2D([], [], color="black", lw=1.65, ls="--", marker="o", ms=4, mew=0,
                      label="GatedV mean")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.01), ncol=4,
               frameon=False, columnspacing=1.3, handlelength=2.3, handletextpad=0.55)
    save_figure(fig, "development_ppl")


def plot_delta(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.80, wspace=0.27)
    for index, (ax, d) in enumerate(zip(axes, data.values())):
        style_axes(ax)
        xx = d["update_fraction"][1:]
        marker = "^" if index == 0 else "o"
        for vv in d["paired_delta_ppl"].values():
            ax.plot(xx, vv[1:], color=ORANGE, alpha=0.70, lw=0.85,
                    marker=marker, ms=3.0, markeredgewidth=0, zorder=2)
        ax.axhline(0, color="0.4", lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.plot(xx, d["mean_paired_delta_ppl"][1:], color="black", lw=1.8,
                marker=marker, ms=4, markeredgewidth=0, zorder=3)
        ax.set_title(f"({chr(97 + index)}) {d['title']}", loc="left", pad=8)
        ax.set_xlabel("Fraction of scheduled updates")
        ax.set_ylabel(r"$\Delta$ perplexity (GatedV $-$ baseline)")
        ax.set_xlim(0, 1.025)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        if index == 0:
            ax.set_ylim(-4.2, 5.5)
            ax.set_yticks([-4, -2, 0, 2, 4])
        else:
            ax.set_ylim(-1.8, 0.18)
            ax.set_yticks([-1.5, -1, -0.5, 0])
            ax.yaxis.set_major_formatter(ScalarFormatter())
    fig.legend(handles=[Line2D([], [], color=ORANGE, lw=0.85, label="Paired seeds"),
                        Line2D([], [], color="black", lw=1.8, label="Mean paired difference")],
               loc="upper center", bbox_to_anchor=(0.53, 1.015), ncol=2,
               frameon=False, columnspacing=2.0, handlelength=2.8)
    save_figure(fig, "development_delta_ppl")


def plot_architecture():
    fig, ax = plt.subplots(figsize=(7.15, 4.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-0.1, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    nodes = {}

    def box(name, x, y, text, w=2.45, h=0.49, color="0.97", edge="0.45", fs=8.4):
        nodes[name] = (x, y, w, h)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                   boxstyle="round,pad=0.025,rounding_size=0.04",
                                   facecolor=color, edgecolor=edge, linewidth=0.65))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def anchor(name, edge):
        x, y, w, h = nodes[name]
        return {"top": (x, y + h / 2), "bottom": (x, y - h / 2),
                "left": (x - w / 2, y), "right": (x + w / 2, y)}[edge]

    def arrow(points, color="0.3"):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.7)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=7.0, linewidth=0.7, color=color,
                                     shrinkA=0.5, shrinkB=1.1))

    def link(a, b, ae="bottom", be="top", color="0.3"):
        arrow([anchor(a, ae), anchor(b, be)], color)

    ax.text(7.9, 8.7, r"Tokens $\rightarrow$ tied embedding $\rightarrow$ $L$ residual blocks"
            r" $\rightarrow$ final RMSNorm $\rightarrow$ tied vocabulary readout",
            fontsize=8.1, ha="center", va="center")
    ax.text(0.15, 8.0, "(a) Residual block", fontweight="bold", fontsize=9.5)
    ax.text(4.7, 8.0, "(b) Causal attention", fontweight="bold", fontsize=9.5)
    for name, y, label in [("x", 7.3, r"Hidden states $X$"), ("norm1", 6.45, "RMSNorm"),
                           ("att", 5.6, "Causal attention"), ("add1", 4.75, "Residual addition"),
                           ("norm2", 3.9, "RMSNorm"), ("ffn", 3.05, "SwiGLU FFN"),
                           ("add2", 2.2, "Residual addition"), ("out", 1.35, r"$X_{\rm next}$")]:
        box(name, 1.85, y, label, color="#ECF2F6" if name == "att" else "0.97")
    names = ["x", "norm1", "att", "add1", "norm2", "ffn", "add2", "out"]
    for a, b in zip(names[:-1], names[1:]):
        link(a, b)
    arrow([anchor("x", "left"), (0.2, 7.3), (0.2, 4.75), anchor("add1", "left")])
    arrow([anchor("add1", "right"), (3.52, 4.75), (3.52, 2.2), anchor("add2", "right")])
    ax.text(1.85, 0.63, r"Repeat for $L$ blocks", fontsize=8, ha="center", color="0.3")
    ax.plot([4.1, 4.1], [0.55, 8.15], color="0.82", lw=0.6)
    box("u", 10.4, 7.3, r"Normalized input $U$", w=3.2)
    box("q", 7.15, 6.05, r"$Q=UW_Q$", w=2.5)
    box("k", 10.4, 6.05, r"$K=UW_K$", w=2.5)
    box("v", 13.65, 6.05, r"$V=UW_V$", w=2.5)
    for name in ["q", "k", "v"]:
        link("u", name)
    box("qr", 7.15, 4.9, "Split heads\nRoPE", w=2.5, h=0.73)
    box("kr", 10.4, 4.9, "Split heads\nRoPE", w=2.5, h=0.73)
    box("vv", 13.65, 4.9, r"Baseline: $V$" + "\n" + r"GatedV: $\mathrm{SiLU}(V)$"
        + "\nthen split heads", w=3.0, h=1.05, color="#F8EADD", edge=ORANGE, fs=8.0)
    link("q", "qr")
    link("k", "kr")
    link("v", "vv", color=ORANGE)
    box("a", 8.75, 3.35, "Scaled dot products + causal mask\n"
        + r"$A_h=\mathrm{softmax}(\widehat Q_h\widehat K_h^\top/\sqrt{d_h}+M)$",
        w=5.5, h=0.88, color="#ECF2F6", fs=8.4)
    arrow([anchor("qr", "bottom"), (7.15, 3.79)])
    arrow([anchor("kr", "bottom"), (10.4, 3.79)])
    box("av", 11.2, 1.98, r"Weighted aggregation $A_h\widetilde V_h$", w=4.85, h=0.52, fs=8.4)
    link("a", "av")
    arrow([anchor("vv", "bottom"), (14.05, 3.95), (14.05, 1.98), anchor("av", "right")], ORANGE)
    box("wo", 11.2, 0.91, r"Concatenate heads; project with $W_O$", w=5.8, h=0.52, fs=8.4)
    link("av", "wo")
    save_figure(fig, "architecture_gatedv")


def write_data(rows, data):
    columns = ["phase", "arm", "seed", "lr", "step", "total_steps", "update_fraction",
               "training_targets", "dev_nll", "dev_ppl", "targets", "source_summary",
               "summary_sha256", "source_arrays", "arrays_sha256", "audit_source", "audit_sha256"]
    with (OUT / "paper_ppl_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "paper_ppl_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "step", "update_fraction", "paired_seeds", "baseline_mean_ppl",
                         "gatedv_mean_ppl", "mean_paired_delta_ppl"])
        for phase, d in data.items():
            for i, step in enumerate(d["steps"]):
                writer.writerow([phase, step, d["update_fraction"][i], len(d["seeds"]),
                                 d["mean_ppl"]["baseline"][i], d["mean_ppl"]["v_silu"][i],
                                 d["mean_paired_delta_ppl"][i]])
    (OUT / "paper_ppl_data.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    rows, data = load_data()
    write_data(rows, data)
    plot_architecture()
    plot_ppl(data)
    plot_delta(data)
    outputs = [f"{stem}.{ext}" for stem in ["architecture_gatedv", "development_ppl",
                                            "development_delta_ppl"] for ext in ["pdf", "png"]]
    outputs += ["paper_ppl_points.csv", "paper_ppl_summary.csv", "paper_ppl_data.json"]
    manifest = {
        "source": "results/learning_curves.csv", "source_sha256": sha(SOURCE),
        "script": "paper/make_paper_figures.py", "script_sha256": sha(Path(__file__)),
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "scope": "Saved development evaluations from confirmation runs only; no held-out trajectories.",
        "formulas": {"run_ppl": "exp(dev_nll)", "mean_ppl": "arithmetic mean of per-seed PPL",
                     "paired_delta": "PPL_GatedV(seed, step) - PPL_baseline(seed, step)",
                     "mean_delta": "arithmetic mean of paired PPL differences",
                     "update_fraction": "step / total_steps"},
        "validation": {"source_digest_matches": True, "saved_ppl_matches_exp_nll": True,
                       "identities_seeds_learning_rates_budgets_checked": True,
                       "complete_identical_step_grids": True, "observations": len(rows),
                       "saved_points_per_run": 17, "selection_runs_included": False,
                       "heldout_values_included": False,
                       "original_source_summary_array_audit_hashes": "Retained from CSV provenance; not rehashed by this plotting script."},
        "display": {"development_ppl": "Linear main panels show 16 post-update points; insets show all 17 including initialization with logarithmic y axes.",
                    "development_delta_ppl": "16 post-update points; initialization omitted from display to resolve training differences; retained in numeric exports.",
                    "thin_lines": "Individual seeds / paired differences, never confidence bounds.",
                    "black_lines": "Arithmetic means across seeds; no smoothing or fitted curves."},
        "initialization": {phase: {"baseline_mean_ppl": d["mean_ppl"]["baseline"][0],
                                    "gatedv_mean_ppl": d["mean_ppl"]["v_silu"][0],
                                    "mean_paired_delta_ppl": d["mean_paired_delta_ppl"][0],
                                    "paired_delta_range": [min(v[0] for v in d["paired_delta_ppl"].values()),
                                                            max(v[0] for v in d["paired_delta_ppl"].values())]}
                           for phase, d in data.items()},
        "outputs_sha256": {name: sha(OUT / name) for name in outputs},
    }
    (OUT / "paper_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for phase, d in data.items():
        print(phase, "final development PPL:", d["mean_ppl"]["baseline"][-1],
              d["mean_ppl"]["v_silu"][-1], "paired delta:", d["mean_paired_delta_ppl"][-1])


if __name__ == "__main__":
    main()
