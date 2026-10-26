"""Student-size tournament -> reports/student_size.md + figures.

    uv run python -I -m sanad.eval.size_report runs/size_v1

Hand-written conclusions in reports/student_size_findings.md are included if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from sanad.paths import CONFIGS, REPORTS
from sanad.train.tournament import per_fold, standings, to_markdown

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4
INK, INK2, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
TEACHERS = {
    "none": "No teacher", "resnet": "ResNet", "transformer_small": "Transformer S",
    "transformer_large": "Transformer L", "oxford_scratch": "Oxford, from scratch",
    "oxford_pretrained": "Oxford, pretrained", "ensemble": "Ensemble of all 5",
}
LINES = [("none", INK2, "--"), ("resnet", BLUE, "-"), ("oxford_pretrained", ORANGE, "-"),
         ("transformer_small", AQUA, "-")]


def name(teacher: str, size: str) -> str:
    return f"S:none@{size}" if teacher == "none" else f"S:{teacher}/standard@{size}"


def grid(folds, sizes, metric="spec_at_target", set_name="test") -> np.ndarray:
    return np.array([[np.nanmean(per_fold(folds, name(t, s), metric, set_name)) for s in sizes]
                     for t in TEACHERS])


def heatmap(folds, sizes, params, out: Path, n_folds: int) -> None:
    g = grid(folds, sizes)
    lo, hi = np.nanmin(g), np.nanmax(g)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    im = ax.imshow(g, cmap=matplotlib.colors.ListedColormap(SEQ), vmin=lo, vmax=hi, aspect="auto")
    best = np.unravel_index(np.nanargmax(g), g.shape)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            dark = (g[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
            ax.text(j, i, f"{g[i, j]:.1%}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if dark else INK, fontweight="bold" if (i, j) == best else "normal")
    ax.set_xticks(range(len(sizes)), [f"{params[s] / 1e3:.0f} k params" for s in sizes], fontsize=9,
                  color=INK2)
    ax.set_yticks(range(len(TEACHERS)), list(TEACHERS.values()), fontsize=9, color=INK2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(sizes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TEACHERS)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Student size × teacher\n", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(0, 1.02, f"Specificity at 95% sensitivity on test people, mean of {n_folds} folds. "
            "Darker = better; best in bold.", transform=ax.transAxes, fontsize=8.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                      format=matplotlib.ticker.PercentFormatter(1.0, 0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def size_lines(folds, sizes, params, out: Path) -> None:
    x = np.array([params[s] for s in sizes]) / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    panels = [("spec_at_target", "test", "Specificity at 95% sensitivity", "higher is better", True),
              ("false_alarms_per_hour", "elderly", "False alarms per hour, elderly", "lower is better", False)]
    for ax, (metric, set_name, title, sub, pct) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ends = []
        for t, color, style in LINES:
            y = [np.nanmean(per_fold(folds, name(t, s), metric, set_name)) for s in sizes]
            ax.plot(x, y, style, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=TEACHERS[t])
            ends.append([y[-1], TEACHERS[t]])
        # direct labels at the line ends, nudged apart so they never overlap
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.07
        ends.sort()
        orig = [e[0] for e in ends]
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        shift = (np.mean([e[0] for e in ends]) - np.mean(orig))  # re-centre the stack on the lines
        for e in ends:
            e[0] -= shift
        for yv, label in ends:
            ax.text(x[-1] * 1.12, yv, label, va="center", fontsize=8.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:.0f} k" for v in x])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(x[0] * 0.85, x[-1] * 2.4)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, 0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASE)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        ax.set_xlabel("student parameters", fontsize=9, color=INK2)
        ax.set_title(f"{title}\n", loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.text(0, 1.02, f"{sub}; mean of folds", transform=ax.transAxes, fontsize=8.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=str(CONFIGS / "tournament_size.yaml"))
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    folds = json.loads((run / "results.json").read_text())
    tcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sizes = list(tcfg["students"])
    params = {s: folds[0]["params"][name("none", s)] for s in sizes}

    from sanad.eval.pico_cost import pico_cost

    cost = {s: pico_cost(**tcfg["students"][s]) for s in sizes}
    REPORTS.mkdir(parents=True, exist_ok=True)
    heatmap(folds, sizes, params, REPORTS / "student_size_grid.png", len(folds))
    size_lines(folds, sizes, params, REPORTS / "student_size_lines.png")
    s = standings(folds, "S:", tcfg["primary_metric"])
    s.to_csv(run / "students.csv", index=False)

    cost_rows = "\n".join(
        f"| {params[k] / 1e3:.1f} k | {c['macs'] / 1e6:.2f} M | {c['tflite_kb']:.0f} KB | "
        f"~{c['arena_kb_est']:.0f} KB | {c['ms_est'][0]:.0f}–{c['ms_est'][1]:.0f} ms |"
        for k, c in cost.items())
    g_spec = grid(folds, sizes)
    g_eld = grid(folds, sizes, "false_alarms_per_hour", "elderly")
    g_falls = grid(folds, sizes, "event_sensitivity", "test")
    hdr = "| Teacher | " + " | ".join(f"{params[k] / 1e3:.0f} k" for k in sizes) + " |\n|---|" + "---|" * len(sizes)

    def tab(g, fmt):
        return hdr + "\n" + "\n".join(
            f"| {TEACHERS[t]} | " + " | ".join(fmt(v) for v in g[i]) + " |" for i, t in enumerate(TEACHERS))

    findings = REPORTS / "student_size_findings.md"
    md = f"""# Student-size tournament

Run `{run.name}`: {len(folds)} person-grouped folds, the same people and teachers as the
teacher tournament. Every Sanad-Tiny size is trained with no teacher and distilled from every
teacher (recipe `standard`: half labels, half teacher, T = 2). {len(s)} contenders in total.

{findings.read_text(encoding="utf-8") if findings.exists() else ""}
![Student size vs teacher](student_size_grid.png)

![Does size help?](student_size_lines.png)

## Specificity at 95% sensitivity (test people)

{tab(g_spec, lambda v: f"{v:.1%}")}

## False alarms per hour, elderly participants (never trained on)

{tab(g_eld, lambda v: f"{v:.1f}")}

## Falls caught (decision threshold from validation people)

{tab(g_falls, lambda v: f"{v:.1%}")}

## Cost on the Raspberry Pi Pico 2 W

| Student | MACs per window | int8 model (flash) | RAM for activations | Inference time (estimate) |
|---|---|---|---|---|
{cost_rows}

Pico 2 W: 4 MB flash, 520 KB RAM, Cortex-M33 at 150 MHz. All sizes use only these TFLite Micro
operators: {', '.join(cost[sizes[0]]['ops'])}. Times are a range for typical CMSIS-NN int8 speed
(0.25–0.6 multiply-accumulates per clock) and must be measured on the board; the model runs only
after a trigger, so it costs almost no battery.

## Full round-robin standings (all {len(s)} students)

{to_markdown(s, folds[0]["params"])}

---
Oxford harnet5 teacher: Copyright © 2022, University of Oxford (Academic Use Licence; students
distilled from it inherit it: fine for this non-commercial capstone, not for a product for sale).
"""
    (REPORTS / "student_size.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {REPORTS / 'student_size.md'} and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
