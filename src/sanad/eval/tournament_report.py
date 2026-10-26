"""Tournament results -> reports/tournament.md + two figures.

    uv run python -I -m sanad.eval.tournament_report runs/tournament_v1

Hand-written conclusions live in reports/tournament_findings.md and are included if present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from sanad.paths import CONFIGS, REPORTS
from sanad.train.tournament import standings, to_markdown

# reference palette (dataviz skill), light mode
BLUE, MUTED, INK, INK2, GRID, BASE, SURFACE = (
    "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
PRETTY = {
    "resnet": "ResNet (0.9 M)", "transformer_small": "Transformer S (0.55 M)",
    "transformer_large": "Transformer L (3.2 M)", "oxford_scratch": "Oxford net, from scratch (4.5 M)",
    "oxford_pretrained": "Oxford net, pretrained (4.5 M)", "ensemble": "Ensemble of all 5 (13.7 M)",
    "none": "No teacher",
}
RECIPES = {"standard": "Standard\n(α 0.5, T 2)", "teacher_only": "Teacher only\n(α 0, T 2)",
           "hot": "Hot\n(α 0.5, T 4)"}
PRIMARY = "Specificity @ 95% sens."


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def teacher_figure(t, out: Path, n_folds: int) -> None:
    df = t.iloc[::-1]  # winner on top
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(df))
    means, sds = df[PRIMARY].to_numpy(), df[f"{PRIMARY} sd"].to_numpy()
    colors = [BLUE if r == 1 else MUTED for r in df["rank"]]
    ax.barh(y, means, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(means, y, xerr=sds, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for yi, m, s, pts in zip(y, means, sds, df["points"], strict=True):
        ax.text(min(m + s, 1.0) + 0.004, yi, f"{m:.1%}  ({pts:g} {'pt' if pts == 1 else 'pts'})",
                va="center", fontsize=9, color=INK, clip_on=False)
    lo = max(0.0, np.floor((means - sds).min() * 20) / 20 - 0.05)
    ax.set_xlim(lo, 1.0)  # specificity can't exceed 100%; labels sit just outside the plot
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_yticks(y, [PRETTY.get(n.split(":", 1)[1], n) for n in df.contender])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Teacher tournament\n", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(0, 1.02, f"Specificity at 95% sensitivity on test people (mean ± SD over {n_folds} folds); "
            "points = round-robin wins. Axis starts at "
            f"{lo:.0%}.", transform=ax.transAxes, fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def distill_figure(s, out: Path, n_folds: int) -> None:
    by_name = {c.split(":", 1)[1]: v for c, v in zip(s.contender, s[PRIMARY], strict=True)}
    base = by_name.get("none", np.nan)
    teachers = [t for t in PRETTY if t != "none" and any(k.startswith(f"{t}/") for k in by_name)]
    grid = np.array([[by_name.get(f"{t}/{r}", np.nan) for r in RECIPES] for t in teachers])
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    cmap = matplotlib.colors.ListedColormap(SEQ)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(teachers) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
            delta = round((v - base) * 100, 1)
            dtxt = "±0.0" if delta == 0 else f"{delta:+.1f}"
            ax.text(j, i, f"{v:.1%}\n{dtxt} pts vs none", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if (i, j) == best else "normal")
    ax.set_xticks(range(len(RECIPES)), list(RECIPES.values()), fontsize=9, color=INK2)
    ax.set_yticks(range(len(teachers)), [PRETTY[t] for t in teachers], fontsize=9, color=INK2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(RECIPES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(teachers)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Distillation tournament: Sanad-Tiny students\n", loc="left", fontsize=12,
                 fontweight="bold", color=INK)
    ax.text(0, 1.02, f"Specificity at 95% sensitivity, mean of {n_folds} folds. No teacher: {base:.1%}. "
            "Darker = better; best in bold.", transform=ax.transAxes, fontsize=8.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0, 0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    folds = json.loads((run / "results.json").read_text())
    tcfg = yaml.safe_load((CONFIGS / "tournament.yaml").read_text(encoding="utf-8"))
    primary = tcfg["primary_metric"]
    t, s = standings(folds, "T:", primary), standings(folds, "S:", primary)
    REPORTS.mkdir(parents=True, exist_ok=True)
    teacher_figure(t, REPORTS / "tournament_teachers.png", len(folds))
    distill_figure(s, REPORTS / "tournament_distillation.png", len(folds))
    findings = REPORTS / "tournament_findings.md"
    md = f"""# Teacher and distillation tournament

Run `{run.name}`: {len(folds)} person-grouped folds, the same people, windows and augmentation as
`cv_v1` (FallAllD + WEDA-FALL + PAMAP2; WEDA's 77–95-year-olds held out as the elderly test).
Every contender is ranked by a round robin on **specificity at 95% sensitivity on test people**:
in each pairing, whoever is better on more folds wins the match (1 point; a draw is 0.5).

{findings.read_text(encoding="utf-8") if findings.exists() else ""}
## Teacher tournament

![Teacher tournament](tournament_teachers.png)

{to_markdown(t, folds[0]["params"])}

## Distillation tournament (every student is Sanad-Tiny: 11.8 k parameters, 23 KB int8)

![Distillation tournament](tournament_distillation.png)

{to_markdown(s, folds[0]["params"])}

Recipes: `standard` = half labels, half teacher (temperature 2, what `cv_v1` used);
`teacher_only` = only the teacher's softened answers; `hot` = half/half with softer answers
(temperature 4). "Falls caught", "Specificity" and the false-alarm columns use the decision
threshold picked on validation people, as on the band.

---
Oxford harnet5: Copyright © 2022, University of Oxford. Used under its Academic Use Licence for
this non-commercial school project (Yuan et al., npj Digital Medicine 2024). The weights are not
redistributed. A Sanad-Tiny distilled from it inherits those terms: fine for the capstone, but a
commercial product would need a licence from Oxford University Innovation.
"""
    (REPORTS / "tournament.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {REPORTS / 'tournament.md'} and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
