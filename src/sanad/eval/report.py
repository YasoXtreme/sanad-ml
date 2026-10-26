"""Turn a CV run (+ the final model report) into reports/results.md and a comparison figure.

    uv run python -I -m sanad.eval.report runs/cv_v1
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

from sanad.paths import MODELS, REPORTS

NAMES = {
    "threshold": "Threshold rule",
    "forest": "Random Forest",
    "teacher": "Teacher (1D ResNet)",
    "student": "Sanad-Tiny, trained alone",
    "student_kd": "Sanad-Tiny, distilled",
}
DEPLOYED = "student_kd"
# reference palette (dataviz skill): deployed model in series blue, others recede in muted ink
BLUE, MUTED, INK, INK2, GRID, BASE, SURFACE = (
    "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb")
PANELS = [
    ("test", "event_sensitivity", "Falls caught", "share of test falls (trigger + model)", "pct"),
    ("test", "specificity", "Non-falls correctly ignored", "share of non-fall trigger windows", "pct"),
    ("test", "false_alarms_per_hour", "False alarms per hour", "daily-activity recordings, test people", "num"),
    ("elderly", "false_alarms_per_hour", "False alarms per hour, elderly",
     "WEDA participants aged 77-95, never trained on", "num"),
]


def fold_values(folds: list[dict], set_name: str, approach: str, metric: str) -> np.ndarray:
    return np.array([f[f"{set_name}/{approach}"][metric] for f in folds
                     if f"{set_name}/{approach}" in f], dtype=float)


def figure(folds: list[dict], out: Path) -> None:
    order = list(NAMES)[::-1]  # deployed model on top
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (set_name, metric, title, sub, kind) in zip(axes, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        means = [np.nanmean(fold_values(folds, set_name, a, metric)) for a in order]
        stds = [np.nanstd(fold_values(folds, set_name, a, metric)) for a in order]
        colors = [BLUE if a == DEPLOYED else MUTED for a in order]
        y = np.arange(len(order))
        ax.barh(y, means, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(means, y, xerr=stds, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        top = max(m + s for m, s in zip(means, stds, strict=True))
        for yi, m, s in zip(y, means, stds, strict=True):
            label = f"{m:.0%}" if kind == "pct" else f"{m:.1f}"
            ax.text(m + s + top * 0.03, yi, label, va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.18 if kind == "pct" else top * 1.3)
        if kind == "pct":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{title}\n", loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
    axes[0].set_yticks(np.arange(len(order)), [NAMES[a] for a in order])
    fig.text(0.01, -0.04, "Bars: mean over 5 person-grouped folds; whiskers: spread (1 SD). "
             "Blue = the model that runs on the band. Decision thresholds tuned for 95% sensitivity "
             "on validation people.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def table(folds: list[dict], set_name: str, metrics: list[tuple[str, str, str]]) -> str:
    head = "| Approach | " + " | ".join(m[1] for m in metrics) + " |"
    rows = [head, "|---|" + "---|" * len(metrics)]
    for a, name in NAMES.items():
        cells = []
        for metric, _, kind in metrics:
            v = fold_values(folds, set_name, a, metric)
            if np.all(np.isnan(v)):
                cells.append("–")
            elif kind == "pct":
                cells.append(f"{np.nanmean(v):.1%} ± {np.nanstd(v):.1%}")
            else:
                cells.append(f"{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}")
        bold = "**" if a == DEPLOYED else ""
        rows.append(f"| {bold}{name}{bold} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    folds = json.loads((run / "results.json").read_text())
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig_path = REPORTS / "cv_comparison.png"
    figure(folds, fig_path)

    test_metrics = [
        ("event_sensitivity", "Falls caught (event)", "pct"),
        ("sensitivity", "Sensitivity (window)", "pct"),
        ("specificity", "Specificity (window)", "pct"),
        ("f1", "F1", "num"),
        ("auc", "AUC", "num"),
        ("false_alarms_per_hour", "False alarms / hour", "num"),
    ]
    elderly_metrics = [("specificity", "Specificity (window)", "pct"),
                       ("false_alarms_per_hour", "False alarms / hour", "num")]
    sizes = folds[0]["sizes"]
    trig = [f["trigger_g"] for f in folds]
    final = MODELS / "final_report.json"
    final_md = ""
    if final.exists():
        fr = json.loads(final.read_text())
        s, q = fr["val_student_float"], fr["val_student_int8"]
        final_md = f"""
## Final model (trained on all data, exported for the Pico 2 W)

| | Value |
|---|---|
| Trigger threshold | {fr['trigger_g']:.2f} g |
| Decision threshold | fall if probability ≥ {fr['decision_prob']:.3f} |
| Sanad-Tiny size | {fr['student']['params']:,} parameters, {fr['student']['macs'] / 1e6:.2f} M MACs, {fr['tflite_bytes'] / 1024:.1f} KB int8 |
| Teacher size | {fr['teacher']['params']:,} parameters, {fr['teacher']['macs'] / 1e6:.1f} M MACs |
| Validation, float | sensitivity {s['sensitivity']:.1%}, specificity {s['specificity']:.1%}, AUC {s['auc']:.3f} |
| Validation, int8 (what the band runs) | sensitivity {q['sensitivity']:.1%}, specificity {q['specificity']:.1%}, AUC {q['auc']:.3f} |
| TFLite ops | {', '.join(fr['tflite_ops'])} |

The validation rows come from only a handful of held-out people (they set the decision threshold),
so they run higher than the cross-validation tables above; quote the cross-validation numbers.

Files: `models/sanad_tiny_int8.tflite`, `models/firmware/sanad_model_data.cc/.h`,
`models/firmware/sanad_model_config.h`.
"""
    md = f"""# Sanad fall detection: results

Cross-validation run `{run.name}`: 5 folds split **by person** (nobody is in both training and test),
over FallAllD, WEDA-FALL and PAMAP2. WEDA's 11 participants aged 77–95 are never trained on and form
a separate "elderly" test. Every number below comes from people the model never saw.

![Comparison of the five approaches](cv_comparison.png)

## Test people (falls and daily activities)

{table(folds, "test", test_metrics)}

"Falls caught" counts a fall only if the trigger fired at the impact **and** the model said fall, so
falls the trigger missed count as misses. Window metrics score only the windows the trigger produced.

## Elderly participants (daily activities only, never trained on)

{table(folds, "elderly", elderly_metrics)}

## Reading the numbers

- **Targets:** the approval form asks for ≥ 90 % sensitivity and ≥ 90 % specificity on unseen people.
  The distilled Sanad-Tiny meets both in cross-validation, with little margin on sensitivity.
- **False alarms per hour are a worst case.** The daily-activity recordings are short, dense clips
  of the hardest movements (jogging, jumping, clapping, banging a table, rope jumping, sitting down
  hard): the trigger fires about 170–310 times per hour of them. Ordinary days have far fewer
  impacts, and every remaining false alarm still has to survive the 30 s "I'm OK" countdown. The
  real-life rate has to be measured with the band (target: ≤ 2 per 24 h).
- **Distillation pays off where it matters:** compared with the same small network trained alone,
  the distilled one catches slightly more falls and gives far fewer false alarms on the elderly.
- **Spread between folds is large** (± several points): only ~35 people, and some folds are mostly
  one dataset. Data from the real band (Phase 4) is the biggest expected improvement.

## Setup

- Trigger threshold per fold (≥ 99 % of training falls): {min(trig):.2f}–{max(trig):.2f} g
- Teacher: {sizes['teacher']['params']:,} parameters, {sizes['teacher']['macs'] / 1e6:.1f} M MACs per window
- Sanad-Tiny: {sizes['student']['params']:,} parameters, {sizes['student']['macs'] / 1e6:.2f} M MACs per window
- Decision thresholds picked on validation people for 95 % sensitivity, then applied unchanged to test people.
- Augmentation: random rotation (±25°), any roll around the forearm, left/right mirroring, gain, time shift, noise.
{final_md}"""
    (REPORTS / "results.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {REPORTS / 'results.md'} and {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
