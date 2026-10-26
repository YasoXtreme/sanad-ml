"""Where false alarms come from, what they'd mean in real life, and a post-fall stillness check.

    uv run python -I -m sanad.eval.false_alarms

Uses the out-of-fold scores written by sanad.train.train_final (runs/deploy/) and the calibrated
threshold in models/final_report.json. Writes reports/false_alarms.md + reports/false_alarms.png.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sanad.data.harmonize import FS
from sanad.data.prepare import load_interim
from sanad.data.windowing import POST, magnitude
from sanad.paths import MODELS, REPORTS, RUNS

BLUE, ORANGE, INK, INK2, GRID, BASE, SURFACE = (
    "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb")
WEDA = {"D01": "walking", "D02": "jogging", "D03": "stairs", "D04": "sit down / stand up",
        "D05": "collapse into a chair", "D06": "crouch, tie shoes", "D07": "stumble while walking",
        "D08": "gentle jump", "D09": "hitting a table", "D10": "clapping", "D11": "opening/closing door"}
FALLALLD = {1: "start clapping", 2: "clapping", 3: "stop clapping", 4: "clapping 1", 5: "start waving",
            6: "waving", 7: "stop waving", 8: "raising hand", 9: "moving hand down", 10: "hand up/down",
            11: "hand shaking", 12: "beating a table", 13: "sitting down", 14: "standing up",
            15: "failing to stand up", 16: "lying down", 17: "turning while lying", 18: "rising up",
            19: "start walking", 20: "walking slowly", 21: "stop walking", 22: "walking quickly",
            23: "stumbling", 24: "jogging slowly", 25: "jogging quickly", 26: "jumping slightly",
            27: "jumping strongly", 28: "bending down", 29: "start upstairs", 30: "upstairs",
            31: "stop upstairs", 32: "start downstairs", 33: "downstairs", 34: "stop downstairs",
            35: "upstairs quickly", 36: "downstairs quickly", 37: "lift up start", 38: "lift up stop",
            39: "lift down start", 40: "lift down stop", 41: "standing on a moving bus",
            42: "sitting on a moving bus", 43: "start jogging", 44: "stop jogging"}
KINDS = {
    "sitting, lying, TV, driving": {"lying", "sitting", "standing", "watching_tv", "computer_work",
                                    "car_driving"},
    "household chores": {"ironing", "folding_laundry", "vacuum_cleaning", "house_cleaning"},
    "walking, stairs": {"walking", "stairs_up", "stairs_down"},
    "sport (not in an elderly day)": {"running", "cycling", "nordic_walking", "playing_soccer",
                                      "rope_jumping"},
}
# an illustrative 16 h waking day of an older person living alone (no sport)
DAY = {"sitting, lying, TV, driving": 12.0, "household chores": 2.0, "walking, stairs": 2.0}
STILL_S = (3.0, 5.0, 8.0)  # how long after the window the stillness check looks (seconds)
GATE_GRID = [(a, g) for a in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30) for g in (20, 30, 40, 60, 80, 120)]
TARGETS = (0.95, 0.90, 0.85, 0.80, 0.70)


def activity_name(ds: str, code: str) -> str:
    if ds == "weda_fall":
        return WEDA.get(code, code)
    if ds == "fallalld":
        return FALLALLD.get(int(code.lstrip("A")), code)
    return code.replace("_", " ")


def post_motion(recs_by_id, rid: str, peak_t: float, seconds: float) -> tuple[float, float] | None:
    """(std of |a| in g, mean |gyro| in deg/s) during `seconds` right after the window ends."""
    r = recs_by_id[rid]
    a = round(peak_t * FS) + POST
    b = a + int(seconds * FS)
    if b > r.n:
        return None
    return float(magnitude(r.acc[a:b]).std()), float(np.linalg.norm(r.gyro[a:b], axis=1).mean())


def main() -> int:
    rep = json.loads((MODELS / "final_report.json").read_text())
    thr = rep["decision_prob"]
    oof = pd.read_parquet(RUNS / "deploy" / "oof_windows.parquet")
    oof_recs = pd.read_parquet(RUNS / "deploy" / "oof_recordings.parquet")
    test = oof[(oof.set == "test") & (oof.label >= 0)].copy()
    test["flag"] = test.prob >= thr
    adl_recs = oof_recs[~oof_recs.is_fall]

    # 1. rates per dataset: triggers/h x share flagged = false alarms/h
    rows = []
    for ds, g in test[test.label == 0].groupby("dataset"):
        hours = adl_recs.loc[adl_recs.dataset == ds, "duration_s"].sum() / 3600
        rows.append((ds, hours, len(g) / hours, g.flag.mean(), g.flag.sum() / hours))
    eld = oof[(oof.set == "elderly") & (oof.label == 0)].copy()
    eld["flag"] = eld.prob >= thr
    eld_h = rep["out_of_fold"]["elderly"]["hours"]
    n_folds = eld.fold.nunique()
    rows.append(("weda elderly (77-95)", eld_h, len(eld) / n_folds / eld_h, eld.flag.mean(),
                 eld.flag.sum() / n_folds / eld_h))
    rates = pd.DataFrame(rows, columns=["data", "hours", "triggers_per_h", "share_flagged", "false_alarms_per_h"])

    # 2. which activities cause them
    test["activity_name"] = [activity_name(d, a) for d, a in zip(test.dataset, test.activity, strict=True)]
    fa = test[(test.label == 0) & test.flag]
    top = (fa.groupby(["dataset", "activity_name"]).size().sort_values(ascending=False).head(12)
           .rename("false_alarms").reset_index())

    # 3. real-life estimate from PAMAP2 (minutes-long sessions of everyday activities)
    from sanad.eval.metrics import threshold_for_event_sensitivity

    pam = test[(test.dataset == "pamap2") & (test.label == 0)]
    pam_recs = adl_recs[adl_recs.dataset == "pamap2"].copy()
    kind_by_act = {a: k for k, acts in KINDS.items() for a in acts}
    pam_recs["kind"] = pam_recs.activity.map(kind_by_act)
    pam = pam.assign(kind=pam.recording_id.map(dict(zip(pam_recs.recording_id, pam_recs.kind, strict=True))))
    hours = pam_recs.groupby("kind").duration_s.sum() / 3600
    kinds = pd.DataFrame({
        "activity group": list(KINDS),
        "hours": [hours.get(k, 0) for k in KINDS],
        "triggers_per_h": [(pam.kind == k).sum() / hours.get(k, np.nan) for k in KINDS],
        "false_alarms_per_h": [((pam.kind == k) & pam.flag).sum() / hours.get(k, np.nan) for k in KINDS],
    })
    t_all = test.reset_index(drop=True)
    op_rows = []
    for target in TARGETS:
        th = threshold_for_event_sensitivity(t_all, oof_recs, t_all.prob.to_numpy(), target)
        per_day = sum(h * ((pam.kind == k) & (pam.prob >= th)).sum() / hours[k] for k, h in DAY.items())
        op_rows.append((target, th, per_day))
    ops = pd.DataFrame(op_rows, columns=["falls_caught", "threshold", "false_alarms_per_day"])
    day_triggers = sum(h * (pam.kind == k).sum() / hours[k] for k, h in DAY.items())
    day_fa = ops.loc[0, "false_alarms_per_day"]

    # 4. stillness check after the window, on everything the model flagged
    recs_by_id = {r.recording_id: r for d in test.dataset.unique() for r in load_interim(d)}
    flagged = test[test.flag].copy()
    recov_ids = [f"A{i}" for i in (102, 104, 106, 108, 110, 116, 118, 120, 122, 124, 126, 128)]
    gate_rows = []
    for secs in STILL_S:
        m = [post_motion(recs_by_id, rid, t, secs)
             for rid, t in zip(flagged.recording_id, flagged.peak_t, strict=True)]
        f = flagged.assign(a_std=[x[0] if x else np.nan for x in m],
                           g_mean=[x[1] if x else np.nan for x in m]).dropna(subset=["a_std"])
        falls = f[(f.label == 1) & ~f.activity.isin(recov_ids)]
        alarms = f[f.label == 0]
        best = None
        for a_lim, g_lim in GATE_GRID:  # loosest setting that keeps >= 95 % of falls, most alarms removed
            kept = ((falls.a_std <= a_lim) & (falls.g_mean <= g_lim)).mean()
            removed = 1 - ((alarms.a_std <= a_lim) & (alarms.g_mean <= g_lim)).mean()
            if kept >= 0.95 and (best is None or removed > best[3]):
                best = (a_lim, g_lim, kept, removed)
        gate_rows.append((secs, len(falls), len(alarms), *(best or (np.nan,) * 4)))
    gate = pd.DataFrame(gate_rows, columns=["seconds_checked", "falls", "false_alarms", "a_std_max_g",
                                            "gyro_mean_max_dps", "falls_kept", "false_alarms_removed"])

    # figure: false alarms by activity (top 12)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lab = [f"{n}  ({d.replace('_fall', '').replace('fallalld', 'FallAllD').replace('weda', 'WEDA').replace('pamap2', 'PAMAP2')})"
           for d, n in zip(top.dataset, top.activity_name, strict=True)][::-1]
    ax.barh(range(len(top)), top.false_alarms[::-1], color=BLUE, height=0.62, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(top.false_alarms[::-1]):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(top)), lab, fontsize=9, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Which activities fool the model\n", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(0, 1.02, "False alarms in test people's daily-activity recordings, all 5 folds pooled "
            "(45 k model, calibrated threshold)", transform=ax.transAxes, fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(REPORTS / "false_alarms.png", dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    def md_table(df, fmt):
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in cols) + " |")
        return "\n".join(out)

    pct = lambda v: f"{v:.1%}"
    f1 = lambda v: f"{v:.1f}"
    md = f"""# False alarms: where they come from and what they mean for the band

Model: the 45 k Sanad-Tiny on the band, decision threshold {thr:.3f} (calibrated to catch
{rep['out_of_fold']['target_event_sensitivity']:.0%} of falls end to end). All numbers are
out-of-fold: every person was scored by a model that never saw them.

## 1. A false alarm needs two things: the trigger fires, then the model says "fall"

{md_table(rates, {"hours": f1, "triggers_per_h": f1, "share_flagged": pct, "false_alarms_per_h": f1})}

The datasets' daily-activity clips are short, back-to-back recordings of exactly the movements
that set off the trigger (clapping, banging a table, jumping, sitting down hard), so the trigger
fires **hundreds of times per hour** in them. The model rejects most of those; the false alarms
per hour are what's left. That's a stress test, not a day in someone's life.

## 2. What fools the model

![False alarms by activity](false_alarms.png)

## 3. A rough real-life estimate: this is the real problem

PAMAP2 has minutes-long sessions of everyday activities (young adults). By activity group:

{md_table(kinds, {"hours": f1, "triggers_per_h": f1, "false_alarms_per_h": f1})}

For an illustrative 16 h waking day of an older person living alone
({', '.join(f'{h:.0f} h {k}' for k, h in DAY.items())}) the trigger would fire about
**{day_triggers:.0f} times**, and the model at today's threshold would let through about
**{day_fa:.0f} false alarms per day**. The trade-off with the threshold:

{md_table(ops, {"falls_caught": pct, "threshold": lambda v: f"{v:.3f}", "false_alarms_per_day": lambda v: f"{v:.0f}"})}

The approval-form target is **≤ 2 false alarms per 24 h**. No threshold gets there while still
catching most falls, so the trigger + model alone are **not enough**: people would quickly learn to
ignore the band. (PAMAP2's people are young and brisk, which inflates chores and stairs; treat this
as an order of magnitude until the band measures it.)

## 4. A stillness check after the impact

After a real fall the wrist mostly stays still; after clapping, vacuuming or going downstairs it
keeps moving. For everything the model flags, look at the next few seconds after the window
(variation of |a| and mean rotation speed) and only continue if the wrist is still. Best setting
per check length that keeps ≥ 95 % of falls (falls where the person gets back up are left out:
they don't need an emergency call):

{md_table(gate, {"falls": str, "false_alarms": str, "a_std_max_g": lambda v: f"{v:.2f}",
                 "gyro_mean_max_dps": lambda v: f"{v:.0f}", "falls_kept": pct, "false_alarms_removed": pct})}

Longer checks are only possible on fewer recordings (most public fall clips end a few seconds after
the impact), so the 8 s row rests mostly on FallAllD. Real fall-detection watches wait much longer
(around a minute without movement) before calling for help; public data can't test that, the band can.

## 5. What this means for Sanad

1. **Don't ring on the model alone.** Model says "fall" → wait for stillness → only then start the
   30 s countdown. Everything that keeps moving (most false alarms) never makes a sound.
2. **Make the first step quiet.** A short vibration ("did you fall?") before the loud alarm costs
   almost nothing when it's wrong.
3. **Collect the band's own "normal day" data.** A few days of wear by the team and grandparents,
   with every trigger logged, turns stairs, chores and prayer movements into training examples.
   The datasets' worst offenders (stairs down, vacuuming, chores, collapsing into a chair) are
   exactly what the band will see.
"""
    (REPORTS / "false_alarms.md").write_text(md, encoding="utf-8", newline="\n")
    print(rates.round(3).to_string(index=False))
    print(kinds.round(2).to_string(index=False))
    print(f"day estimate: {day_triggers:.0f} triggers, {day_fa:.2f} false alarms")
    print(ops.round(3).to_string(index=False))
    print(gate.round(3).to_string(index=False))
    print(f"top activities:\n{top.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
