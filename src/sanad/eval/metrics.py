"""Metrics reported for every model, at the window level and at the event level.

Event level is what matters for the band: a fall counts as caught only if the trigger fired near the
impact AND the model said "fall"; falls the trigger missed count against sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def threshold_for_sensitivity(y: np.ndarray, prob: np.ndarray, target: float = 0.95) -> float:
    """Highest probability threshold that still flags >= `target` of the fall windows."""
    pos = np.sort(prob[y == 1])
    if len(pos) == 0:
        raise ValueError("no fall windows")
    k = int(np.floor((1.0 - target) * len(pos)))  # how many falls we may miss
    return float(pos[k])


def threshold_for_event_sensitivity(meta: pd.DataFrame, recs: pd.DataFrame, prob: np.ndarray,
                                    target: float = 0.95) -> float:
    """Highest threshold that still catches `target` of all falls end to end.

    A fall's score is the best probability among its fall windows; falls the trigger missed have
    no window and can never be caught, so they push the threshold down (they count in the total).
    """
    fall_ids = recs.loc[recs.is_fall, "recording_id"].astype(str)
    is_fall_win = (meta.label == 1).to_numpy()
    best = (pd.Series(prob[is_fall_win], index=meta.loc[is_fall_win, "recording_id"].astype(str).to_numpy())
            .groupby(level=0).max())
    scores = np.sort(best.reindex(fall_ids).fillna(-np.inf).to_numpy())[::-1]
    k = int(np.ceil(target * len(scores)))  # falls we must catch
    if k == 0:
        raise ValueError("no falls")
    thr = scores[min(k, len(scores)) - 1]
    if not np.isfinite(thr):  # target unreachable because of trigger misses: catch all we can
        thr = scores[np.isfinite(scores)].min()
    return float(thr)


def window_metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    keep = y >= 0
    y, prob = y[keep], prob[keep]
    pred = prob >= thr
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else float("nan")  # nan: no falls in this set
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": 2 * prec * sens / max(prec + sens, 1e-12) if tp + fn else float("nan"),
        "accuracy": (tp + tn) / max(len(y), 1),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def event_metrics(meta: pd.DataFrame, recs: pd.DataFrame, prob: np.ndarray, thr: float) -> dict:
    """Fall-event sensitivity (including trigger misses) and false alarms per hour of daily activity."""
    flagged = prob >= thr
    caught = set(meta.loc[(meta.label == 1).to_numpy() & flagged, "recording_id"].astype(str))
    falls = recs[recs.is_fall]
    adl = recs[~recs.is_fall]
    adl_ids = set(adl.recording_id.astype(str))
    false_alarms = int(np.sum(flagged & meta.recording_id.astype(str).isin(adl_ids).to_numpy()))
    hours = adl.duration_s.sum() / 3600.0
    return {
        "falls": len(falls),
        "falls_missed_by_trigger": int((~falls.detected.astype(bool)).sum()),
        "event_sensitivity": (len(caught & set(falls.recording_id.astype(str))) / len(falls)
                              if len(falls) else float("nan")),
        "adl_hours": hours,
        "false_alarms": false_alarms,
        "false_alarms_per_hour": false_alarms / hours if hours > 0 else float("nan"),
    }
