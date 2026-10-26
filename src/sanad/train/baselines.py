"""Two non-neural baselines the CNNs must beat (required for the ML rubric).

1. Threshold rule: score = peak |a|; flag a fall above a tuned second threshold.
2. Random Forest on ~20 hand-built features; small enough to run on the Pico via emlearn.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sanad.data.harmonize import FS
from sanad.data.windowing import PRE

FEATURES = [
    "peak_g", "min_g_before", "freefall_to_peak_s", "mean_g", "std_g", "ptp_g",
    "pre_std_g", "post_std_g", "post_mean_g", "jerk_max", "energy",
    "gyro_peak", "gyro_mean", "gyro_std", "post_gyro_mean",
    "tilt_change_deg", "pre_ax", "pre_az", "post_ax", "post_az",
]


def features(x6: np.ndarray) -> np.ndarray:
    """(N, 200, 6) raw windows -> (N, 20) features. The peak is at index PRE in every window."""
    acc, gyro = x6[..., :3], x6[..., 3:]
    a = np.linalg.norm(acc, axis=-1)
    w = np.linalg.norm(gyro, axis=-1)
    pre = slice(0, PRE - FS // 2)  # calm-ish part well before the impact
    post = slice(PRE + FS // 2, None)  # after the impact settles (last 1 s)
    lead = a[:, PRE - FS : PRE + 1]  # 1 s before the peak
    min_before = lead.min(axis=1)
    t_min = (FS - lead.argmin(axis=1)) / FS
    jerk = np.abs(np.diff(a, axis=1)) * FS
    pre_dir = acc[:, pre].mean(axis=1)
    post_dir = acc[:, post].mean(axis=1)
    cos = (pre_dir * post_dir).sum(-1) / (
        np.linalg.norm(pre_dir, axis=-1) * np.linalg.norm(post_dir, axis=-1) + 1e-6
    )
    f = np.stack([
        a[:, PRE], min_before, t_min, a.mean(1), a.std(1), np.ptp(a, axis=1),
        a[:, pre].std(1), a[:, post].std(1), a[:, post].mean(1), jerk.max(1), (a**2).mean(1),
        w.max(1), w.mean(1), w.std(1), w[:, post].mean(1),
        np.degrees(np.arccos(np.clip(cos, -1, 1))),
        pre_dir[:, 0], pre_dir[:, 2], post_dir[:, 0], post_dir[:, 2],
    ], axis=1)
    return f.astype(np.float32)


def threshold_scores(x6: np.ndarray) -> np.ndarray:
    """Score = peak |a| (g). Turned into a decision by a threshold tuned on validation."""
    return np.linalg.norm(x6[:, PRE, :3], axis=-1)


def train_forest(x6: np.ndarray, y: np.ndarray, seed: int = 0) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=60, max_depth=8, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    keep = y >= 0
    rf.fit(features(x6[keep]), y[keep])
    return rf
