"""Cut trigger windows exactly the way the band does.

Device rule (the firmware must implement the same thing):
  1. Armed: wait until |a| >= threshold (|a| from acceleration already clipped to +-8 g).
  2. Peak = the highest |a| in the next PEAK_SEARCH samples (0.5 s), starting at the crossing.
  3. Until the window is complete (peak + 1.5 s), a strictly higher |a| moves the peak there and
     restarts the 1.5 s wait. So a stumble just before the impact can't hide the impact itself.
  4. Window = 2.5 s before the final peak to 1.5 s after it (PRE + POST = 200 samples at 50 Hz).
     Run the model, then arm again.

Labels:
  1  -> trigger in a fall recording whose peak is within +-MATCH_S of the labelled impact
  0  -> trigger in a daily-activity recording, or far (> IGNORE_S) from the impact in a fall recording
  -1 -> trigger near (but not at) a fall impact: ambiguous, left out of training and scoring
Falls that never produce a label-1 trigger are recorded as misses, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sanad.data.harmonize import FS, Recording

PRE = int(2.5 * FS)  # 125 samples before the peak
POST = int(1.5 * FS)  # 75 samples after (the peak sample counts as "after")
WINDOW = PRE + POST  # 200
PEAK_SEARCH = int(0.5 * FS)  # 25 samples
MATCH_S = 1.0
IGNORE_S = 4.0
FALL_PASS_RATE = 0.99  # threshold is set so this share of training falls trigger


def magnitude(acc: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(acc, dtype=np.float32) ** 2).sum(axis=-1))


def find_triggers(amag: np.ndarray, threshold: float) -> list[int]:
    """Peak sample indices produced by the device trigger rule."""
    peaks: list[int] = []
    n = len(amag)
    i = 0
    above = np.flatnonzero(amag >= threshold)
    while True:
        k = np.searchsorted(above, i)
        if k >= len(above):
            break
        start = int(above[k])
        seg = amag[start : min(start + PEAK_SEARCH, n)]
        p = start + int(np.argmax(seg))
        while True:  # a higher peak before the window completes takes over
            nxt = amag[p + 1 : min(p + POST, n)]
            higher = np.flatnonzero(nxt > amag[p])
            if not len(higher):
                break
            p = p + 1 + int(higher[0])
        peaks.append(p)
        i = p + POST  # disarmed until the window ends
    return peaks


def cut_window(acc: np.ndarray, gyro: np.ndarray, peak: int) -> np.ndarray:
    """(WINDOW, 6) raw window [ax, ay, az, gx, gy, gz]; edges are repeated if the recording is short."""
    x = np.concatenate([acc, gyro], axis=1)
    idx = np.clip(np.arange(peak - PRE, peak + POST), 0, len(x) - 1)
    return x[idx].astype(np.float32)


def impact_peak_g(rec: Recording) -> float:
    """Highest |a| within +-MATCH_S of the labelled impact (used to set the trigger threshold)."""
    amag = magnitude(rec.acc)
    c = round(rec.impact_t * FS)
    lo, hi = max(0, c - int(MATCH_S * FS)), min(len(amag), c + int(MATCH_S * FS) + 1)
    return float(amag[lo:hi].max())


def choose_threshold(train_recs: list[Recording], pass_rate: float = FALL_PASS_RATE) -> float:
    peaks = [impact_peak_g(r) for r in train_recs if r.is_fall]
    if not peaks:
        raise ValueError("no fall recordings to set the trigger threshold from")
    return float(np.quantile(peaks, 1.0 - pass_rate))


@dataclass
class Windows:
    x: np.ndarray  # (N, WINDOW, 6) float32, g and deg/s
    y: np.ndarray  # (N,) int8: 1 fall, 0 not a fall, -1 ignore
    meta: pd.DataFrame  # one row per window
    recs: pd.DataFrame  # one row per recording, including ones that never triggered

    def __len__(self) -> int:
        return len(self.y)

    @property
    def falls(self) -> pd.DataFrame:
        """Fall recordings with whether the trigger caught the impact (False = missed by the trigger)."""
        return self.recs[self.recs.is_fall].reset_index(drop=True)


def extract_windows(recs: list[Recording], threshold: float) -> Windows:
    xs, ys, rows, rec_rows = [], [], [], []
    for r in recs:
        amag = magnitude(r.acc)
        detected = False
        peaks = find_triggers(amag, threshold)
        for p in peaks:
            t = p / FS
            if r.is_fall:
                dt = abs(t - r.impact_t)
                label = 1 if dt <= MATCH_S else (-1 if dt <= IGNORE_S else 0)
            else:
                label = 0
            detected |= label == 1
            xs.append(cut_window(r.acc, r.gyro, p))
            ys.append(label)
            rows.append((r.dataset, r.subject_id, r.recording_id, r.activity, r.age_group,
                         r.is_fall, t, float(amag[p]), label))
        rec_rows.append((r.dataset, r.subject_id, r.recording_id, r.activity, r.age_group,
                         r.is_fall, r.duration_s, len(peaks), detected))
    meta = pd.DataFrame(rows, columns=["dataset", "subject_id", "recording_id", "activity",
                                       "age_group", "rec_is_fall", "peak_t", "peak_g", "label"])
    rec_df = pd.DataFrame(rec_rows, columns=["dataset", "subject_id", "recording_id", "activity",
                                             "age_group", "is_fall", "duration_s", "n_triggers",
                                             "detected"])
    x = np.stack(xs) if xs else np.zeros((0, WINDOW, 6), np.float32)
    return Windows(x=x, y=np.asarray(ys, dtype=np.int8), meta=meta, recs=rec_df)
