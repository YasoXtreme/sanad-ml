"""WEDA-FALL (Fitbit Sense on the wrist). https://github.com/joaojtmarques/WEDA-FALL

Facts checked on the files (2026-10-26):
- accel in m/s^2 (z ~ 9.8 face-up at rest), gyro in rad/s (Fitbit/W3C sensor API).
- Axes already match the Sanad frame on the left wrist: x toward the hand (gravity sits on -x while
  walking with the arm down), z out of the screen. Treated as a left-wrist device.
- Timestamps are *delivery* times: batches of 4-5 samples ~2 ms apart, then a ~100 ms jump. The
  true rate differs per recording (~40 or ~50 Hz). So each recording is split at gaps > 0.5 s, the
  rate is estimated per segment from (samples / duration), samples are spaced evenly, and the
  segment is resampled to 50 Hz.
- Falls: fall_timestamps.csv gives a ~3 s fall window per trial; the impact is the |a| peak in it.
- Users U01-U14 are young (U14 is 46) and did falls + ADLs; U21-U31 are 77-95 and did ADLs only.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from sanad.data.harmonize import FS, Recording, harmonize, ms2_to_g, rads_to_dps
from sanad.data.windowing import magnitude

GAP_S = 0.5
MIN_SEGMENT_S = 2.0
AGES = {**{f"U{i:02d}": a for i, a in enumerate(
    [22, 22, 20, 21, 23, 22, 21, 23, 22, 23, 23, 23, 22, 46], start=1)},
    **{f"U{i}": a for i, a in zip(range(21, 32), [95, 85, 82, 81, 81, 83, 89, 88, 77, 80, 88], strict=True)}}


def age_group(age: int) -> str:
    return "elderly" if age >= 65 else ("adult" if age >= 30 else "young")


def _segments(t: np.ndarray) -> list[slice]:
    cuts = np.flatnonzero(np.diff(t) > GAP_S) + 1
    bounds = np.r_[0, cuts, len(t)]
    return [slice(a, b) for a, b in pairwise(bounds)]


def _dataset_dir(raw_dir: Path) -> Path:
    hits = list(raw_dir.glob("*/dataset/50Hz")) + list(raw_dir.glob("dataset/50Hz"))
    if not hits:
        raise FileNotFoundError(f"WEDA-FALL 50Hz folder not found under {raw_dir}")
    return hits[0]


def load(raw_dir: Path, log=print) -> list[Recording]:
    root = _dataset_dir(raw_dir)
    falls = pd.read_csv(root.parent / "fall_timestamps.csv").set_index("filename")
    recs: list[Recording] = []
    dropped = 0
    for act_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        act = act_dir.name
        is_fall = act.startswith("F")
        for acc_file in sorted(act_dir.glob("U*_R*_accel.csv")):
            user, trial, kind = acc_file.stem.split("_", 2)
            if kind != "accel":  # skip *_vertical_accel.csv (gravity-aligned copy, time in col 4)
                continue
            key = f"{act}/{user}_{trial}"
            a = pd.read_csv(acc_file).to_numpy(dtype=np.float64)
            g = pd.read_csv(acc_file.with_name(f"{user}_{trial}_gyro.csv")).to_numpy(dtype=np.float64)
            n = min(len(a), len(g))
            t, acc, gyro = a[:n, 0], a[:n, 1:4], g[:n, 1:4]
            if is_fall and key not in falls.index:
                dropped += 1
                continue
            win = (falls.loc[key, "start_time"], falls.loc[key, "end_time"]) if is_fall else None
            kept = False
            for k, seg in enumerate(_segments(t)):
                ts = t[seg]
                dur = ts[-1] - ts[0]
                if dur < MIN_SEGMENT_S:
                    continue
                if is_fall and not (ts[0] <= win[0] and win[1] <= ts[-1] + 0.5):
                    continue  # this piece doesn't contain the fall
                fs_in = round((len(ts) - 1) / dur)  # e.g. 40 or 50
                acc_h, gyro_h = harmonize(ms2_to_g(acc[seg]), rads_to_dps(gyro[seg]), fs_in)
                impact = None
                if is_fall:
                    amag = magnitude(acc_h)
                    lo = min(int((win[0] - ts[0]) * FS), len(amag) - 1)
                    hi = min(int((win[1] - ts[0]) * FS) + 1, len(amag))
                    impact = (lo + int(np.argmax(amag[lo:hi]))) / FS
                    kept = True
                age = AGES.get(user, 30)
                recs.append(Recording(
                    dataset="weda_fall", subject_id=f"weda_fall:{user}",
                    recording_id=f"weda_fall:{key}" + (f"#s{k}" if k else ""),
                    activity=act, is_fall=is_fall, acc=acc_h, gyro=gyro_h, impact_t=impact,
                    age_group=age_group(age),
                    meta={"fs_in": fs_in, "age": age},
                ))
                if not is_fall:
                    kept = True
            if not kept:
                dropped += 1
    log(f"[weda_fall] {len(recs)} recordings, {dropped} trials dropped (gaps over the fall / too short)")
    return recs
