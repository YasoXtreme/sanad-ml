"""PAMAP2 (UCI 231): daily activities only, IMU on the dominant wrist, 100 Hz. Role: negatives.

File layout: PAMAP2_Dataset/{Protocol,Optional}/subjectNNN.dat, space-separated, 54 columns:
  0 timestamp (s), 1 activityID, 2 heart rate, then 17 columns per IMU (hand, chest, ankle):
  +0 temperature, +1..3 acc +-16 g (m/s^2), +4..6 acc +-6 g, +7..9 gyro (rad/s), +10..12 mag,
  +13..16 orientation (invalid).
Axes, checked from gravity on all 8 usable subjects (2026-10-26):
  - walking (arm down): gravity on -x in everyone -> sensor x points to the hand.
  - desk / sitting: "up" lies between +y and +z (wrist rolled ~45 deg thumb-up), and the y sign
    flips only for subject 108, the one left-hander -> the sensor sits the same way on each wrist,
    so its axes already are the Sanad frame (x to the hand, z out of the back of the wrist, y = z x x).
  Picking z per subject from desk gravity was unreliable (y and z are almost equal there), so the
  mapping is fixed and only *checked* per subject. Right-wrist subjects are mirrored to the left.
Subject 109 has no walking/sitting data in this release and is skipped.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from sanad.data.harmonize import Recording, harmonize, mirror, ms2_to_g, rads_to_dps

FS_IN = 100
HAND = 3  # first hand-IMU column
ACTIVITIES = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work", 11: "car_driving",
    12: "stairs_up", 13: "stairs_down", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry", 19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping",
}
LEFT_HANDED = {"108"}  # subjectInformation.pdf: subject 108 is left-handed
MIN_SEGMENT_S = 4.0
MAX_NAN_GAP_S = 0.1


def _check_axes(acc_g: np.ndarray, act: np.ndarray) -> str | None:
    """Return a warning if this subject's gravity doesn't fit the fixed mounting."""
    walk = acc_g[act == 4].mean(0)
    desk = acc_g[np.isin(act, [2, 10, 17, 18])].mean(0)
    if walk[0] > -0.6:
        return f"walking gravity {np.round(walk, 2)} not mostly on -x"
    if desk[2] < 0.2:
        return f"sitting/desk gravity {np.round(desk, 2)} has no +z part"
    return None


def _segments(act: np.ndarray, valid: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, stop, activity) runs of one activity with valid IMU data."""
    key = np.where(valid, act, -1)
    edges = np.flatnonzero(np.diff(key)) + 1
    bounds = np.r_[0, edges, len(key)]
    return [(a, b, int(key[a])) for a, b in pairwise(bounds) if key[a] >= 0]


def _fill_short_nans(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate NaN gaps up to MAX_NAN_GAP_S; returns data and a valid mask for longer gaps."""
    df = pd.DataFrame(x)
    filled = df.interpolate(limit=int(MAX_NAN_GAP_S * FS_IN), limit_area="inside")
    return filled.to_numpy(), ~filled.isna().any(axis=1).to_numpy()


def load(raw_dir: Path, log=print) -> list[Recording]:
    files = sorted(raw_dir.rglob("subject1*.dat"))
    if not files:
        inner = list(raw_dir.glob("*.zip"))
        raise FileNotFoundError(f"no PAMAP2 subject files under {raw_dir} (unzip {inner}?)")
    sessions = []
    for f in files:
        data = pd.read_csv(f, sep=r"\s+", header=None).to_numpy(dtype=np.float64)
        act = data[:, 1].astype(int)
        imu, valid = _fill_short_nans(data[:, [HAND + 1, HAND + 2, HAND + 3, HAND + 7, HAND + 8, HAND + 9]])
        sessions.append((f.stem.replace("subject", ""), f.parent.name.lower(), act, valid,
                         ms2_to_g(imu[:, :3]), rads_to_dps(imu[:, 3:])))

    # one mounting per subject: check it on all of that subject's sessions together
    usable = set()
    for subj in sorted({s[0] for s in sessions}):
        acts = np.concatenate([s[2][s[3]] for s in sessions if s[0] == subj])
        accs = np.concatenate([s[4][s[3]] for s in sessions if s[0] == subj])
        if not ((acts == 4).any() and np.isin(acts, [2, 10, 17, 18]).any()):
            log(f"[pamap2] subject {subj}: no walking/sitting data to check the mounting, skipped")
        elif warning := _check_axes(accs, acts):
            log(f"[pamap2] subject {subj}: {warning}, skipped")
        else:
            usable.add(subj)

    recs: list[Recording] = []
    for subj, session, act, valid, acc_g, gyro_dps in sessions:
        if subj not in usable:
            continue
        m = np.eye(3)  # sensor axes are the Sanad frame (see module docstring)
        right = subj not in LEFT_HANDED
        for k, (a, b, code) in enumerate(_segments(act, valid)):
            if (b - a) / FS_IN < MIN_SEGMENT_S:
                continue
            acc = acc_g[a:b] @ m.T
            gyro = gyro_dps[a:b] @ m.T
            if right:
                acc, gyro = mirror(acc, gyro)
            acc_h, gyro_h = harmonize(acc, gyro, FS_IN)  # axes already mapped
            recs.append(Recording(
                dataset="pamap2", subject_id=f"pamap2:{subj}",
                recording_id=f"pamap2:{subj}:{session}:{k}", activity=ACTIVITIES.get(code, str(code)),
                is_fall=False, acc=acc_h, gyro=gyro_h, age_group="young",
                meta={"axes": m.tolist(), "right_wrist": right},
            ))
        log(f"[pamap2] subject {subj} {session}: {'right (mirrored)' if right else 'left'} wrist")
    log(f"[pamap2] {len(recs)} recordings")
    return recs
