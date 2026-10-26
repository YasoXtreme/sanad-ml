"""FallAllD (IEEE DataPort, doi 10.21227/bnya-mn34): wrist logger only. Role: main falls + ADLs.

Format (from the authors' FallAllD_2_PYTHON_Structure.py / Plot_FallAllD_Record.m):
- one file per sensor: S<subject:2>_D<device:1>_A<activity:3>_T<trial:2>_<A|G|M|B>.dat
  device 1 = neck, 2 = wrist, 3 = waist. Comma-separated int16 counts, 3 columns.
- 238 Hz, 20 s per recording; acc 0.000244 g/count (+-8 g), gyro 0.07 deg/s per count (+-2000 dps).
- Activity IDs 1-44 are ADLs, 101-135 falls (the authors' ActivityID2Str.m; "rec." = fall then
  recovery, still a fall).
- No impact labels: the impact is the |a| peak within +-2 s of the recording centre (t = 10 s).
- Mounting is NOT consistent across subjects (checked 2026-10-26 from gravity): most read +1 g on
  sensor +x while walking (x points up the arm), S04 reads -1 g on x (logger reversed), S13 reads
  -1 g on y (turned 90 deg), S01 splits it between x and y (~45 deg), and the y sign while walking
  differs between subjects (wrist side / roll vary). So per subject we rotate the arm-down gravity
  direction onto the typical FallAllD one (median of the normally mounted subjects), then apply the
  usual sensor -> Sanad mapping (x points up the arm, so Sanad x = -sensor x). The roll around the
  forearm stays unknown, which is why training randomizes roll (augment.roll_deg) and mirroring.
  A real band also slides around the wrist.
We read the plain .dat text files, never the .pkl files (unpickling untrusted files can run code).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from sanad.data.harmonize import FS, Recording, axis_map_matrix, harmonize, rotation_between
from sanad.data.windowing import magnitude

FS_IN = 238
ACC_G_PER_COUNT = 0.000244
GYRO_DPS_PER_COUNT = 0.07
FALL_MIN_ID = 101
IMPACT_CENTRE_S, IMPACT_SEARCH_S = 10.0, 2.0
WRIST = "2"
WALK_IDS = {19, 20, 21, 22}  # start walking, walking slowly, stop walking, walking quickly
NAME = re.compile(r"S(\d{2})_D(\d)_A(\d{3})_T(\d{2})_A\.dat$")


def _read(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)


STANDARD = "x=-x,y=-y,z=z"  # usual mounting: sensor +x points up the arm
NORMAL_MIN = 0.85  # a subject is "normally mounted" if arm-down gravity is mostly on +x


def subject_rotations(arm_down: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-subject sensor -> Sanad rotation from each subject's arm-down gravity vector."""
    unit = {s: g / np.linalg.norm(g) for s, g in arm_down.items()}
    normal = [u for u in unit.values() if u[0] >= NORMAL_MIN]
    ref = np.median(normal, axis=0)
    ref /= np.linalg.norm(ref)
    std = axis_map_matrix(STANDARD)
    return {s: std @ rotation_between(u, ref) for s, u in unit.items()}


def load(raw_dir: Path, log=print) -> list[Recording]:
    files = sorted(p for p in raw_dir.rglob("*_A.dat") if (m := NAME.search(p.name)) and m[2] == WRIST)
    if not files:
        raise FileNotFoundError(
            f"no FallAllD wrist files (S??_D2_A???_T??_A.dat) under {raw_dir}; unzip the download there"
        )
    # per-subject forearm axis from walking (fallback: the first second of every ADL, mostly standing)
    by_subj: dict[str, list[np.ndarray]] = {}
    for f in files:
        subj, _, act, _ = NAME.search(f.name).groups()
        if int(act) in WALK_IDS:
            by_subj.setdefault(subj, []).append(_read(f).mean(0) * ACC_G_PER_COUNT)
    for f in files:
        subj, _, act, _ = NAME.search(f.name).groups()
        if subj not in by_subj and int(act) < FALL_MIN_ID:
            by_subj.setdefault(f"fallback:{subj}", []).append(_read(f)[:FS_IN].mean(0) * ACC_G_PER_COUNT)
    arm_down = {k.removeprefix("fallback:"): np.mean(v, axis=0) for k, v in by_subj.items()}
    axes = subject_rotations(arm_down)
    std = axis_map_matrix(STANDARD)
    for key in sorted(by_subj):
        subj = key.removeprefix("fallback:")
        turn = np.degrees(np.arccos(np.clip((np.trace(std.T @ axes[subj]) - 1) / 2, -1, 1)))
        log(f"[fallalld] S{subj}: arm-down gravity {np.round(arm_down[subj], 2)}, "
            f"corrected by {turn:.0f} deg -> Sanad {np.round(axes[subj] @ arm_down[subj], 2)}"
            + (" (from standing, no walking data)" if key.startswith("fallback:") else ""))

    recs: list[Recording] = []
    for f in files:
        subj, _, act, trial = NAME.search(f.name).groups()
        gyro_file = f.with_name(f.name[:-5] + "G.dat")
        if not gyro_file.exists() or subj not in axes:
            continue
        acc, gyro = _read(f), _read(gyro_file)
        n = min(len(acc), len(gyro))
        acc_h, gyro_h = harmonize(acc[:n] * ACC_G_PER_COUNT, gyro[:n] * GYRO_DPS_PER_COUNT, FS_IN,
                                  axes=axes[subj])
        is_fall = int(act) >= FALL_MIN_ID
        impact = None
        if is_fall:
            amag = magnitude(acc_h)
            lo = int((IMPACT_CENTRE_S - IMPACT_SEARCH_S) * FS)
            hi = min(int((IMPACT_CENTRE_S + IMPACT_SEARCH_S) * FS), len(amag))
            impact = (lo + int(np.argmax(amag[lo:hi]))) / FS
        recs.append(Recording(
            dataset="fallalld", subject_id=f"fallalld:S{subj}",
            recording_id=f"fallalld:S{subj}_A{act}_T{trial}", activity=f"A{act}",
            is_fall=is_fall, acc=acc_h, gyro=gyro_h, impact_t=impact, age_group="adult",
        ))
    log(f"[fallalld] {len(recs)} wrist recordings "
        f"({sum(r.is_fall for r in recs)} falls) from {len({r.subject_id for r in recs})} subjects")
    return recs
