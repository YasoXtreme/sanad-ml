"""FallAllD (IEEE DataPort, doi 10.21227/bnya-mn34): wrist logger only. Role: main falls + ADLs.

Format (from the authors' FallAllD_2_PYTHON_Structure.py / Plot_FallAllD_Record.m):
- one file per sensor: S<subject:2>_D<device:1>_A<activity:3>_T<trial:2>_<A|G|M|B>.dat
  device 1 = neck, 2 = wrist, 3 = waist. Comma-separated int16 counts, 3 columns.
- 238 Hz, 20 s per recording; acc 0.000244 g/count (+-8 g), gyro 0.07 deg/s per count (+-2000 dps).
- Activity IDs >= 101 are falls (checked against the activity list PDF when the data arrived).
- No impact labels: the impact is the |a| peak within +-2 s of the recording centre (t = 10 s).
We read the plain .dat text files, never the .pkl files (unpickling untrusted files can run code).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from sanad.data.harmonize import FS, Recording, harmonize
from sanad.data.windowing import magnitude

FS_IN = 238
ACC_G_PER_COUNT = 0.000244
GYRO_DPS_PER_COUNT = 0.07
FALL_MIN_ID = 101
IMPACT_CENTRE_S, IMPACT_SEARCH_S = 10.0, 2.0
WRIST = "2"
# Sensor axes -> Sanad frame, and which wrist the logger was on. Set from the gravity check in
# scripts (see CLAUDE.md "Data"); identity until verified.
AXES = "x=x,y=y,z=z"
RIGHT_WRIST = False
NAME = re.compile(r"S(\d{2})_D(\d)_A(\d{3})_T(\d{2})_A\.dat$")


def _read(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)


def load(raw_dir: Path, log=print, axes: str = AXES, right_wrist: bool = RIGHT_WRIST) -> list[Recording]:
    files = sorted(p for p in raw_dir.rglob("*_A.dat") if (m := NAME.search(p.name)) and m[2] == WRIST)
    if not files:
        raise FileNotFoundError(
            f"no FallAllD wrist files (S??_D2_A???_T??_A.dat) under {raw_dir}; unzip the download there"
        )
    recs: list[Recording] = []
    for f in files:
        subj, _, act, trial = NAME.search(f.name).groups()
        gyro_file = f.with_name(f.name[:-5] + "G.dat")
        if not gyro_file.exists():
            continue
        acc, gyro = _read(f), _read(gyro_file)
        n = min(len(acc), len(gyro))
        acc_h, gyro_h = harmonize(acc[:n] * ACC_G_PER_COUNT, gyro[:n] * GYRO_DPS_PER_COUNT, FS_IN,
                                  axes=axes, right_wrist=right_wrist)
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
