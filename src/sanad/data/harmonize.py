"""Turn any dataset's recordings into Sanad's common format.

Common format = 50 Hz, acceleration in g (clipped to +-8 g like the deployed model sees it),
angular rate in deg/s, axes in the Sanad frame (see CLAUDE.md), left wrist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

G = 9.80665  # m/s^2 per g
FS = 50  # Hz, the band's sample rate
ACC_LIMIT_G = 8.0  # the model never sees more than +-8 g on any axis

# Mirroring a recording from one wrist to the other (same frame definition on both wrists):
# acceleration is a polar vector -> only y flips; angular rate is an axial vector -> x and z flip.
MIRROR_ACC = np.array([1.0, -1.0, 1.0])
MIRROR_GYRO = np.array([-1.0, 1.0, -1.0])

COLUMNS = [
    "dataset", "subject_id", "recording_id", "activity", "is_fall", "impact_t", "age_group",
    "t", "ax", "ay", "az", "gx", "gy", "gz",
]


@dataclass
class Recording:
    """One continuous recording, already harmonized (50 Hz, g, deg/s, Sanad frame, left wrist)."""

    dataset: str
    subject_id: str  # globally unique: "<dataset>:<subject>"
    recording_id: str  # globally unique
    activity: str  # dataset's own activity code, e.g. "F03" or "D11"
    is_fall: bool
    acc: np.ndarray  # (N, 3) g
    gyro: np.ndarray  # (N, 3) deg/s
    impact_t: float | None = None  # seconds from start, falls only
    age_group: str = "young"  # young | adult | elderly
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.acc = np.asarray(self.acc, dtype=np.float32)
        self.gyro = np.asarray(self.gyro, dtype=np.float32)
        if self.acc.shape != self.gyro.shape or self.acc.ndim != 2 or self.acc.shape[1] != 3:
            raise ValueError(f"{self.recording_id}: acc {self.acc.shape} vs gyro {self.gyro.shape}")
        if self.is_fall and self.impact_t is None:
            raise ValueError(f"{self.recording_id}: fall recording without impact_t")

    @property
    def n(self) -> int:
        return len(self.acc)

    @property
    def duration_s(self) -> float:
        return self.n / FS


# --- unit conversion -----------------------------------------------------------------------

def ms2_to_g(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) / G


def rads_to_dps(x: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.asarray(x, dtype=np.float64))


def counts_to_units(x: np.ndarray, units_per_count: float) -> np.ndarray:
    """Raw sensor counts -> physical units, e.g. LSM9DS1 at +-8 g: 0.244e-3 g per count."""
    return np.asarray(x, dtype=np.float64) * units_per_count


# --- resampling ----------------------------------------------------------------------------

def resample_to_fs(x: np.ndarray, fs_in: float, fs_out: int = FS) -> np.ndarray:
    """Polyphase resampling with an anti-aliasing filter along axis 0.

    238 -> 50 Hz uses up=25, down=119 exactly; 100 -> 50 Hz uses up=1, down=2.
    """
    x = np.asarray(x, dtype=np.float64)
    ratio = Fraction(fs_out) / Fraction(fs_in).limit_denominator(1000)
    if ratio == 1:
        return x.copy()
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=0, padtype="line")


# --- axes ----------------------------------------------------------------------------------

def axis_map_matrix(spec: str) -> np.ndarray:
    """Build a 3x3 signed permutation from a spec like "x=-y,y=x,z=z".

    Each entry says which source axis (with sign) becomes that Sanad axis. Must be a proper
    rotation (det = +1); a reflection means the source is the other wrist, use `mirror` instead.
    """
    idx = {"x": 0, "y": 1, "z": 2}
    m = np.zeros((3, 3))
    parts = dict(p.split("=") for p in spec.replace(" ", "").split(","))
    if sorted(parts) != ["x", "y", "z"]:
        raise ValueError(f"axis spec needs x, y and z: {spec!r}")
    for dst, src in parts.items():
        sign = -1.0 if src.startswith("-") else 1.0
        m[idx[dst], idx[src.lstrip("+-")]] = sign
    if not np.allclose(np.abs(m).sum(axis=0), 1) or round(np.linalg.det(m)) != 1:
        raise ValueError(f"axis spec is not a proper rotation: {spec!r}")
    return m


def apply_axes(xyz: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.asarray(xyz) @ m.T


def mirror(acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right wrist <-> left wrist."""
    return np.asarray(acc) * MIRROR_ACC, np.asarray(gyro) * MIRROR_GYRO


def clip_acc(acc: np.ndarray, limit: float = ACC_LIMIT_G) -> np.ndarray:
    return np.clip(acc, -limit, limit)


# --- one-call harmonization ----------------------------------------------------------------

def harmonize(
    acc_g: np.ndarray,
    gyro_dps: np.ndarray,
    fs_in: float,
    axes: str = "x=x,y=y,z=z",
    right_wrist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Already-in-units arrays -> 50 Hz, Sanad frame, left wrist, acceleration clipped."""
    m = axis_map_matrix(axes)
    acc = apply_axes(resample_to_fs(acc_g, fs_in), m)
    gyro = apply_axes(resample_to_fs(gyro_dps, fs_in), m)
    if right_wrist:
        acc, gyro = mirror(acc, gyro)
    return clip_acc(acc).astype(np.float32), gyro.astype(np.float32)


# --- storage -------------------------------------------------------------------------------

def recordings_to_frame(recs: list[Recording]) -> pd.DataFrame:
    frames = []
    for r in recs:
        frames.append(pd.DataFrame({
            "dataset": r.dataset, "subject_id": r.subject_id, "recording_id": r.recording_id,
            "activity": r.activity, "is_fall": r.is_fall,
            "impact_t": np.nan if r.impact_t is None else r.impact_t, "age_group": r.age_group,
            "t": np.arange(r.n, dtype=np.float32) / FS,
            "ax": r.acc[:, 0], "ay": r.acc[:, 1], "az": r.acc[:, 2],
            "gx": r.gyro[:, 0], "gy": r.gyro[:, 1], "gz": r.gyro[:, 2],
        }))
    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    for c in ("dataset", "subject_id", "recording_id", "activity", "age_group"):
        df[c] = df[c].astype("category")
    return df


def frame_to_recordings(df: pd.DataFrame) -> list[Recording]:
    recs = []
    for rid, g in df.groupby("recording_id", observed=True, sort=False):
        first = g.iloc[0]
        impact = None if pd.isna(first.impact_t) else float(first.impact_t)
        recs.append(Recording(
            dataset=str(first.dataset), subject_id=str(first.subject_id), recording_id=str(rid),
            activity=str(first.activity), is_fall=bool(first.is_fall), impact_t=impact,
            age_group=str(first.age_group),
            acc=g[["ax", "ay", "az"]].to_numpy(), gyro=g[["gx", "gy", "gz"]].to_numpy(),
        ))
    return recs
