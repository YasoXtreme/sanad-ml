"""Build a teacher from a config dict (shared by the tournaments and final training)."""

from __future__ import annotations

import numpy as np

from sanad.models.oxford import OxfordTeacher
from sanad.models.teacher import SanadTeacher
from sanad.models.transformer import SanadTransformer
from sanad.paths import RAW

OXFORD_WEIGHTS = RAW / "oxford_ssl" / "mtl_5_best.mdl"


def build_teacher(spec: dict, mean: np.ndarray, std: np.ndarray):
    kind = spec["kind"]
    if kind == "resnet":
        return SanadTeacher()
    if kind == "transformer":
        return SanadTransformer(d_model=spec["d_model"], heads=spec["heads"], layers=spec["layers"],
                                ff=spec["ff"])
    if kind == "oxford":
        return OxfordTeacher(mean, std, pretrained=OXFORD_WEIGHTS if spec["pretrained"] else None)
    raise ValueError(f"unknown teacher kind {kind!r}")
