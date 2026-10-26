import numpy as np

from sanad.data.harmonize import rotation_between
from sanad.data.loaders.fallalld import subject_rotations


def test_rotation_between():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        r = rotation_between(a, b)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(r), 1)
        assert np.allclose(r @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
    r = rotation_between([1, 0, 0], [-1, 0, 0])  # opposite directions
    assert np.allclose(r @ [1, 0, 0], [-1, 0, 0]) and np.isclose(np.linalg.det(r), 1)


def test_fallalld_mountings_all_end_up_arm_down():
    arm_down = {
        "02": np.array([0.94, -0.17, -0.30]),  # usual mounting
        "09": np.array([1.02, -0.19, -0.19]),
        "10": np.array([1.00, -0.13, -0.17]),
        "04": np.array([-1.00, 0.10, -0.16]),  # logger reversed
        "13": np.array([0.14, -0.94, 0.07]),  # turned 90 deg
        "01": np.array([0.71, 0.70, 0.03]),  # ~45 deg
    }
    rots = subject_rotations(arm_down)
    mapped = {s: rots[s] @ (g / np.linalg.norm(g)) for s, g in arm_down.items()}
    ref = mapped["09"]
    for s, v in mapped.items():
        assert np.isclose(np.linalg.det(rots[s]), 1)
        assert v[0] < -0.9, s  # arm down: gravity reads "up" = -x in the Sanad frame
        assert np.degrees(np.arccos(np.clip(v @ ref, -1, 1))) < 15, s  # all in the same place
