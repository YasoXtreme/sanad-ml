import numpy as np
import pytest

from sanad.data.harmonize import (
    FS,
    MIRROR_ACC,
    MIRROR_GYRO,
    Recording,
    axis_map_matrix,
    clip_acc,
    frame_to_recordings,
    harmonize,
    ms2_to_g,
    rads_to_dps,
    recordings_to_frame,
    resample_to_fs,
)


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def test_units():
    assert ms2_to_g(9.80665) == pytest.approx(1.0)
    assert rads_to_dps(np.pi) == pytest.approx(180.0)


@pytest.mark.parametrize("fs_in", [238, 100, 50])
def test_resample_length_and_slow_signal(fs_in):
    t = np.arange(20 * fs_in) / fs_in
    x = np.sin(2 * np.pi * 2.0 * t)[:, None]  # 2 Hz is well below the 25 Hz Nyquist limit
    y = resample_to_fs(x, fs_in)
    assert len(y) == 20 * FS
    t50 = np.arange(len(y)) / FS
    core = slice(FS, -FS)  # ignore filter edges
    assert np.max(np.abs(y[core, 0] - np.sin(2 * np.pi * 2.0 * t50[core]))) < 0.02


def test_resample_keeps_most_of_an_impact_peak():
    fs = 238
    x = np.ones((20 * fs, 1))
    i = 10 * fs
    x[i - 6 : i + 7, 0] += 5.0 * np.hanning(13)  # ~50 ms, 6 g impact
    y = resample_to_fs(x, fs)
    assert y.max() > 4.0  # the anti-alias filter smooths it but the peak stays clearly above 3 g


def test_axis_map_is_a_proper_rotation():
    m = axis_map_matrix("x=-y,y=x,z=z")
    assert np.allclose(m @ np.array([1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0])
    with pytest.raises(ValueError):
        axis_map_matrix("x=y,y=x,z=z")  # a reflection: that's the other wrist, not a rotation


def test_mirror_matches_physics():
    # Reflection S = diag(MIRROR_ACC) maps a rotation R(t) to S R S. Its angular velocity must be
    # det(S) * S * w, which is exactly MIRROR_GYRO * w.
    s = np.diag(MIRROR_ACC)
    assert round(np.linalg.det(s)) == -1
    w = np.array([0.3, -1.2, 2.5])
    assert np.allclose(s @ skew(w) @ s, skew(MIRROR_GYRO * w))


def test_harmonize_clips_and_mirrors():
    acc = np.tile([[0.0, 9.0, 1.0]], (100, 1))
    gyro = np.tile([[10.0, 20.0, 30.0]], (100, 1))
    a, g = harmonize(acc, gyro, fs_in=50, right_wrist=True)
    assert np.allclose(a[0], [0.0, -8.0, 1.0])
    assert np.allclose(g[0], [-10.0, 20.0, -30.0])
    assert np.abs(clip_acc(np.array([[20.0, -20.0, 0.0]]))).max() == 8.0


def test_frame_round_trip():
    n = 150
    recs = [
        Recording("demo", "demo:s1", "demo:r1", "F01", True, np.zeros((n, 3)), np.ones((n, 3)),
                  impact_t=1.5),
        Recording("demo", "demo:s2", "demo:r2", "D01", False, np.ones((n, 3)), np.zeros((n, 3))),
    ]
    back = frame_to_recordings(recordings_to_frame(recs))
    assert [r.recording_id for r in back] == ["demo:r1", "demo:r2"]
    assert back[0].impact_t == 1.5 and back[1].impact_t is None
    assert np.allclose(back[0].gyro, 1.0) and np.allclose(back[1].acc, 1.0)


def test_fall_needs_impact_time():
    with pytest.raises(ValueError):
        Recording("demo", "demo:s1", "demo:r1", "F01", True, np.zeros((10, 3)), np.zeros((10, 3)))
