import numpy as np
import pytest

from sanad.data.harmonize import FS, Recording
from sanad.data.windowing import (
    POST,
    PRE,
    WINDOW,
    choose_threshold,
    extract_windows,
    find_triggers,
    magnitude,
)


def rec(rid, spikes, is_fall=False, impact_t=None, seconds=20):
    acc = np.tile([[0.0, 0.0, 1.0]], (seconds * FS, 1))  # still, gravity along z
    for t, g in spikes:
        acc[int(t * FS), 2] = g
    return Recording("demo", f"demo:{rid}", f"demo:{rid}", "F01" if is_fall else "D01", is_fall,
                     acc, np.zeros_like(acc), impact_t=impact_t)


def test_window_shape_and_peak_position():
    w = extract_windows([rec("a", [(10.0, 4.0)], is_fall=True, impact_t=10.0)], threshold=2.5)
    assert w.x.shape == (1, WINDOW, 6) and WINDOW == PRE + POST == 200
    assert w.y.tolist() == [1]
    assert np.argmax(magnitude(w.x[0, :, :3])) == PRE
    assert w.falls.detected.tolist() == [True]


def test_adl_trigger_is_negative():
    w = extract_windows([rec("b", [(5.0, 4.0)])], threshold=2.5)
    assert w.y.tolist() == [0] and w.falls.empty


def test_missed_fall_is_kept():
    # spike far from the labelled impact -> negative window, and the fall counts as a miss
    w = extract_windows([rec("c", [(3.0, 4.0)], is_fall=True, impact_t=15.0)], threshold=2.5)
    assert w.y.tolist() == [0]
    assert w.falls.detected.tolist() == [False]


def test_near_impact_is_ignored():
    w = extract_windows([rec("d", [(12.5, 4.0)], is_fall=True, impact_t=10.0)], threshold=2.5)
    assert w.y.tolist() == [-1]


def test_refractory_period():
    amag = np.ones(20 * FS)
    amag[[500, 540]] = 4.0  # 0.8 s apart: second spike falls inside the disarmed 1.5 s
    assert find_triggers(amag, 2.5) == [500]
    amag = np.ones(20 * FS)
    amag[[500, 600]] = 4.0  # 2 s apart: two triggers
    assert find_triggers(amag, 2.5) == [500, 600]


def test_trigger_picks_highest_peak_in_search_span():
    amag = np.ones(1000)
    amag[[400, 410]] = [3.0, 6.0]
    assert find_triggers(amag, 2.5) == [410]


def test_threshold_lets_99_percent_of_falls_through():
    recs = [rec(f"f{i}", [(10.0, g)], is_fall=True, impact_t=10.0)
            for i, g in enumerate(np.linspace(2.0, 6.0, 200))]
    thr = choose_threshold(recs)
    peaks = np.array([magnitude(r.acc).max() for r in recs])
    assert (peaks >= thr).mean() >= 0.99 - 1e-9
    assert thr == pytest.approx(np.quantile(peaks, 0.01))
