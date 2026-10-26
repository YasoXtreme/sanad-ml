import numpy as np
import pandas as pd
import pytest

from sanad.eval.metrics import event_metrics, threshold_for_event_sensitivity


def make(scores, missed=0):
    """One fall window per caught-by-trigger fall (with its score), plus `missed` falls with no window."""
    n = len(scores)
    meta = pd.DataFrame({"recording_id": [f"f{i}" for i in range(n)], "label": 1})
    recs = pd.DataFrame({"recording_id": [f"f{i}" for i in range(n + missed)], "is_fall": True,
                         "detected": [True] * n + [False] * missed, "duration_s": 20.0})
    return meta, recs, np.asarray(scores, dtype=float)


def test_threshold_hits_event_target():
    meta, recs, p = make(np.linspace(0.05, 0.99, 100))
    thr = threshold_for_event_sensitivity(meta, recs, p, 0.95)
    assert event_metrics(meta, recs, p, thr)["event_sensitivity"] >= 0.95
    assert event_metrics(meta, recs, p, thr + 1e-6)["event_sensitivity"] < 0.95


def test_trigger_misses_count_against_the_target():
    meta, recs, p = make(np.linspace(0.05, 0.99, 96), missed=4)  # 96% caught by the trigger at most
    thr = threshold_for_event_sensitivity(meta, recs, p, 0.95)
    assert event_metrics(meta, recs, p, thr)["event_sensitivity"] == pytest.approx(0.95)
    unreachable = threshold_for_event_sensitivity(meta, recs, p, 0.99)
    assert unreachable == pytest.approx(0.05)  # best possible: every window the trigger produced
