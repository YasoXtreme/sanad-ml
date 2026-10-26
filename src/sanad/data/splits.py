"""Person-based splits: a subject's windows are never in both train and test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def subject_folds(meta: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold number (0..n_splits-1) per window, grouped by subject and stratified by label.

    Windows labelled -1 (ignore) still get a fold so they follow their subject.
    """
    groups = meta["subject_id"].astype(str).to_numpy()
    strat = np.where(y == 1, 1, 0)
    fold = np.full(len(y), -1, dtype=np.int8)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), strat, groups)):
        fold[test_idx] = k
    assert (fold >= 0).all()
    return fold


def assert_no_leakage(meta: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> None:
    s = meta["subject_id"].astype(str)
    shared = set(s[train_mask]) & set(s[test_mask])
    if shared:
        raise AssertionError(f"subjects in both train and test: {sorted(shared)[:10]}")
