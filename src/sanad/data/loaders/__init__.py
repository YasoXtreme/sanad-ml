"""One loader per dataset: raw files in data/raw/<id>/ -> list of harmonized Recordings."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sanad.data.harmonize import Recording


def get_loader(ds_id: str) -> Callable[[Path], list[Recording]]:
    if ds_id == "weda_fall":
        from sanad.data.loaders.weda_fall import load
    elif ds_id == "pamap2":
        from sanad.data.loaders.pamap2 import load
    elif ds_id == "fallalld":
        from sanad.data.loaders.fallalld import load
    else:
        raise KeyError(f"no loader for dataset {ds_id!r}")
    return load
