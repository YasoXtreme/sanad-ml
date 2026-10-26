"""Raw downloads -> harmonized 50 Hz parquet, one file per dataset, in data/interim/.

    uv run python -I -m sanad.data.prepare weda_fall pamap2 fallalld
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from sanad.data.harmonize import FS, frame_to_recordings, recordings_to_frame
from sanad.data.loaders import get_loader
from sanad.paths import INTERIM, RAW


def summary(df: pd.DataFrame) -> pd.Series:
    rec = df.groupby("recording_id", observed=True).agg(
        subject=("subject_id", "first"), fall=("is_fall", "first"), age=("age_group", "first"),
        n=("t", "size"),
    )
    return pd.Series({
        "recordings": len(rec),
        "subjects": rec.subject.nunique(),
        "fall_recordings": int(rec.fall.sum()),
        "adl_recordings": int((~rec.fall).sum()),
        "hours": round(rec.n.sum() / FS / 3600, 2),
        "elderly_subjects": rec[rec.age == "elderly"].subject.nunique(),
    })


def prepare(ds_id: str) -> pd.DataFrame:
    recs = get_loader(ds_id)(RAW / ds_id)
    df = recordings_to_frame(recs)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"{ds_id}.parquet"
    df.to_parquet(out, index=False)
    print(f"[{ds_id}] wrote {out.relative_to(INTERIM.parent.parent)}")
    return df


def load_interim(ds_id: str):
    return frame_to_recordings(pd.read_parquet(INTERIM / f"{ds_id}.parquet"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    args = ap.parse_args(argv)
    rows = {ds: summary(prepare(ds)) for ds in args.ids}
    print(pd.DataFrame(rows).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
