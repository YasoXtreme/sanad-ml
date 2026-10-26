"""Download the datasets marked `access: auto` in data/manifest.yaml into data/raw/<id>/.

    uv run python -I -m sanad.data.download weda_fall pamap2
    uv run python -I -m sanad.data.download --tier 1 2      # everything automatic in those tiers

- Resumes interrupted downloads (HTTP Range), so a slow or dropped connection doesn't restart from 0.
- Verifies sha256/md5 when the manifest has them, and prints the sha256 of new files to record there.
- Unpacks .zip archives into data/raw/<id>/ with a path-traversal check (downloads are untrusted).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from sanad.paths import MANIFEST, RAW

CHUNK = 1 << 20


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["datasets"]


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, attempts: int = 6) -> Path:
    """Download with retries; resumes when the server supports HTTP Range, else starts over."""
    for i in range(1, attempts + 1):
        try:
            return _fetch_once(url, dest)
        except (requests.ConnectionError, requests.Timeout) as exc:
            print(f"  {dest.name}: attempt {i}/{attempts} failed ({type(exc).__name__}), retrying")
    raise RuntimeError(f"{dest.name}: download failed after {attempts} attempts")


def _fetch_once(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "sanad-ml dataset downloader"}
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        if done and r.status_code != 206:  # server ignored Range: start over
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done or None
        mode = "ab" if done else "wb"
        with part.open(mode) as f, tqdm(total=total, initial=done, unit="B", unit_scale=True,
                                        desc=dest.name, mininterval=5, file=sys.stdout) as bar:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest


def safe_unzip(archive: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (out_dir / member.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"{archive.name}: unsafe path in archive: {member.filename}")
        z.extractall(out_dir)


def download_dataset(ds_id: str, spec: dict) -> None:
    if spec.get("access") != "auto":
        print(f"[{ds_id}] manual download: see {spec.get('url')} -> put files in {RAW / ds_id}")
        return
    out = RAW / ds_id
    out.mkdir(parents=True, exist_ok=True)
    for f in spec.get("files", []):
        if not f.get("url"):
            print(f"[{ds_id}] {f['name']}: no direct URL in the manifest yet, skipped")
            continue
        path = fetch(f["url"], out / f["name"])
        for algo in ("sha256", "md5"):
            want = f.get(algo)
            got = file_hash(path, algo) if (want or algo == "sha256") else None
            if want and got != want:
                raise RuntimeError(f"[{ds_id}] {path.name}: {algo} mismatch ({got} != {want})")
            if algo == "sha256" and not want:
                print(f"[{ds_id}] {path.name} sha256 {got}  <- record this in data/manifest.yaml")
        if path.suffix == ".zip":
            marker = out / f".unzipped_{path.stem}"
            if not marker.exists():
                safe_unzip(path, out)
                marker.touch()
        print(f"[{ds_id}] {path.name} ready")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ids", nargs="*", help="dataset ids from data/manifest.yaml")
    ap.add_argument("--tier", nargs="*", type=int, help="all datasets in these tiers")
    args = ap.parse_args(argv)
    manifest = load_manifest()
    ids = list(args.ids)
    if args.tier:
        ids += [k for k, v in manifest.items() if v.get("tier") in args.tier and k not in ids]
    unknown = [i for i in ids if i not in manifest]
    if unknown or not ids:
        ap.error(f"unknown or no dataset ids: {unknown}; known: {sorted(manifest)}")
    for ds_id in ids:
        download_dataset(ds_id, manifest[ds_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
