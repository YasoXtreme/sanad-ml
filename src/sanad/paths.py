"""Project paths. Everything is relative to the repo root so scripts work from any folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"  # downloads, untouched (DVC)
INTERIM = DATA / "interim"  # one harmonized 50 Hz parquet per dataset (DVC)
PROCESSED = DATA / "processed"  # trigger windows + labels + splits (DVC)
MANIFEST = DATA / "manifest.yaml"  # dataset registry (git)

CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"  # checkpoints and exported .tflite / C arrays (DVC)
RUNS = ROOT / "runs"  # TensorBoard logs (ignored)
REPORTS = ROOT / "reports"
