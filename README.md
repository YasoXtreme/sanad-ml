# sanad-ml

Machine-learning side of **Sanad (سند)**, a wrist band that detects falls for elderly people living alone.
G12 capstone project by Fares Hassan, Omar Ahmed and Yassen Mohamed.

The band runs the fall model **on the wrist** (Raspberry Pi Pico 2 W). This repo trains it:

1. **Public data → one format.** Wrist fall/daily-activity datasets are converted to 50 Hz, g and °/s, ±8 g, in one shared axis frame.
2. **Trigger windows.** The same |a| trigger the band uses cuts 4 s windows (200 × 7) around each impact-like peak.
3. **Teacher.** A large 1D CNN trained on the GPU on all usable public data.
4. **Student "Sanad-Tiny".** A ≈12k-parameter CNN distilled from the teacher, then exported as a full-int8 TFLite model + C array for TFLite Micro.
5. **Later:** retrain with data recorded by the real band.

## Setup (Windows)

```powershell
winget install astral-sh.uv      # once
uv sync                          # Python 3.12 venv + all dependencies (PyTorch CUDA 13.0, TF CPU)
uv run python scripts/check_env.py
```

`check_env.py` must report all PASS (GPU visible, int8 TFLite export works, DVC store reachable) before training.

## Data and models

Datasets, processed windows and trained models are **not** in git. They're versioned with [DVC](https://dvc.org)
and backed up to `D:\dvc-store\sanad-ml`.

```powershell
uv run dvc pull     # get data/models from the store
uv run dvc push     # back up new data/models
```

The dataset registry (sources, licenses, how to download) is `data/manifest.yaml`.

## Layout

| Path | What |
|---|---|
| `src/sanad/` | library code: data loaders, harmonization, windowing, models, training, export |
| `scripts/` | command-line entry points (`check_env.py`, …) |
| `configs/` | YAML settings for data, teacher, student, distillation |
| `data/` | `raw/` downloads, `interim/` harmonized parquet, `processed/` windows (all DVC) |
| `models/` | checkpoints and exported `.tflite` / C files (DVC) |
| `reports/` | result tables and figures |
| `tests/` | unit tests (`uv run pytest`) |
