# Sanad ML: project context

Sanad (سند, "someone to lean on") is a STEM Egypt G12 capstone (Fares Hassan, Omar Ahmed, Yassen Mohamed):
a wrist band for elderly people living alone that detects falls **on the band**, runs a 30 s "I'm OK"
countdown (vibration + rising beep; single press cancels, 3 s hold = manual SOS), then alerts family via
the cloud (Express + MongoDB, Telegram bot). Design history is in the claude.ai project "Sanad".

## Hardware the model must fit
- **Raspberry Pi Pico 2 W** (RP2350, Cortex-M33 + FPU, 520 KB SRAM, 4 MB flash). ESP boards are banned by the school.
- Firmware in C/C++ (Pico SDK or Arduino-Pico), model via TFLite Micro, int8.
- **MPU-6050** at 50 Hz: record ±16 g / ±2000 °/s, the model sees acceleration clipped to ±8 g.
- MAX30102 is only for wear detection + heart-rate snapshot during the countdown. No ML for it.

## On-band pipeline (the training pipeline must mirror it exactly)
1. Trigger: |a| (g) crosses a threshold chosen so ≥ 99 % of training falls trigger.
2. Window: 2.5 s before the peak to 1.5 s after = 200 samples × 7 channels `ax, ay, az, gx, gy, gz, |a|`,
   normalized with training-set per-channel mean/std.
3. Student CNN "Sanad-Tiny" (≈11.7k params, ≈0.7 M MACs):
   Conv(16,k7)-BN-ReLU-MaxPool2 → Conv(32,k5)-BN-ReLU-MaxPool2 → Conv(64,k3)-BN-ReLU-GAP → Dense32-ReLU-Dropout0.3 → Dense1.
4. Decision threshold picked on validation for ≥ 95 % sensitivity.

## Conventions
- **Units**: acceleration in g, angular rate in °/s, time in s, 50 Hz everywhere after harmonization.
- **Sanad frame** (band on the back of the wrist): +x along the forearm toward the hand, +z out of the band
  face, +y = z × x (right-handed). On the left wrist +y points to the little-finger side; on the right
  wrist, to the thumb side. Every dataset loader maps into this frame; the firmware rotates the MPU-6050
  axes into it too.
- **Mirroring** left ↔ right wrist (same frame definition on both): flip y for acceleration; flip x and z
  for gyroscope (axial vector).
- **Splits are by person** (`subject_id` prefixed with the dataset id). Never split windows randomly.
- Falls that never trigger are counted as misses, never dropped.

## Commands
- `uv sync`: install (Python 3.12, torch cu130, TF 2.21 CPU for TFLite conversion).
- `uv run python scripts/check_env.py`: environment check, all PASS required.
- `uv run pytest`, `uv run ruff check .`
- `uv run dvc pull` / `uv run dvc push`: data + models, store at `D:\dvc-store\sanad-ml`.

## Data
Registry: `data/manifest.yaml`. Main sets: FallAllD (wrist, 238 Hz; manual download, IEEE login),
WEDA-FALL (wrist, 50 Hz, elderly ADLs), PAMAP2 (wrist ADLs, 100 Hz). 18–25 Hz sets (UMAFall, Erciyes,
UP-Fall, Nandi) are external test sets only. Raw data never goes into git (repo is public; licenses vary).
