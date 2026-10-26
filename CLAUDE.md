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
1. Trigger: |a| (g) crosses a threshold chosen so ≥ 99 % of training falls trigger. The peak is the
   highest |a| in the next 0.5 s; a strictly higher |a| before the window completes moves the peak
   there (a stumble can't hide the impact). Exact rule: `src/sanad/data/windowing.py` docstring.
2. Window: 2.5 s before the peak to 1.5 s after = 200 samples × 7 channels `ax, ay, az, gx, gy, gz, |a|`,
   normalized with training-set per-channel mean/std (values in `models/firmware/sanad_model_config.h`).
3. Student CNN "Sanad-Tiny", 45 k version (configs/deploy.yaml; 45,153 params, 2.57 M MACs, 59 KB int8):
   Conv(32,k7)-BN-ReLU-MaxPool2 → Conv(64,k5)-BN-ReLU-MaxPool2 → Conv(128,k3)-BN-ReLU-GAP → Dense64-ReLU-Dropout0.3 → Dense1,
   distilled from the small transformer teacher (won the student-size tournament, reports/student_size.md).
4. Decision threshold calibrated on pooled out-of-fold predictions to catch 95 % of falls end to end
   (sanad.train.train_final; currently 0.282).
5. **Open problem: false alarms.** Trigger + model alone ≈ 120 false alarms per normal day (estimate,
   reports/false_alarms.md; target ≤ 2/24 h). The firmware must not ring on the model alone: require
   post-fall stillness (provisional: next 5 s with |a| std ≤ 0.30 g and mean |gyro| ≤ 60 °/s, removes
   ~2/3 of false alarms, keeps ~96 % of falls), start with a quiet vibration, and log every trigger so
   the band's own recordings become training data.

## Conventions
- **Units**: acceleration in g, angular rate in °/s, time in s, 50 Hz everywhere after harmonization.
- **Sanad frame** (band on the back of the wrist): +x along the forearm toward the hand, +z out of the band
  face, +y = z × x (right-handed). On the left wrist +y points to the little-finger side; on the right
  wrist, to the thumb side. Every dataset loader maps into this frame; the firmware rotates the MPU-6050
  axes into it too.
- **Mirroring** left ↔ right wrist (same frame definition on both): flip y for acceleration; flip x and z
  for gyroscope (axial vector).
- **Roll around the forearm is randomized in training** (`augment.roll_deg: 180`): FallAllD's loggers
  were mounted inconsistently and a real band slides around the wrist. So only the forearm axis (+x,
  toward the hand) must be right on the band; the y/z roll doesn't matter.
- **Splits are by person** (`subject_id` prefixed with the dataset id). Never split windows randomly.
- Falls that never trigger are counted as misses, never dropped.

## Commands
- `uv sync`: install (Python 3.12, torch cu130, TF 2.21 CPU for TFLite conversion).
- `uv run python scripts/check_env.py`: environment check, all PASS required.
- `uv run pytest`, `uv run ruff check .`
- `uv run dvc pull` / `uv run dvc push`: data + models, store at `D:\dvc-store\sanad-ml`.
- `uv run python -I -m sanad.data.download <ids>` → `uv run dvc repro` (harmonize all + train final + export).
- `uv run python -I -m sanad.train.run_cv --name <run>` → `uv run python -I -m sanad.eval.report runs/<run>`.
- `uv run python -I -m sanad.train.tournament --name <run>` → `uv run python -I -m sanad.eval.tournament_report runs/<run>`
  (teacher + distillation tournament, configs/tournament.yaml; results in reports/tournament.md).
- Oxford SSL-wearables weights (`data/raw/oxford_ssl`, DVC only) are under Oxford's Academic Use Licence:
  never commit them, credit "Copyright © 2022, University of Oxford", no commercial use.
- Training needs the GPU: the laptop powers the RTX 5060 off on battery (plug in the charger).
  Scripts that read downloaded data run with `python -I`.

## Data
Registry: `data/manifest.yaml`. Main sets: FallAllD (wrist, 238 Hz; manual download, IEEE login),
WEDA-FALL (wrist, 50 Hz, elderly ADLs), PAMAP2 (wrist ADLs, 100 Hz). 18–25 Hz sets (UMAFall, Erciyes,
UP-Fall, Nandi) are external test sets only. Raw data never goes into git (repo is public; licenses vary).
