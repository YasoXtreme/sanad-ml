# Sanad band: recording format for training data

When the band is built, the firmware (or a logging build of it) records sessions in this format.
The `sanad_device` loader reads exactly this, so the band's own data can be used to fine-tune the model.
Recordings can't be re-downloaded, so they go into DVC (`uv run dvc add` + `dvc push`) as soon as they're copied off the band.

## Sensor settings (must match the deployed firmware)
| Setting | Value |
|---|---|
| IMU | MPU-6050 |
| Sample rate | 50 Hz (1 kHz / (1 + SMPLRT_DIV 19)), read from the FIFO |
| Digital low-pass | DLPF_CFG = 4 (≈20 Hz) |
| Accelerometer range | ±16 g → 2048 counts per g |
| Gyroscope range | ±2000 °/s → 16.4 counts per °/s |

## Folder layout
```
data/raw/sanad_device/
  sessions.csv                      one row per session (metadata below)
  <session_id>/imu.csv              raw samples
  <session_id>/events.csv           labels and band events
```
`session_id` = `<wearer_id>_<YYYYMMDD-HHMMSS>`, e.g. `W03_20261201-171500`.

## `sessions.csv`
| Column | Example | Notes |
|---|---|---|
| `session_id` | `W03_20261201-171500` | |
| `wearer_id` | `W03` | anonymous code; names stay off the repo |
| `age_group` | `young` / `adult` / `elderly` | elderly = 65+ |
| `wrist` | `L` / `R` | |
| `firmware` | `0.3.1` | |
| `mount` | `x=-y,y=x,z=z` | how chip axes map into the Sanad frame (see `CLAUDE.md`) |
| `kind` | `scripted` / `free_living` | |
| `notes` | `mat falls, supervised` | |

## `imu.csv`
Raw chip-axis counts, **not** rotated or scaled, so mistakes in `mount` can be fixed later.
| Column | Type | Notes |
|---|---|---|
| `sample` | uint32 | FIFO sample counter from session start; gaps mean dropped samples |
| `t_ms` | uint32 | ms since session start |
| `ax, ay, az` | int16 | accelerometer counts |
| `gx, gy, gz` | int16 | gyroscope counts |

## `events.csv`
| Column | Notes |
|---|---|
| `t_ms` | time of the event, same clock as `imu.csv` |
| `event` | one of the codes below |
| `value` | activity code, probability, etc. |

Band events (written by firmware): `trigger` (value = peak |a| in g), `model` (value = fall probability),
`countdown_start`, `button_ok`, `button_sos`, `alert_sent`, `wear_on`, `wear_off`.

Labels (written by the person running the session, e.g. from a phone timer app synced at session start):
`activity_start` / `activity_end` (value = activity code), `fall_impact` (value = fall code, approximate time;
the loader snaps it to the nearest |a| peak within ±1 s).

## Activity codes
| Code | Activity | Why |
|---|---|---|
| F01–F08 | mat falls, same types as WEDA-FALL F01–F08 | supervised, onto thick gym mats, teacher-approved |
| D01–D11 | daily activities, same as WEDA-FALL D01–D11 | comparable with public data |
| P01 | rukū' (bowing in prayer) | not in any public dataset |
| P02 | sujūd (prostration) | goes down to the floor; easy to confuse with a fall |
| P03 | sitting down onto a floor cushion | |
| P04 | getting up from the floor | |
| H01 | flopping onto a bed | hard negative |
| H02 | sitting down hard on a chair | hard negative |
| H03 | clapping | hard negative |
| H04 | banging a hand on a table | hard negative |
| X00 | free living, unlabeled | false-alarms-per-day estimate |
