# Sanad firmware v0.1: sensor wiring test

Reads the MPU-6050 and the MAX30102 and prints their values over USB. Nothing else: no model, no Wi-Fi.

## 1. Wiring

Unplug the Pico from USB while you wire. Both sensors share **one I2C bus**: all four wires are shared.

Pico pin numbers count from **pin 1 = top-left when the USB port points up** (look for the small "1" or
square pad), down the left side to pin 20, then from pin 21 at the bottom-right up to pin 40 at the top-right.

| Signal | Pico 2 W pin | MPU-6050 | MAX30102 |
|---|---|---|---|
| 3.3 V power | **pin 36 (3V3 OUT)** | VCC | VIN |
| Ground | **pin 38 (GND)** (or any GND) | GND | GND |
| I2C data | **pin 6 (GP4, I2C0 SDA)** | SDA | SDA |
| I2C clock | **pin 7 (GP5, I2C0 SCL)** | SCL | SCL |
| (leave unconnected) | | XDA, XCL, AD0, INT | INT, IRD, RD |

- Use **3V3 OUT (pin 36), not VBUS (pin 40) or VSYS (pin 39)**. The Pico's pins are 3.3 V only.
- MPU-6050 AD0 unconnected = address 0x68 (the module pulls it low). Connect AD0 to 3.3 V only if you want 0x69.
- Both modules already have pull-up resistors on SDA/SCL, so you need no extra resistors.
- Breadboard: join the four shared lines on the breadboard rails, or twist/solder them together.

## 2. Build and upload (VS Code + PlatformIO, like your ESP projects)

1. VS Code → PlatformIO → **Open Project** → choose this folder (`firmware/v0_1_sensor_test`).
   The first build downloads the Raspberry Pi platform and toolchain (a few minutes, a few hundred MB).
   - If the download fails on Windows with a "Filename too long" error, run once in a terminal, then retry:
     `git config --global core.longpaths true`
2. Click **Build** (✓ in the bottom bar). It should end with `SUCCESS`.
3. **First upload only** (the Pico is blank):
   1. Hold the **BOOTSEL** button on the Pico, plug in the USB cable, then release the button.
   2. A USB drive called **RP2350** appears in File Explorer.
   3. Click **Upload** (→) in PlatformIO. If it fails to find the board, just drag
      `.pio/build/pico2w/firmware.uf2` onto the RP2350 drive. The drive disappears and the Pico reboots
      into the firmware.
4. **Later uploads**: just click **Upload**. The firmware lets PlatformIO reset the Pico over USB, no
   button needed. (If that ever stops working, do the BOOTSEL step again.)

## 3. Run

Click **Serial Monitor** (🔌 in the bottom bar, 115200 baud). The on-board LED blinks twice a second
while the firmware runs. Type a letter + Enter in the monitor:

- `s`: rescan the bus and re-initialise the sensors (do this first if you opened the monitor late and missed the startup messages)
- `h`: human view, 2 lines a second (default)
- `r`: raw CSV at 50 Hz: `t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red` (for the laptop visualizer later)

Good startup output:

```
# I2C scan: 0x57 0x68   (expected: 0x57 = MAX30102, 0x68 = MPU-6050)
# MPU-6050 WHO_AM_I = 0x68 (ok)
# MPU-6050 init OK
# MAX30102 PART_ID = 0x15 (ok)
# MAX30102 init OK
acc[g] x=+0.01 y=-0.02 z=+1.00 |a|=1.00  gyro[dps] x=  +0.4 y=  -0.8 z=  +0.2  27.3C   IR=  1200 red=   900 finger=no
```

## 4. Checklist: is it working?

| Test | What you should see |
|---|---|
| MPU-6050 flat on the table, chip up | `z ≈ +1.00`, `x, y ≈ 0`, `|a| ≈ 1.00` (gravity) |
| Flip it upside down | `z ≈ -1.00` |
| Stand it on each edge | `x` or `y` becomes ±1.00, `|a|` stays ≈ 1.00 |
| Hold still | gyro values all within a few °/s of 0 |
| Rotate it quickly | gyro values jump to hundreds of °/s |
| Shake it hard | `|a|` jumps above 2–3 g |
| Temperature | room temperature, roughly 25–35 °C |
| MAX30102 uncovered | `IR` small (hundreds to a few thousand), `finger=no`; a faint red LED glows |
| Fingertip resting lightly on it | `IR` jumps above 50 000, `finger=YES`. In raw mode (`r`) the IR value rises and falls slightly with each heartbeat |
| Wiggle each wire | values keep coming; if a sensor says `LOST`, that wire is loose. It reconnects by itself within 3 s once fixed |

## 5. Troubleshooting

| Symptom | Likely cause |
|---|---|
| No RP2350 drive appears | Charge-only USB cable (try another), or BOOTSEL released too early |
| No COM port / blank monitor | Firmware not uploaded yet, or monitor opened before the Pico finished booting: close and reopen it, then type `s` |
| `I2C scan: NOTHING FOUND` | 3V3/GND missing, or SDA and SCL swapped, or not on GP4/GP5 |
| Only one address found | That sensor's wires (often its VCC or GND) or a cold solder joint on its header |
| MPU WHO_AM_I is 0x70, 0x72 or 0x98 | Clone chip (MPU-6500-like). Usually works the same: check the gravity test above |
| MAX30102 found but IR never rises | Check that the LED glows red; some purple boards have weak 1.8 V pull-ups: try a slower bus by changing `Wire.setClock(400000)` to `100000` |
| Values freeze / garbage | Long or loose jumper wires: keep I2C wires short (< 20 cm) |
