# Sanad firmware v0.2: Wi-Fi stream

Streams the MPU-6050 and MAX30102 readings from the band to the laptop **over Wi-Fi** (for the browser
visualizer in `tools/visualizer/`), and over USB like v0.1. Still no model: this is for the
presentation demo and for checking the sensors without a cable.

## What's new since v0.1

| | v0.1 | v0.2 |
|---|---|---|
| Output | USB only | USB **and** Wi-Fi (WebSocket at `ws://<band>/ws`, up to 3 visualizers at once) |
| Wi-Fi | none | joins a saved network (phone hotspot, home Wi-Fi); if none is in range it makes its own network **Sanad-Band** |
| Timing | the Pico reads the newest sample every 20 ms | samples come from the MPU's own FIFO at 50 Hz, so a busy moment loses nothing |
| Raw line | `t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red` | same, plus `seq` (sample counter): **a gap in `seq` = lost samples** |
| Low-pass filter | ~44 Hz | ~20 Hz (DLPF 4), as in `docs/device_data_format.md`; clone chips get it on the accelerometer too |
| Commands | `h`, `r`, `s` | also `w` (Wi-Fi status), `a` (own network now), `j` (look for saved networks again) |
| Status page | none | open `http://<band>/` in a browser |

Wiring is **exactly the same as v0.1**: see [`../v0_1_sensor_test/README.md`](../v0_1_sensor_test/README.md).

## 1. Save your Wi-Fi networks (optional)

Skip this and the band always makes its own network "Sanad-Band" (section 4).

1. In `src/`, copy `wifi_secrets.example.h` to **`wifi_secrets.h`** (same folder).
2. Put in up to 3 networks, most important first. Names and passwords are case-sensitive:

   ```cpp
   static const WifiNetwork SAVED_NETWORKS[] = {
       {"Fares iPhone", "hotspot-password"},
       {"Home WiFi", "home-password"},
   };
   ```
3. Build and upload again (the networks are built into the firmware).

`wifi_secrets.h` is in `.gitignore`, so passwords never reach the public repo.
The Pico 2 W only has **2.4 GHz** Wi-Fi: on an iPhone hotspot turn on **Maximise Compatibility**; on Android
choose the 2.4 GHz band if the hotspot settings offer it. WPA2 or WPA2/WPA3 works; "WPA3 only" doesn't.

## 2. Build and upload

Same as v0.1, but open **this** folder (`firmware/v0_2_wifi_stream`) in VS Code → PlatformIO → Open Project,
then **Build** (✓) and **Upload** (→). The Pico already runs v0.1, so no BOOTSEL press is needed.

## 3. Run

Open the **Serial Monitor** (🔌, 115200 baud). Startup takes a few seconds while the band looks for Wi-Fi.
Good startup output (here with a saved phone hotspot in range):

```
# Sanad v0.2.0 Wi-Fi stream. Commands: h = human view, r = raw 50 Hz CSV, s = rescan sensors, w = Wi-Fi status, a = own network now, j = join saved network again
# I2C scan: 0x57 0x68   (expected: 0x57 = MAX30102, 0x68 = MPU-6050)
# MPU-6050 WHO_AM_I = 0x70 (not 0x68: MPU-6500-type clone, works the same)
# MPU-6050 init OK (FIFO, MPU clock)
# MAX30102 PART_ID = 0x15 (ok)
# MAX30102 init OK
# WS self-test OK
# Wi-Fi: looking for saved networks ...
# Wi-Fi: joining 'Fares iPhone' ...
# Wi-Fi: joined 'Fares iPhone', band at 172.20.10.4 (or sanad-band.local), signal -45 dBm
# Visualizer: ws://172.20.10.4/ws   status page: http://172.20.10.4/
# Visualizers connected: 0 of 3, lines sent 0, dropped 0
```

Then the human view starts, as in v0.1, with a Wi-Fi line every 5 s. If you opened the monitor late and
missed all this, type `w`.

Type a letter + Enter in the monitor (a visualizer can send the same letters over the WebSocket):

| Key | What it does |
|---|---|
| `h` | human view, 2 lines a second (default) |
| `r` | raw CSV at 50 Hz on USB: `t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red,seq` |
| `s` | rescan the bus and re-initialise the sensors |
| `w` | Wi-Fi status: network, the band's address, visualizers connected, lines dropped |
| `a` | switch to the band's own network "Sanad-Band" now |
| `j` | look for the saved networks again (e.g. you turned the hotspot on after the band started) |

The on-board LED: **slow blink** = running, no visualizer connected; **on, with a short blink-off every
second** = streaming to at least one visualizer.

## 4. Find the band

| The band is on... | Laptop joins... | Visualizer address | Status page |
|---|---|---|---|
| a saved network | the **same** network (e.g. the phone hotspot) | `ws://sanad-band.local/ws`, or the IP from `w`: `ws://172.20.10.4/ws` | `http://sanad-band.local/` |
| its own network | Wi-Fi **Sanad-Band**, password **`sanad2026`** | `ws://192.168.42.1/ws` | `http://192.168.42.1/` |

- The band falls back to its own network when no saved network is in range at startup, or when the saved
  network disappears for more than 15 s. It stays on its own network until you type `j` or restart it.
- `sanad-band.local` needs mDNS, which some networks block: the IP address always works (type `w`).
- Check the band is reachable by opening the status page in a browser before starting the visualizer.

## 5. The stream (for the visualizer)

- Every WebSocket text message is **exactly one line, without a newline**. USB raw mode sends the same
  lines, each ending in `\n`.
- Data line: `t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red,seq`, e.g. `81234,0.012,-0.981,0.104,0.4,-1.2,0.0,1250,980,4021`.
  Acceleration in g (3 decimals, range ±16 g), rotation in °/s (1 decimal, ±2000), `ir`/`red` = raw MAX30102
  counts (newest value), `seq` = sample counter. **Axes are the MPU's chip axes**, not yet the Sanad frame.
- `t_ms` = the Pico's clock when the MPU took the sample (good to ~5 ms).
- Lines starting with `#` are comments (status messages). On connecting, a visualizer first gets
  `# Sanad v0.2.0 band: ...` and then the column header `# t_ms,ax_g,...,seq`.
- `seq` goes up by 1 per sample. A jump means samples were lost: on Wi-Fi (a slow link: the band drops a
  line rather than fall behind), or for everyone (the band was busy, e.g. reconnecting to Wi-Fi).
- If a visualizer can't keep up for 3 s, the band closes its connection (reconnect to continue).

## 6. On stage: checklist

- [ ] **Laptop charger plugged in.** On battery, Windows' Wi-Fi power saving delays packets and the
      visualizer stutters.
- [ ] **Best setup: phone hotspot**, saved in `wifi_secrets.h`. Laptop and band both join it, and the laptop
      keeps its internet. iPhone: keep the Personal Hotspot screen open while the band starts, so it can be found.
- [ ] **Joining "Sanad-Band" cuts the laptop's internet** (the band has none). Open everything you need from
      the internet (slides, videos, the visualizer page) before switching.
- [ ] **Stop Windows from roaming away**: Settings → Network & internet → Wi-Fi → Manage known networks →
      for every other network (school, home), untick **Connect automatically**. Otherwise Windows may leave a
      network "with no internet" in the middle of the demo.
- [ ] Do a full run 10 minutes before, in the room, with the real phone: status page opens, visualizer
      moves when you move the band, `dropped` stays near 0 (type `w`, or look at the status page).
- [ ] School/venue Wi-Fi often blocks devices from talking to each other: use the hotspot or Sanad-Band.
- [ ] **USB cable in your bag** as a fallback: the same lines come out of the USB port in raw mode (`r`).

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `none of the saved networks is in range` | Hotspot off or 5 GHz only (iPhone: Maximise Compatibility), name typed differently (case matters), or `wifi_secrets.h` changed but not uploaded again |
| `could not join '...'` | Wrong password, hotspot set to WPA3 only, or the band is too far away |
| `no saved networks` | `wifi_secrets.h` doesn't exist (only the example): the band uses its own network, which is fine |
| `sanad-band.local` doesn't open | mDNS blocked on that network: use the IP address from `w` |
| Status page opens, visualizer doesn't connect | Address must be `ws://.../ws` (not `http://`). A visualizer page opened from an `https://` site can't use `ws://`: open it from the laptop (file or `http://localhost`) |
| `refused: already 3 connected` | Three visualizers (browser tabs) are open: close the others |
| Many `dropped` lines / jumps in `seq` | Weak Wi-Fi (move closer, keep the phone between laptop and band), laptop on battery, or a busy 2.4 GHz band: change `AP_CHANNEL` in `src/config.h` to 1 or 11 |
| `MPU FIFO overflow: ~N samples lost` | The band was busy for more than ~0.8 s (normal while it reconnects to Wi-Fi). If it happens all the time, report it |
| Data looks wrong on a clone chip | Set `#define SANAD_USE_FIFO 0` in `src/config.h` (v0.1-style timing) and compare |
| `WS self-test FAILED` | The build is broken (the WebSocket handshake can't work): report it |
| Sensor problems | Same as v0.1: see its troubleshooting table |

## How it works (for the curious)

- `src/sensors.cpp`: the MPU puts every 50 Hz sample (12 bytes: accel + gyro) into its FIFO; the Pico
  checks every 5 ms and reads up to 20 samples at a time. The genuine MPU-6050 FIFO holds 1024 bytes, the
  clone's 512 (≈ 0.85 s). When it gets full, the band empties it and adds the lost samples to `seq`.
- `src/net.cpp`: Wi-Fi, plus a tiny WebSocket server (RFC 6455) with no libraries. A line is written only
  when the TCP buffer has room for all of it, so a slow laptop never holds up the sensors.
- `src/config.h`: everything you might want to change (own-network name/password/channel, timeouts).
