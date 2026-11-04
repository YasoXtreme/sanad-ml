# Sanad live visualizer

A pitch-black page where the band appears as a glowing, breathing cloud of particles:

- **The blob is the band.** It turns exactly as the band turns (gyro + gravity filter). The flowing ring lies in
  the band-face plane, and the brighter side points along the chip's +x axis.
- **It's elastic.** When the band really moves (a swipe, a wave), the blob drifts a little toward the motion (at
  most about a sixth of its width) and springs back with one small bounce. It stretches along the motion like a
  soft droplet (the trailing side longer), lags when the band speeds up and sloshes when it brakes, flattens
  slightly on fast spins, and squashes along the hit on an impact. Small movements (tremor, turning the wrist,
  little nudges) do nothing: they sit inside dead zones (`elastic` and `motion` in `src/config.js`).
- **Dust shows motion.** The "air" streams past the blob: dust flows opposite to how the band moves and
  counter-rotates when it spins, leaving short trails that fade within about a second.
- **Breathing.** Idle, it breathes on a slow 5 s cycle. With a finger on the heart sensor, it pulses with each
  detected heartbeat instead, sends out thin rose rings, and shows the bpm at the bottom.
- **Impacts flash.** When |a| crosses the band's own fall trigger (2.30 g, from `models/firmware/sanad_model_config.h`)
  the blob flashes, a shockwave ring goes out and the dust bursts outward. A flash means "here the band would
  wake the fall model".

Everything is local (three.js is in `vendor/`), so it works with no internet, e.g. while the laptop is on the
band's own Wi-Fi.

## Run

From the repo root:

```powershell
uv run python -I tools/visualizer/serve.py
```

This opens <http://localhost:8000> in your browser. Use **Chrome or Edge** (USB needs them).

## Connect

| Button | What it does |
|---|---|
| **Wi-Fi** | Connects to `ws://<address>/ws`. Default address `192.168.42.1` = the band's own network "Sanad-Band". If the band joined your phone hotspot or home Wi-Fi instead, type the IP it printed (serial command `w`) or `sanad-band.local`. Reconnects by itself if the link drops. |
| **USB** | Pick the Pico in the list (needs firmware v0.1 or v0.2). The page switches the band to raw mode itself. If the cable comes out, it reconnects when it's back. **Close PlatformIO's Serial Monitor first**: Windows lets only one program use the port (the status line says "USB port busy" until it's free, then connects by itself). |
| **Demo** | A built-in fake band, no hardware needed. Good for rehearsing and as a last-resort fallback. |
| drop a `.csv` | Replays a recording made with **C** at its original speed. |

The status line shows the sample rate (should be ~50 Hz), how far behind live it plays (the jitter buffer,
normally 120–150 ms), dropped samples and the current |a|.

## Phone alerts (proof of concept)

Every impact flash also sends a push notification to a phone through [ntfy.sh](https://ntfy.sh) (free, no
account). It's only the trigger (|a| > 2.30 g), not the fall model, and at most one alert per 10 s.

1. Install the **ntfy** app (Android / iPhone) and subscribe to the topic `serve.py` prints at startup
   (`sanad-...`; it's random, saved in `tools/visualizer/.alert_topic`, not in git).
2. Shake or tap the band hard (or **Demo** + Space). The page shows "📱 alert sent to phone".

The laptop needs internet. Turn it off with `serve.py --no-alerts`. The real band will alert through the
cloud server and the Telegram bot instead.

## Keys

| Key | |
|---|---|
| F | fullscreen |
| H | hide / show the overlay (it also fades out by itself after 3 s without mouse movement) |
| R | recentre: make the band's current heading "front" (heading drifts slowly; it also re-centres itself over ~20 s) |
| C | start / stop recording; stopping downloads `sanad-<date>-<time>.csv` (the raw stream lines) |
| Space | demo only: a fall |
| B | demo only: finger on / off the heart sensor |
| J | demo only: simulate bad Wi-Fi (stalls up to 0.3 s) to see the jitter buffer cope |

## Stage checklist

1. Laptop on the charger (Windows' Wi-Fi power saving adds lag on battery).
2. Band powered, laptop joined to **Sanad-Band** (or both on the same phone hotspot).
3. `uv run python -I tools/visualizer/serve.py`, press **Wi-Fi**, then **F** for fullscreen, move the mouse away.
4. If Wi-Fi misbehaves: plug the USB cable in and press **USB** (Serial Monitor closed). Last resort: **Demo**.

## How it stays smooth

Wi-Fi delivers samples in bursts. Samples are not drawn when they land: they are played back on the band's own
clock (`t_ms`), about 120 ms behind live, interpolating between samples every frame (`src/core/playback.js`).
If the link hiccups, the buffer grows to cover it (up to 0.4 s) and slowly shrinks back; during a gap the motion
coasts for 0.1 s and then eases to a stop instead of freezing. Impacts and heartbeats are detected on the raw
samples before any smoothing.

The band's velocity (for the drift, stretch and dust) comes from integrating gravity-free acceleration, which
an IMU can only do for a moment: it leaks back to zero slowly while the band moves and quickly once it is
still (`src/core/motion.js`). So a quick push looks right, but a slow, steady move shows mostly its start and
its stop, with a small swing back at the end.

## Code

| Path | What |
|---|---|
| `src/core/` | pure logic, no graphics: stream parsing, jitter buffer, orientation filter (Mahony), band velocity, impact and heartbeat detection, demo band. Tested with `node --test "tools/visualizer/test/*.test.js"` (Node 18+, no packages) |
| `src/render/` | three.js scene: blob, dust, rings, bloom |
| `src/transport.js` | Wi-Fi (WebSocket), USB (Web Serial), demo, replay |
| `src/config.js` | every tunable: colours, gains, thresholds, buffer sizes; `elastic` sets how far the blob drifts, how much it stretches and how wobbly it is |
| `vendor/three/` | three.js r170 (MIT licence) |

Stream format (one line per 50 Hz sample, lines starting with `#` are comments from the firmware):
`t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,ir,red[,seq]` (`seq` from firmware v0.2 on). Axes are the chip's own
(not yet rotated into the Sanad frame).

Credits: [three.js](https://threejs.org) (MIT), simplex noise by Ashima Arts / Stefan Gustavson (MIT).
