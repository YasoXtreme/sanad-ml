// Every tunable of the visualizer in one place.

export const CONFIG = {
  periodMs: 20, // 50 Hz band stream

  // Jitter buffer (src/core/playback.js)
  playback: {
    baseDelayMs: 120, // how far behind live we play: hides Wi-Fi hiccups up to this size
    maxDelayMs: 400, // the delay grows up to this after bigger hiccups, then shrinks slowly
    extrapolateMs: 100, // on a gap, keep moving this long, then ease to a stop
    holdEaseMs: 300,
  },

  // Orientation filter (src/core/ahrs.js)
  ahrs: {
    kp: 1.0, // how hard gravity pulls the tilt back (rad/s per unit error)
    accelGateG: 0.2, // trust the accelerometer only when | |a| - 1 g | is below this
    yawTauS: 20, // heading slowly returns to "front" (no magnetometer, so yaw drifts)
  },

  // Band velocity for the dust (src/core/motion.js)
  // accDeadzoneMs2: ignore accelerations below ~0.12 g (sensor offsets turning with the wrist, tremor).
  motion: { hpTauS: 3, leakTauS: 2, quietAccMs2: 0.6, quietForS: 0.2, quietTauS: 0.25, accDeadzoneMs2: 1.2 },

  // Impact flash: the band's own trigger threshold (models/firmware/sanad_model_config.h SANAD_TRIGGER_G),
  // so a flash means "the band would wake the fall model here".
  impact: { thresholdG: 2.3047, windowMs: 500 },

  heart: { fingerOnIr: 50000, fingerOffIr: 40000 },

  breath: { periodS: 5, inhaleS: 2 },

  // Ice blue + warm heart
  palette: {
    deep: '#0b2a6e',
    mid: '#2f6fd6',
    ice: '#cfe8ff',
    core: '#ffffff',
    heart: '#ff8fa3',
    dust: '#9cc4ff',
    impact: '#e8f4ff',
  },

  blob: { shellCount: 6000, ringCount: 1400, size: 24 },

  // The blob is elastic: it drifts toward where the band moves and springs back, stretches like a soft
  // droplet along the motion (lagging behind when the band speeds up, sloshing forward when it brakes),
  // flattens a little when it spins, and squashes on impacts. Gains act on what is left above each dead
  // zone, so small movements do nothing. Tuned so a normal swipe (~1 m/s) drifts ~0.25 and a big
  // wave reaches the caps.
  elastic: {
    speedDeadzone: 0.1, // m/s of band speed ignored by drift and stretch
    offsetGain: 0.4, // scene units of drift per m/s above the dead zone (the blob is ~3 units wide)
    offsetMax: 0.5, // the blob never drifts further than this from the centre
    offsetHz: 2.8,
    offsetDamping: 0.75, // < 1: one small bounce when it comes back
    stretchGain: 0.22, // elongation per m/s above the dead zone
    stretchMax: 0.28,
    stretchHz: 3.2,
    stretchDamping: 0.5, // lower = more jelly wobble
    accDeadzone: 2, // m/s^2 of acceleration ignored by the lag / slosh
    inertia: 0.01, // extra stretch per m/s^2 above that
    spinDeadzone: 2, // rad/s (~115 deg/s) of rotation ignored by the flattening
    spinGain: 0.02, // flattening per rad/s above that
    spinMax: 0.12,
    impactKick: 5, // how hard an impact squashes it (spring speed)
  },
  dust: { count: 650, transGain: 2.5, rotGain: 0.35, trailS: 0.12, size: 16 },
  bloom: { strength: 0.95, radius: 0.55, threshold: 0.04 },
};
