// Band velocity (for the dust) and impact detection.

import { qRotate } from './quat.js';

const G = 9.80665;

// Velocity from gravity-removed acceleration. Integrating acceleration drifts without bound, so:
// a slow high-pass removes leftover gravity/orientation error first, and the integral leaks back to 0:
// slowly while the band moves (so braking at the end of a push doesn't read as moving backwards),
// quickly once it has been quiet for a moment (a "zero-velocity update", so offsets can't build up).
// The result is "how the band moved in the last second or two", which is all the visuals need.
// An IMU alone can't see steady motion: a slow constant-speed move still swings back a bit at the stop.
export class Motion {
  constructor(cfg = {}) {
    // accDeadzoneMs2: acceleration below this is ignored. A cheap accelerometer reads a few hundredths
    // of a g off, and when the wrist turns that error turns with it and looks like movement.
    this.cfg = {
      hpTauS: 3,
      leakTauS: 2,
      quietAccMs2: 0.6,
      quietForS: 0.2,
      quietTauS: 0.25,
      accDeadzoneMs2: 0,
      ...cfg,
    };
    this.reset();
  }

  reset() {
    this.mean = null;
    this.v = [0, 0, 0]; // m/s, world frame (Z up)
    this.lin = [0, 0, 0]; // m/s^2, world frame
    this.quietS = 0;
  }

  update(a, q, dt) {
    const aw = qRotate(q, a);
    aw[2] -= 1;
    if (!this.mean) {
      this.mean = aw;
      return;
    }
    if (!(dt > 0)) return;
    const c = this.cfg;
    if (dt > 0.1) {
      // gap in the data: let the velocity fade, don't integrate across it
      const fade = Math.exp(-dt / c.quietTauS);
      this.v = this.v.map((x) => x * fade);
      return;
    }
    const k = Math.min(1, dt / c.hpTauS);
    for (let i = 0; i < 3; i++) {
      this.mean[i] += k * (aw[i] - this.mean[i]);
      this.lin[i] = (aw[i] - this.mean[i]) * G;
    }
    const mag = Math.hypot(this.lin[0], this.lin[1], this.lin[2]);
    this.quietS = mag < c.quietAccMs2 ? this.quietS + dt : 0;
    const keep = mag > c.accDeadzoneMs2 ? (mag - c.accDeadzoneMs2) / mag : 0; // soft dead zone
    for (let i = 0; i < 3; i++) this.lin[i] *= keep;
    const leak = Math.exp(-dt / (this.quietS > c.quietForS ? c.quietTauS : c.leakTauS));
    for (let i = 0; i < 3; i++) this.v[i] = this.v[i] * leak + this.lin[i] * dt;
  }
}

// |a| with each axis clipped like the band does before its trigger.
export function accMagnitude(a, limitG = 8) {
  const c = (x) => Math.max(-limitG, Math.min(limitG, x));
  return Math.hypot(c(a[0]), c(a[1]), c(a[2]));
}

// Impact events from RAW samples (before any smoothing). An event starts when |a| crosses the
// threshold; for the next windowMs a higher |a| moves the event to that sample, so the event ends up
// on the peak. The returned object keeps updating while its window is open.
export class ImpactDetector {
  constructor(cfg = {}) {
    this.cfg = { thresholdG: 2.3047, windowMs: 500, ...cfg };
    this.reset();
  }

  reset() {
    this.armed = true;
    this.ev = null;
  }

  push(s) {
    const m = accMagnitude(s.a);
    if (this.ev) {
      if (s.t <= this.ev.end) {
        if (m > this.ev.peakG) {
          this.ev.peakG = m;
          this.ev.t = s.t;
          this.ev.a = s.a;
        }
        return null;
      }
      this.ev = null;
    }
    if (!this.armed) {
      if (m < this.cfg.thresholdG) this.armed = true;
      return null;
    }
    if (m >= this.cfg.thresholdG) {
      this.armed = false;
      this.ev = { type: 'impact', t: s.t, peakG: m, a: s.a, end: s.t + this.cfg.windowMs };
      return this.ev;
    }
    return null;
  }
}
