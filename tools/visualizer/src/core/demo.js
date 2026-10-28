// Synthetic band for rehearsing without hardware: 50 Hz lines in the v0.2 format, with consistent
// accelerometer + gyroscope (gentle wrist sway, a swipe every few seconds), a heartbeat on the IR
// channel while the "finger" is on, and a scripted fall on demand.

import {
  DEG,
  qConj,
  qFromAxisAngle,
  qIntegrate,
  qMul,
  qNormalize,
  qRotate,
  vAdd,
  vLen,
  vNormalize,
  vScale,
} from './quat.js';

const G = 9.80665;
const TAU = 2 * Math.PI;

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const FALL_MS = 2200;
const LIE_MS = 4000; // stays down after the fall, then sways back up

export class DemoGenerator {
  constructor(seed = 7) {
    this.rand = mulberry32(seed);
    this.t = 0; // ms
    this.seq = 0;
    this.base = qNormalize(qMul(qFromAxisAngle([1, 0, 0], 0.45), qFromAxisAngle([0, 1, 0], -0.2)));
    this.q = this.base.slice();
    this.finger = false;
    this.fallAt = null;
    this.fallAxis = [1, 0, 0];
    this.impactDir = [1, 0, 0];
    this.phase = 0; // heartbeat phase 0..1
    this.biasDps = [0.8, -0.5, 0.3];
  }

  noise(k) {
    return (this.rand() - 0.5) * 2 * k;
  }

  toggleFinger() {
    this.finger = !this.finger;
    return this.finger;
  }

  triggerFall() {
    if (this.fallAt !== null && this.t - this.fallAt < FALL_MS + LIE_MS) return;
    this.fallAt = this.t;
    this.fallAxis = vNormalize([this.noise(1), this.noise(1), this.noise(1)]);
    this.impactDir = vNormalize([this.noise(1), this.noise(1), 2.5]);
  }

  // Body angular velocity (rad/s) and world linear acceleration (m/s^2) at time ts (s).
  motionAt(ts) {
    let sway = 1;
    let restore = 0.6;
    let w = [0, 0, 0];
    let lin = [0, 0, 0];
    if (this.fallAt !== null) {
      const u = this.t - this.fallAt;
      if (u < FALL_MS) {
        sway = 0;
        restore = 0;
        if (u < 350) {
          w = vScale(this.fallAxis, 220 * DEG);
          lin = [0, 0, -0.8 * G];
        } else if (u < 410) {
          w = vScale(vNormalize([this.fallAxis[1], -this.fallAxis[0], 0.5]), 350 * DEG);
          lin = vScale(this.impactDir, 4.2 * G);
        } else {
          w = vScale(this.fallAxis, 60 * DEG * Math.exp(-(u - 410) / 250));
        }
      } else if (u < FALL_MS + LIE_MS) {
        sway = 0;
        restore = 0;
      } else {
        const k = Math.min(1, (u - FALL_MS - LIE_MS) / 3000);
        sway = k;
        restore = 0.6 * k;
        if (k >= 1) this.fallAt = null;
      }
    }

    if (sway > 0) {
      w = vAdd(
        w,
        vScale(
          [
            25 * Math.sin(TAU * 0.13 * ts + 0.3) + 12 * Math.sin(TAU * 0.41 * ts),
            30 * Math.sin(TAU * 0.09 * ts + 1.1) + 10 * Math.sin(TAU * 0.33 * ts + 2),
            20 * Math.sin(TAU * 0.07 * ts + 2.2) + 8 * Math.sin(TAU * 0.27 * ts),
          ],
          sway * DEG,
        ),
      );
      // Small sway of the arm plus a swipe every 7 s (alternating left/right).
      const f = [0.37, 0.29, 0.23];
      const amp = [0.12, 0.1, 0.06];
      const ph = [0, 1, 2];
      for (let i = 0; i < 3; i++) {
        lin[i] += -sway * amp[i] * (TAU * f[i]) ** 2 * Math.sin(TAU * f[i] * ts + ph[i]);
      }
      const cycle = ts % 7;
      if (cycle < 0.8) {
        const dir = Math.floor(ts / 7) % 2 ? -1 : 1;
        lin[0] += sway * dir * 6 * Math.sin((TAU * cycle) / 0.8);
      }
    }
    if (restore > 0) {
      // Pull back toward the resting pose so the demo hovers nicely.
      const e = qMul(qConj(this.q), this.base);
      const s = vLen([e[1], e[2], e[3]]);
      if (s > 1e-6) {
        const ang = 2 * Math.atan2(s, e[0]);
        w = vAdd(w, vScale([e[1] / s, e[2] / s, e[3] / s], ang * restore));
      }
    }
    return { w, lin };
  }

  // Next 20 ms sample as a stream line.
  next() {
    const dt = 0.02;
    const ts = this.t / 1000;
    const { w, lin } = this.motionAt(ts);
    this.q = qIntegrate(this.q, w, dt);

    const specific = vAdd([0, 0, 1], vScale(lin, 1 / G)); // what an accelerometer feels, in g
    const a = qRotate(qConj(this.q), specific).map((x) => x + this.noise(0.01));
    const g = w.map((x, i) => x / DEG + this.biasDps[i] + this.noise(0.3));

    let ir;
    if (this.finger) {
      const bpm = 72 + 3 * Math.sin(TAU * 0.1 * ts);
      this.phase = (this.phase + (dt * bpm) / 60) % 1;
      const p = this.phase;
      const bv = Math.exp(-(((p - 0.15) / 0.07) ** 2)) + 0.35 * Math.exp(-(((p - 0.45) / 0.09) ** 2));
      ir = 86000 - 420 * bv + 300 * Math.sin(TAU * 0.05 * ts) + this.noise(15);
    } else {
      ir = 1200 + this.noise(30);
    }
    ir = Math.round(ir);
    const red = Math.round(this.finger ? ir * 0.62 : 900 + this.noise(20));

    const line =
      `${Math.round(this.t)},${a.map((x) => x.toFixed(3)).join(',')},` +
      `${g.map((x) => x.toFixed(1)).join(',')},${ir},${red},${this.seq}`;
    this.t += 20;
    this.seq++;
    return line;
  }
}

// Simulated Wi-Fi delivery for the jitter buffer: in order (like TCP), usually a few ms late, sometimes
// stalled for 80-300 ms and then delivered in a burst.
export class JitterSim {
  constructor(seed = 11) {
    this.rand = mulberry32(seed);
    this.last = -Infinity;
    this.stallUntil = -Infinity;
  }

  release(t) {
    if (t > this.stallUntil && this.rand() < 0.005) this.stallUntil = t + 80 + this.rand() * 220;
    const r = Math.max(t + 3 + this.rand() * 15, this.stallUntil, this.last);
    this.last = r;
    return r;
  }
}
