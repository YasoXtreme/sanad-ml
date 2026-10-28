// Small animation helpers.

// Idle breathing: 0 -> 1 (inhale) -> 0 (exhale), eased, one cycle per periodS.
export function breathCurve(tS, periodS = 5, inhaleS = 2) {
  const u = ((tS % periodS) + periodS) % periodS;
  const ease = (x) => 0.5 - 0.5 * Math.cos(Math.PI * x);
  return u < inhaleS ? ease(u / inhaleS) : 1 - ease((u - inhaleS) / (periodS - inhaleS));
}

// One heartbeat: quick swell, slower fade. sinceS = seconds since the beat; peak value 1.
export function pulseEnvelope(sinceS) {
  if (!(sinceS >= 0) || sinceS > 2.5) return 0;
  const attack = 0.07;
  const decay = 0.3;
  return sinceS < attack ? Math.sin(((sinceS / attack) * Math.PI) / 2) : Math.exp(-(sinceS - attack) / decay);
}

// Critically damped spring (exact update, stable for any dt): smooth, no overshoot.
export class Spring {
  constructor(x = 0, freq = 6) {
    this.x = x;
    this.v = 0;
    this.w = freq;
  }

  update(target, dt) {
    const w = this.w;
    const x0 = this.x - target;
    const v0 = this.v;
    const e = Math.exp(-w * dt);
    this.x = target + (x0 + (v0 + w * x0) * dt) * e;
    this.v = (v0 - w * (v0 + w * x0) * dt) * e;
    return this.x;
  }
}

export const easeOutCubic = (u) => 1 - (1 - u) ** 3;

// Damped spring on a 3-vector. damping < 1 = elastic (overshoots and wobbles), 1 = no overshoot.
// extAcc: extra acceleration, e.g. inertia when the band accelerates (the jelly lags, then sloshes).
export class Spring3 {
  constructor(freqHz = 3, damping = 0.5) {
    this.x = [0, 0, 0];
    this.v = [0, 0, 0];
    this.w = 2 * Math.PI * freqHz;
    this.z = damping;
  }

  update(target, dt, extAcc = null) {
    const steps = Math.max(1, Math.ceil(dt * 240)); // small steps keep it stable
    const h = dt / steps;
    const { w, z } = this;
    for (let k = 0; k < steps; k++) {
      for (let i = 0; i < 3; i++) {
        const a = -w * w * (this.x[i] - target[i]) - 2 * z * w * this.v[i] + (extAcc ? extAcc[i] : 0);
        this.v[i] += a * h;
        this.x[i] += this.v[i] * h;
      }
    }
    return this.x;
  }

  kick(dv) {
    for (let i = 0; i < 3; i++) this.v[i] += dv[i];
  }
}

// v shortened to at most maxLen.
export function clampLen(v, maxLen) {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > maxLen ? [(v[0] * maxLen) / n, (v[1] * maxLen) / n, (v[2] * maxLen) / n] : v;
}

// Soft dead zone: vectors shorter than dz become 0, longer ones are shortened by dz (no jump at the edge).
export function deadzone(v, dz) {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > dz ? [(v[0] * (n - dz)) / n, (v[1] * (n - dz)) / n, (v[2] * (n - dz)) / n] : [0, 0, 0];
}
