// Orientation from accelerometer + gyroscope: a Mahony complementary filter.
// The gyro is integrated for fast, smooth rotation; gravity seen by the accelerometer slowly pulls the
// tilt back so it can't drift. Heading (yaw) has no reference (no magnetometer), so it slowly returns
// to "front" instead. The gyro bias is learned whenever the band lies still.

import {
  DEG,
  qFromAxisAngle,
  qFromTwoVectors,
  qIdentity,
  qIntegrate,
  qMul,
  qRotate,
  vAdd,
  vCross,
  vLen,
  vScale,
  vSub,
} from './quat.js';

const DEFAULTS = {
  kp: 1.0,
  accelGateG: 0.2,
  yawTauS: 20,
  stillTauS: 0.5, // averaging time for the stillness detector
  stillStdDps: 1.5, // gyro wobble below this...
  stillMaxDps: 15, // ...and gyro reading below this (a steady fast spin is not "still")
  stillAccG: 0.1,
  stillForS: 1.0,
  biasTauS: 2.0,
};

export class Ahrs {
  constructor(cfg = {}) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.reset();
  }

  reset() {
    this.q = null;
    this.bias = [0, 0, 0];
    this.wBody = [0, 0, 0]; // rad/s, bias removed
    this.wWorld = [0, 0, 0];
    this.gMean = [0, 0, 0];
    this.gVar = [0, 0, 0];
    this.stillS = 0;
  }

  get still() {
    return this.stillS > this.cfg.stillForS;
  }

  // a in g, gDps in deg/s (body frame), dt in s.
  update(a, gDps, dt) {
    const c = this.cfg;
    const g = vScale(gDps, DEG);
    const an = vLen(a);
    if (!this.q) {
      this.q = an > 0.5 ? qFromTwoVectors(a, [0, 0, 1]) : qIdentity();
      this.gMean = g.slice();
      return;
    }
    if (!(dt > 0)) return;
    dt = Math.min(dt, 0.1);

    // Stillness: low gyro wobble, small gyro reading, |a| close to 1 g.
    const k = Math.min(1, dt / c.stillTauS);
    for (let i = 0; i < 3; i++) {
      const d = g[i] - this.gMean[i];
      this.gMean[i] += k * d;
      this.gVar[i] += k * (d * d - this.gVar[i]);
    }
    const wobble = Math.sqrt(Math.max(...this.gVar)) / DEG;
    const calm =
      wobble < c.stillStdDps && vLen(this.gMean) / DEG < c.stillMaxDps && Math.abs(an - 1) < c.stillAccG;
    this.stillS = calm ? this.stillS + dt : 0;
    if (this.still) {
      const kb = Math.min(1, dt / c.biasTauS);
      for (let i = 0; i < 3; i++) this.bias[i] += kb * (this.gMean[i] - this.bias[i]);
    }

    const w = vSub(g, this.bias);
    this.wBody = w;
    let wc = w;
    // Trust the accelerometer fully at exactly 1 g, less and less up to the gate (then not at all).
    const trust = Math.max(0, 1 - Math.abs(an - 1) / c.accelGateG) ** 2;
    if (trust > 0) {
      const [qw, qx, qy, qz] = this.q;
      // Gravity direction predicted in the body frame.
      const v = [2 * (qx * qz - qw * qy), 2 * (qw * qx + qy * qz), qw * qw - qx * qx - qy * qy + qz * qz];
      const e = vCross(vScale(a, 1 / an), v);
      wc = vAdd(w, vScale(e, c.kp * trust));
    }
    this.q = qIntegrate(this.q, wc, dt);

    if (c.yawTauS > 0) {
      const { heading, weight } = this.headingInfo();
      const turn = -heading * weight * Math.min(1, dt / c.yawTauS);
      if (turn) this.q = qMul(qFromAxisAngle([0, 0, 1], turn), this.q);
    }
    this.wWorld = qRotate(this.q, w);
  }

  // Heading of the band's +x axis in the horizontal plane; weight ~0 when +x points up or down.
  headingInfo() {
    const xw = qRotate(this.q, [1, 0, 0]);
    const h = Math.hypot(xw[0], xw[1]);
    return { heading: h > 1e-6 ? Math.atan2(xw[1], xw[0]) : 0, weight: Math.min(1, h / 0.5) };
  }

  // Make the current heading "front" right now.
  recenter() {
    if (!this.q) return;
    const { heading } = this.headingInfo();
    this.q = qMul(qFromAxisAngle([0, 0, 1], -heading), this.q);
  }
}
