import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Ahrs } from '../src/core/ahrs.js';
import { DemoGenerator } from '../src/core/demo.js';
import { HeartDetector } from '../src/core/heart.js';
import { ImpactDetector, Motion } from '../src/core/motion.js';
import { parseLine } from '../src/core/parse.js';
import { DEG, qAngle, qConj, qFromAxisAngle, qIdentity, qRotate, vLen } from '../src/core/quat.js';

const s = (t, a, g = [0, 0, 0], ir = 0) => ({ t, a, g, ir, red: 0, seq: null });

test('ahrs: 90 deg/s about z for 1 s turns the heading 90 deg', () => {
  const f = new Ahrs({ yawTauS: 0 });
  f.update([0, 0, 1], [0, 0, 0], 0);
  for (let i = 0; i < 50; i++) f.update([0, 0, 1], [0, 0, 90], 0.02);
  assert.ok(Math.abs(f.headingInfo().heading / DEG - 90) < 1, `${f.headingInfo().heading / DEG}`);
});

test('ahrs: tilt converges to gravity without gyro help', () => {
  const f = new Ahrs({ yawTauS: 0 });
  f.update([0, 0, 1], [0, 0, 0], 0); // starts level
  const tilted = qFromAxisAngle([1, 0, 0], 30 * DEG);
  const a = qRotate(qConj(tilted), [0, 0, 1]); // gravity as the tilted band sees it
  for (let i = 0; i < 500; i++) f.update(a, [0, 0, 0], 0.02);
  const up = qRotate(qConj(f.q), [0, 0, 1]);
  const err = Math.acos(Math.min(1, up[0] * a[0] + up[1] * a[1] + up[2] * a[2])) / DEG;
  assert.ok(err < 1, `tilt error ${err} deg`);
});

test('ahrs: learns the gyro bias while still', () => {
  const f = new Ahrs({ yawTauS: 0 });
  for (let i = 0; i < 500; i++) f.update([0, 0, 1], [2, -1.5, 0.7], 0.02);
  assert.ok(Math.abs(f.bias[0] / DEG - 2) < 0.1 && Math.abs(f.bias[1] / DEG + 1.5) < 0.1);
  // ...but not from a steady fast spin
  const f2 = new Ahrs({ yawTauS: 0 });
  for (let i = 0; i < 500; i++) f2.update([0, 0, 1], [0, 0, 90], 0.02);
  assert.ok(vLen(f2.bias) / DEG < 0.01);
});

test('ahrs follows the demo band (tilt error small)', () => {
  const gen = new DemoGenerator(3);
  const f = new Ahrs({ yawTauS: 0 });
  let worst = 0;
  for (let i = 0; i < 1500; i++) {
    const smp = parseLine(gen.next());
    f.update(smp.a, smp.g, i ? 0.02 : 0);
    if (i > 500) {
      const est = qRotate(qConj(f.q), [0, 0, 1]);
      const tru = qRotate(qConj(gen.q), [0, 0, 1]);
      const err = Math.acos(Math.min(1, est[0] * tru[0] + est[1] * tru[1] + est[2] * tru[2])) / DEG;
      worst = Math.max(worst, err);
    }
  }
  assert.ok(worst < 6, `worst tilt error ${worst} deg`);
});

test('motion: velocity follows a push and leaks back to zero', () => {
  const m = new Motion();
  const q = qIdentity();
  m.update([0, 0, 1], q, 0);
  let peak = 0;
  for (let i = 0; i < 25; i++) {
    m.update([0.2, 0, 1], q, 0.02); // 0.2 g forward for 0.5 s
    peak = Math.max(peak, vLen(m.v));
  }
  for (let i = 0; i < 250; i++) m.update([0, 0, 1], q, 0.02);
  assert.ok(peak > 0.3, `peak ${peak}`);
  // the high-pass leaves a small, fading backflow; after 5 s it is a few % of the peak
  assert.ok(vLen(m.v) < 0.05 * peak, `left ${vLen(m.v)} of ${peak}`);
});

test('motion: a constant accelerometer offset does not drift', () => {
  const m = new Motion();
  const q = qIdentity();
  for (let i = 0; i < 1000; i++) m.update([0.05, 0, 1.03], q, i ? 0.02 : 0);
  assert.ok(vLen(m.v) < 0.01, `v ${vLen(m.v)}`);
});

test('impact: one event per spike, moved onto the peak', () => {
  const d = new ImpactDetector({ thresholdG: 2.3 });
  const mags = [1, 1, 2.5, 4.0, 3.0, 1.2, 1, 1];
  const evs = mags.map((m, i) => d.push(s(i * 20, [0, 0, m]))).filter(Boolean);
  assert.equal(evs.length, 1);
  assert.equal(evs[0].t, 60);
  assert.equal(evs[0].peakG, 4);
  // re-arms after |a| drops and the window has passed
  for (let i = 8; i < 40; i++) d.push(s(i * 20, [0, 0, 1]));
  assert.ok(d.push(s(900, [0, 0, 3])));
});

test('heart: finds 72 bpm on the demo PPG, nothing without a finger', () => {
  const gen = new DemoGenerator(5);
  const h = new HeartDetector();
  let beats = 0;
  for (let i = 0; i < 500; i++) if (h.push(parseLine(gen.next()))) beats++;
  assert.equal(beats, 0);
  assert.equal(h.bpm, null);
  gen.toggleFinger();
  for (let i = 0; i < 1500; i++) if (h.push(parseLine(gen.next()))) beats++;
  // demo bpm swings 69-75
  assert.ok(h.bpm > 66 && h.bpm < 78, `bpm ${h.bpm}`);
  assert.ok(beats > 30 * 1.1 && beats < 30 * 1.3 + 2, `beats ${beats}`);
});

test('quaternion helpers sanity', () => {
  const q = qFromAxisAngle([0, 0, 1], Math.PI / 2);
  const v = qRotate(q, [1, 0, 0]);
  assert.ok(Math.abs(v[0]) < 1e-9 && Math.abs(v[1] - 1) < 1e-9);
  assert.ok(Math.abs(qAngle(q, qIdentity()) - Math.PI / 2) < 1e-9);
});

test('elastic spring: settles on target; low damping overshoots (wobble), damping 1 does not', async () => {
  const { Spring3 } = await import('../src/render/anim.js');
  const run = (damping) => {
    const sp = new Spring3(3, damping);
    let peak = 0;
    for (let i = 0; i < 300; i++) peak = Math.max(peak, sp.update([1, 0, 0], 1 / 60)[0]);
    return { end: sp.x[0], peak };
  };
  const wobbly = run(0.28);
  const firm = run(1);
  assert.ok(Math.abs(wobbly.end - 1) < 0.01 && Math.abs(firm.end - 1) < 0.01);
  assert.ok(wobbly.peak > 1.3, `overshoot ${wobbly.peak}`);
  assert.ok(firm.peak < 1.001, `no overshoot ${firm.peak}`);
});

test('motion: a one-way push barely swings back (slow leak while moving)', () => {
  const m = new Motion();
  const q = qIdentity();
  m.update([0, 0, 1], q, 0);
  let vmax = 0;
  let vmin = 0;
  for (let i = 0; i < 200; i++) {
    const t = i * 0.02;
    const a = t < 0.8 ? 6 * Math.sin((2 * Math.PI * t) / 0.8) : 0; // true speed: up to 1.53 m/s and back to 0
    m.update([a / 9.80665, 0, 1], q, 0.02);
    vmax = Math.max(vmax, m.v[0]);
    vmin = Math.min(vmin, m.v[0]);
  }
  assert.ok(vmax > 1.1, `peak ${vmax}`);
  assert.ok(-vmin < 0.4 * vmax, `backswing ${-vmin} of ${vmax}`);
  assert.ok(Math.abs(m.v[0]) < 0.05, `settles, left ${m.v[0]}`);
});

test('motion: with the dead zone, a turning wrist with a biased accelerometer reads as no movement', () => {
  const m = new Motion({ accDeadzoneMs2: 1.2 });
  const bias = [0.06, -0.04, 0.05]; // g, typical clone offsets
  let q = qIdentity();
  let worst = 0;
  for (let i = 0; i < 400; i++) {
    const t = i * 0.02;
    q = qFromAxisAngle([1, 0, 0], 0.8 * Math.sin(2 * Math.PI * t)); // ±45° twist, no travel
    const a = qRotate(qConj(q), [0, 0, 1]).map((x, k) => x + bias[k]);
    m.update(a, q, i ? 0.02 : 0);
    if (t > 1) worst = Math.max(worst, vLen(m.v));
  }
  assert.ok(worst < 0.05, `fake speed ${worst} m/s`);
});
