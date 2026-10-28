import assert from 'node:assert/strict';
import { test } from 'node:test';

import { mulberry32 } from '../src/core/demo.js';
import { Playback } from '../src/core/playback.js';

const sample = (t, seq = null) => ({ t, seq, a: [0, 0, 1], g: [0, 0, 0], ir: 0, red: 0 });

// Feed samples (arrival times from arrivalOf) and render frames at 60 fps; returns per-frame results.
function run(pb, { durMs, arrivalOf, skip = () => false }) {
  const samples = [];
  for (let t = 0; t <= durMs; t += 20) {
    if (!skip(t)) samples.push({ s: sample(t, t / 20), at: arrivalOf(t) });
  }
  samples.sort((x, y) => x.at - y.at);
  const frames = [];
  let i = 0;
  for (let now = 0; now <= durMs + 1000; now += 1000 / 60) {
    while (i < samples.length && samples[i].at <= now) pb.push(samples[i].s, samples[i++].at);
    frames.push({ now, ...pb.advance(now) });
  }
  return frames;
}

// Wi-Fi-like delivery: in order, 40-130 ms late.
function jittery(seed) {
  const r = mulberry32(seed);
  let last = -Infinity;
  return (t) => (last = Math.max(last, t + 40 + r() * 90));
}

test('jittered stream plays smoothly: monotonic, each sample once, in order, never starved', () => {
  const pb = new Playback();
  const frames = run(pb, { durMs: 10000, arrivalOf: jittery(1) });
  const seen = frames.flatMap((f) => f.process.map((s) => s.t));
  for (let k = 1; k < seen.length; k++) assert.equal(seen[k], seen[k - 1] + 20);
  let prev = -Infinity;
  for (const f of frames) {
    if (f.playhead === null) continue;
    assert.ok(f.playhead >= prev, 'playhead went backwards');
    prev = f.playhead;
    if (f.now > 1500 && f.now < 10000) assert.equal(f.starvedMs, 0, `starved at ${f.now}`);
  }
});

test('interpolation brackets the playhead', () => {
  const pb = new Playback();
  const frames = run(pb, { durMs: 3000, arrivalOf: jittery(2) });
  for (const f of frames.filter((x) => x.now > 500 && x.now < 3000)) {
    assert.ok(f.b, 'needs a next sample');
    assert.ok(f.a.t <= f.playhead && f.playhead <= f.b.t);
    assert.ok(Math.abs(f.a.t + f.frac * (f.b.t - f.a.t) - f.playhead) < 1e-6);
  }
});

test('a gap starves, then playback continues without jumping back', () => {
  const pb = new Playback();
  const frames = run(pb, {
    durMs: 4000,
    arrivalOf: (t) => t + 50,
    skip: (t) => t > 2000 && t < 2500,
  });
  assert.ok(frames.some((f) => f.starvedMs > 200), 'gap should starve');
  const last = frames.filter((f) => f.now > 3200 && f.now < 4000);
  assert.ok(last.every((f) => f.starvedMs === 0), 'recovers after the gap');
  assert.equal(pb.drops, 24);
});

test('a burst after a stall grows the delay, then it shrinks back', () => {
  const pb = new Playback();
  // 300 ms stall at t = 2 s: samples 2000..2300 all arrive at 2350.
  run(pb, { durMs: 3000, arrivalOf: (t) => (t >= 2000 && t <= 2300 ? 2350 : t + 50) });
  assert.ok(pb.delay >= 300, `delay ${pb.delay}`);
  const pb2 = new Playback();
  run(pb2, { durMs: 15000, arrivalOf: (t) => (t >= 2000 && t <= 2300 ? 2350 : t + 50) });
  assert.ok(pb2.delay < 200, `delay should shrink, is ${pb2.delay}`);
});

test('drops counted from seq, or from t_ms when there is no seq', () => {
  const pb = new Playback();
  pb.push(sample(0, 10), 0);
  pb.push(sample(20, 13), 20);
  assert.equal(pb.drops, 2);
  const pb2 = new Playback();
  pb2.push(sample(0), 0);
  pb2.push(sample(80), 80); // 20, 40, 60 missing
  assert.equal(pb2.drops, 3);
});

test('band reboot (time going back) restarts the buffer', () => {
  const pb = new Playback();
  pb.push(sample(100000), 0);
  assert.equal(pb.push(sample(100020), 20), false);
  assert.equal(pb.push(sample(100020), 21), null); // duplicate ignored
  assert.equal(pb.push(sample(40), 40), true);
  assert.equal(pb.buf.length, 1);
});

test('a hidden tab (no frames) does not pile up samples forever', () => {
  const pb = new Playback();
  for (let i = 0; i < 10000; i++) pb.push(sample(i * 20, i), i * 20);
  assert.ok(pb.buf.length <= 3000);
  const f = pb.advance(10000 * 20 + 150);
  assert.ok(f.process.length <= 3000 && f.a);
});

test('v0.2: two samples stamped in the same millisecond are both kept', () => {
  const pb = new Playback();
  pb.push(sample(1000, 50), 0);
  assert.equal(pb.push(sample(1000, 51), 1), false);
  assert.equal(pb.buf.length, 2);
  assert.ok(pb.buf[1].t > pb.buf[0].t);
  assert.equal(pb.drops, 0);
});
