// Sanad live visualizer: band stream -> jitter buffer -> filters -> breathing particles.

import { CONFIG } from './config.js';
import { Ahrs } from './core/ahrs.js';
import { HeartDetector } from './core/heart.js';
import { ImpactDetector, Motion, accMagnitude } from './core/motion.js';
import { HEADER, parseLine } from './core/parse.js';
import { Playback } from './core/playback.js';
import { qIntegrate, qRotate, qSlerp, vLerp, vNormalize, vScale } from './core/quat.js';
import { Spring, Spring3, breathCurve, clampLen, deadzone, pulseEnvelope } from './render/anim.js';
import { BandBlob } from './render/blob.js';
import { Dust } from './render/dust.js';
import { Rings } from './render/rings.js';
import { createScene } from './render/scene.js';
import { Transport } from './transport.js';
import { Ui } from './ui.js';

// ---------- data pipeline ----------
const pb = new Playback({ periodMs: CONFIG.periodMs, ...CONFIG.playback });
const ahrs = new Ahrs(CONFIG.ahrs);
const motion = new Motion(CONFIG.motion);
const impacts = new ImpactDetector(CONFIG.impact);
const heart = new HeartDetector(CONFIG.heart);
let events = []; // impacts and beats, fired when the playhead reaches them
let prevT = null;
let prevSeq = null;
let lastAccG = 1;
let recording = null; // array of raw lines while recording

function resetFilters() {
  ahrs.reset();
  motion.reset();
  impacts.reset();
  heart.reset();
  events = [];
  prevT = null;
  prevSeq = null;
}

function onLine(line, now) {
  const s = parseLine(line);
  if (!s) return;
  if (s.comment !== undefined) {
    ui.comment(s.comment);
    return;
  }
  const r = pb.push(s, now);
  if (r === null) return;
  if (r) resetFilters();
  if (recording) recording.push(line.trim());
  // Detect on the raw samples, before any smoothing.
  const ev = impacts.push(s);
  if (ev) events.push(ev);
  const beat = heart.push(s);
  if (beat) events.push(beat);
}

const transport = new Transport({
  onLine,
  onStatus: (st) => ui.setStatus(st),
  onRestart: () => {
    pb.reset();
    resetFilters();
  },
});

// ---------- scene ----------
const gl = createScene(document.getElementById('gl'), CONFIG);
const blob = new BandBlob(gl.scene, CONFIG);
const dust = new Dust(gl.scene, CONFIG);
gl.trackPixelRatio(blob.uniforms.uPixelRatio);
gl.trackPixelRatio(dust.uniforms.uPixelRatio);
const rings = new Rings(gl.scene, gl.camera, CONFIG);

const dimSpring = new Spring(0.55, 3);
const flashSpring = new Spring(0, 22);
const heartMix = new Spring(0, 2.5);
const E = CONFIG.elastic;
const offsetSpring = new Spring3(E.offsetHz, E.offsetDamping); // blob drifts off-centre with motion
const stretchSpring = new Spring3(E.stretchHz, E.stretchDamping); // jelly-like stretch along the motion
const spinSpring = new Spring3(4, 1);
let flashTarget = 0;
let dispQ = [1, 0, 0, 0];
let lastBeatWall = -1e9;
let lastWall = performance.now();

function fire(ev, nowWall) {
  if (ev.type === 'impact') {
    const k = Math.min(1, 0.35 + (ev.peakG - CONFIG.impact.thresholdG) / 3);
    flashTarget = Math.max(flashTarget, k);
    // Squash + wobble along the hit (gravity removed), and a small jolt of the whole blob.
    const hit = qRotate(dispQ, ev.a);
    hit[2] -= 1;
    const d = vNormalize(toThree(hit));
    stretchSpring.kick(vScale(d, E.impactKick * k));
    offsetSpring.kick(vScale(d, 1.5 * k));
    dust.burst(k, offsetSpring.x);
    rings.spawn('impact', k, offsetSpring.x);
  } else if (ev.type === 'beat' && heart.finger) {
    lastBeatWall = nowWall;
    rings.spawn('heart', 1, offsetSpring.x);
    ui.beat();
  }
}

// World (Z up) -> three.js (Y up)
const toThree = (v) => [v[0], v[2], -v[1]];

function frame(nowWall) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, Math.max(0, (nowWall - lastWall) / 1000));
  lastWall = nowWall;

  const f = pb.advance(nowWall);
  for (const s of f.process) {
    // With seq (v0.2) the MPU's own clock gives exact 20 ms steps; t_ms is only good to ~5 ms.
    let sdt = prevT === null ? 0 : (s.t - prevT) / 1000;
    if (prevSeq !== null && s.seq !== null && s.seq > prevSeq) sdt = ((s.seq - prevSeq) * CONFIG.periodMs) / 1000;
    prevT = s.t;
    prevSeq = s.seq;
    ahrs.update(s.a, s.g, sdt);
    motion.update(s.a, ahrs.q, sdt);
    s.q = ahrs.q;
    s.v = motion.v.slice();
    s.lin = motion.lin.slice();
    s.w = ahrs.wWorld;
    lastAccG = accMagnitude(s.a);
  }

  // Interpolate between the two samples around the playhead; on a gap, coast then ease to a stop.
  let target = dispQ;
  let v = [0, 0, 0];
  let w = [0, 0, 0];
  let lin = [0, 0, 0];
  const { extrapolateMs, holdEaseMs } = CONFIG.playback;
  if (f.a?.q) {
    if (f.b?.q) {
      target = qSlerp(f.a.q, f.b.q, f.frac);
      v = vLerp(f.a.v, f.b.v, f.frac);
      w = vLerp(f.a.w, f.b.w, f.frac);
      lin = vLerp(f.a.lin, f.b.lin, f.frac);
    } else {
      const coast = Math.min(f.starvedMs, extrapolateMs);
      const fade = f.starvedMs <= extrapolateMs ? 1 : Math.exp(-(f.starvedMs - extrapolateMs) / holdEaseMs);
      target = qIntegrate(f.a.q, f.a.w, coast / 1000, true);
      v = vScale(f.a.v, fade);
      w = vScale(f.a.w, fade);
    }
  }
  dispQ = qSlerp(dispQ, target, 1 - Math.exp(-dt / 0.045)); // light spring: hides tiny steps

  if (f.playhead !== null) {
    const keep = [];
    for (const ev of events) {
      if (ev.t <= f.playhead) fire(ev, nowWall);
      else keep.push(ev);
    }
    events = keep;
  }

  const live = transport.kind !== null && f.a !== null && f.starvedMs < 1000;
  const t = nowWall / 1000;
  const breath = breathCurve(t, CONFIG.breath.periodS, CONFIG.breath.inhaleS);
  const heartOn = heart.finger && heart.bpm !== null && nowWall - lastBeatWall < 2500;
  const mix = heartMix.update(heartOn ? 1 : 0, dt);
  const pulse = pulseEnvelope((nowWall - lastBeatWall) / 1000);
  const dim = dimSpring.update(live ? 1 : 0.55, dt);
  flashTarget *= Math.exp(-dt / 0.35);
  const flash = Math.max(0, flashSpring.update(flashTarget, dt));

  // Elastic body: drift toward the motion, stretch along it, lag/slosh with acceleration, flatten when spinning.
  const vT = toThree(v);
  const wT = toThree(w);
  const vE = deadzone(vT, E.speedDeadzone); // small movements do nothing
  const aE = clampLen(deadzone(toThree(lin), E.accDeadzone), 15);
  const wE = deadzone(wT, E.spinDeadzone);
  const offset = offsetSpring.update(clampLen(vScale(vE, E.offsetGain), E.offsetMax), dt);
  const ww = stretchSpring.w;
  const stretch = stretchSpring.update(
    clampLen(vScale(vE, E.stretchGain), E.stretchMax),
    dt,
    vScale(aE, -E.inertia * ww * ww),
  );
  const spin = spinSpring.update(clampLen(vScale(wE, E.spinGain), E.spinMax), dt);

  blob.update({
    time: t,
    breath: breath * (1 - mix),
    pulse: pulse * mix,
    flash,
    heartMix: mix,
    dim,
    q: dispQ,
    offset,
    stretch,
    spin,
  });
  dust.update(dt, { v: vT, w: wT, time: t, dim });
  rings.update(dt, dim);
  gl.bloom.strength =
    CONFIG.bloom.strength * (0.85 + 0.2 * breath * (1 - mix) + 0.25 * pulse * mix) + 0.7 * flash;
  gl.render(dt);

  updateStats(nowWall);
}

// ---------- HUD stats (4 per second) ----------
let statsAt = 0;
const counts = []; // [time, received] over the last 2 s, so the rate doesn't flicker with Wi-Fi bursts
function updateStats(now) {
  if (now - statsAt < 250) return;
  statsAt = now;
  counts.push([now, pb.received]);
  while (counts.length > 2 && now - counts[0][0] > 2000) counts.shift();
  const [t0, n0] = counts[0];
  const hz = now > t0 && pb.received >= n0 ? ((pb.received - n0) * 1000) / (now - t0) : 0;
  ui.setBpm(heart.finger, heart.bpm);
  if (!transport.kind) {
    ui.setStats('');
    return;
  }
  ui.setStats(
    `${hz.toFixed(0)} Hz · ${pb.delay.toFixed(0)} ms buffer · ${pb.drops} dropped · |a| ${lastAccG.toFixed(2)} g` +
      (transport.kind === 'demo' && transport.jitterOn ? ' · jitter sim on' : ''),
  );
}

// ---------- recording ----------
function toggleRecord() {
  if (!recording) {
    recording = [];
    ui.setRecording(true);
    return;
  }
  const lines = recording;
  recording = null;
  ui.setRecording(false);
  if (!lines.length) return;
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  const head = `# Sanad visualizer recording ${d.toISOString()} source=${transport.kind}\n# ${HEADER}\n`;
  const blobUrl = URL.createObjectURL(new Blob([head + lines.join('\n') + '\n'], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = `sanad-${stamp}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
}

// ---------- input ----------
const ui = new Ui({
  onWifi: (host) => transport.connectWifi(host),
  onUsb: () => transport.connectUsb(),
  onDemo: () => transport.startDemo(),
  onStop: () => transport.stop(),
});

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'f') {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => {});
  } else if (k === 'h') ui.toggleHidden();
  else if (k === 'r') ahrs.recenter();
  else if (k === 'c') toggleRecord();
  else if (k === ' ' && transport.demo) {
    e.preventDefault();
    transport.demo.triggerFall();
  } else if (k === 'b' && transport.demo) {
    ui.toast(transport.demo.toggleFinger() ? 'Demo: finger on the sensor' : 'Demo: finger off');
  } else if (k === 'j' && transport.kind === 'demo') {
    ui.toast(transport.toggleJitter() ? 'Demo: simulated Wi-Fi jitter ON' : 'Demo: jitter off');
  }
});

window.addEventListener('dragover', (e) => {
  e.preventDefault();
  document.body.classList.add('dragging');
});
window.addEventListener('dragleave', () => document.body.classList.remove('dragging'));
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  const file = e.dataTransfer?.files?.[0];
  if (file) transport.replay(await file.text(), file.name);
});

// for debugging from the console
window.sanad = { pb, ahrs, motion, heart, impacts, transport, CONFIG, offsetSpring, stretchSpring, gl };

ui.setStatus({ kind: null, state: 'idle', detail: 'Not connected' });
requestAnimationFrame(frame);
