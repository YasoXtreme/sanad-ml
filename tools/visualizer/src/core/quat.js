// Small vector + quaternion helpers. Quaternions are [w, x, y, z] and rotate body-frame vectors
// into the world frame (world Z up): v_world = q * v_body * q^-1.

export const DEG = Math.PI / 180;

export const vAdd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vSub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vScale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
export const vDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vLen = (a) => Math.hypot(a[0], a[1], a[2]);
export const vCross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const vLerp = (a, b, t) => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];
export function vNormalize(a) {
  const n = vLen(a);
  return n > 0 ? vScale(a, 1 / n) : [0, 0, 0];
}

export const qIdentity = () => [1, 0, 0, 0];
export const qConj = (q) => [q[0], -q[1], -q[2], -q[3]];

export function qMul(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function qNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function qRotate(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

export function qFromAxisAngle(axis, angle) {
  const [x, y, z] = vNormalize(axis);
  const s = Math.sin(angle / 2);
  return [Math.cos(angle / 2), x * s, y * s, z * s];
}

// Shortest rotation taking direction a onto direction b.
export function qFromTwoVectors(a, b) {
  const an = vNormalize(a);
  const bn = vNormalize(b);
  const d = vDot(an, bn);
  if (d < -0.999999) {
    let axis = vCross([1, 0, 0], an);
    if (vLen(axis) < 1e-6) axis = vCross([0, 1, 0], an);
    return qFromAxisAngle(axis, Math.PI);
  }
  const c = vCross(an, bn);
  return qNormalize([1 + d, c[0], c[1], c[2]]);
}

// Rotate q by angular velocity w (rad/s) for dt seconds. Body-frame w: q * dq. World-frame w: dq * q.
export function qIntegrate(q, w, dt, worldFrame = false) {
  const n = vLen(w);
  if (n * dt < 1e-12) return q.slice();
  const dq = qFromAxisAngle(w, n * dt);
  return qNormalize(worldFrame ? qMul(dq, q) : qMul(q, dq));
}

export function qSlerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) {
    bb = [-b[0], -b[1], -b[2], -b[3]];
    d = -d;
  }
  if (d > 0.9995) {
    return qNormalize([0, 1, 2, 3].map((i) => a[i] + (bb[i] - a[i]) * t));
  }
  const th = Math.acos(d);
  const s = Math.sin(th);
  const ka = Math.sin((1 - t) * th) / s;
  const kb = Math.sin(t * th) / s;
  return [0, 1, 2, 3].map((i) => a[i] * ka + bb[i] * kb);
}

// Angle (rad) between two orientations.
export function qAngle(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, d));
}
