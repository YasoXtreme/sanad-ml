// The breathing blob: a glowing particle body that IS the band.
//  - shell: a soft, slowly churning sphere. Its noise pattern is fixed in the band's frame, so turning
//    the band visibly turns the blob.
//  - ring: particles flowing around the band-face plane (chip x-y), so tilt reads at a glance.
//  - the +x side (forearm, toward the hand) is a little brighter.
//  - core: a soft glow sprite in the middle.
// Body axes are the chip's (Z up); a fixed -90 deg turn about X maps them onto three.js (Y up).

import * as THREE from 'three';

import { POINT_FRAGMENT } from './scene.js';

// 3D simplex noise: Ashima Arts / Stefan Gustavson, MIT licence (github.com/ashima/webgl-noise).
const SIMPLEX = /* glsl */ `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
  }
`;

const VERTEX = /* glsl */ `
  attribute vec4 aRand;
  attribute float aKind; // 0 shell, 1 ring, 2 inner
  uniform float uTime, uBreath, uPulse, uFlash, uHeartMix, uDim, uSize, uPixelRatio;
  uniform vec3 uDeep, uMid, uIce, uCore, uHeart;
  uniform vec3 uCenter; // where the blob is (it drifts with motion)
  uniform vec3 uStretch; // elastic stretch: direction * amount, scene axes
  uniform vec3 uSpin; // spin flattening: rotation axis * amount
  varying vec3 vColor;
  varying float vAlpha;
  ${SIMPLEX}

  // Elastic shape, applied in scene space so it follows the motion however the band is turned.
  // Stretch along u keeps the volume: longer along u, thinner across it. The trailing side stretches
  // more than the leading side, like a soft droplet. Spinning flattens it across the spin axis.
  vec3 deform(vec3 r) {
    float s = min(length(uStretch), 0.8);
    if (s > 1e-4) {
      vec3 u = normalize(uStretch);
      float along = dot(r, u);
      vec3 across = r - along * u;
      float k = 1.0 + s * (along > 0.0 ? 0.7 : 1.3);
      r = across / sqrt(1.0 + s) + u * along * k;
    }
    float sp = min(length(uSpin), 0.3);
    if (sp > 1e-4) {
      vec3 n = normalize(uSpin);
      float along = dot(r, n);
      r = (r - along * n) * (1.0 + 0.5 * sp) + n * along * (1.0 - sp);
    }
    return r;
  }

  void main() {
    vec3 dir = normalize(position);
    float swell = 1.0 + 0.10 * uBreath + 0.08 * uPulse + 0.22 * uFlash;
    float twinkle = 0.75 + 0.25 * sin(uTime * (0.8 + 2.0 * aRand.w) + aRand.y * 40.0);
    vec3 p;
    vec3 col;
    float a;
    if (aKind < 0.5) {
      float n = snoise(dir * 1.5 + vec3(0.0, 0.0, uTime * 0.11));
      float n2 = snoise(dir * 3.4 + vec3(uTime * 0.17, 0.0, 0.0));
      p = dir * (1.0 + 0.17 * n + 0.05 * n2 + (aRand.x - 0.5) * 0.10) * swell;
      vec3 nView = normalize(normalMatrix * dir);
      float rim = 1.0 - abs(nView.z);
      float forearm = smoothstep(0.35, 1.0, dir.x);
      col = mix(uDeep, uMid, smoothstep(0.0, 0.6, rim));
      col = mix(col, uIce, clamp(smoothstep(0.55, 1.0, rim) * 0.8 + 0.4 * forearm, 0.0, 1.0));
      a = (0.55 + 0.45 * smoothstep(-0.2, 0.8, n)) * (0.3 + 0.7 * rim) * (1.0 + 0.5 * forearm);
    } else if (aKind < 1.5) {
      float ang = aRand.y * 6.2831853 + uTime * (0.22 + 0.14 * aRand.z);
      float rr = 1.5 + (aRand.x - 0.5) * 0.12 + 0.04 * sin(ang * 3.0 + uTime * 0.7);
      float rs = 1.0 + 0.06 * uBreath + 0.05 * uPulse + 0.30 * uFlash;
      p = vec3(cos(ang) * rr, sin(ang) * rr, (aRand.w - 0.5) * 0.06) * rs;
      col = mix(uMid, uIce, 0.55 + 0.45 * smoothstep(0.3, 1.0, cos(ang)));
      a = 0.5;
    } else {
      float n = snoise(dir * 2.0 + uTime * 0.08);
      p = dir * (0.25 + 0.6 * aRand.x) * (1.0 + 0.15 * n) * swell;
      col = mix(uMid, uIce, 0.4);
      a = 0.12;
    }
    col = mix(col, uHeart, uHeartMix * uPulse * 0.3);
    col = mix(col, uCore, uFlash * 0.5);
    vColor = col;
    vAlpha = a * twinkle * (0.75 + 0.35 * uBreath + 0.45 * uPulse + 0.35 * uFlash) * uDim;

    vec4 wp = modelMatrix * vec4(p, 1.0);
    wp.xyz = uCenter + deform(wp.xyz - uCenter);
    vec4 mv = viewMatrix * wp;
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * (0.55 + 0.9 * aRand.z) * (1.0 + 0.3 * uPulse) * uPixelRatio / -mv.z;
  }
`;

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.35)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export class BandBlob {
  constructor(scene, cfg) {
    const { shellCount, ringCount, size } = cfg.blob;
    const pal = cfg.palette;
    const n = shellCount + ringCount;
    const inner = Math.round(shellCount * 0.12);
    const pos = new Float32Array(n * 3);
    const rand = new Float32Array(n * 4);
    const kind = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // uniform direction on the sphere
      const z = Math.random() * 2 - 1;
      const t = Math.random() * Math.PI * 2;
      const r = Math.sqrt(1 - z * z);
      pos.set([r * Math.cos(t), r * Math.sin(t), z], i * 3);
      rand.set([Math.random(), Math.random(), Math.random(), Math.random()], i * 4);
      kind[i] = i < shellCount - inner ? 0 : i < shellCount ? 2 : 1;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aRand', new THREE.BufferAttribute(rand, 4));
    geo.setAttribute('aKind', new THREE.BufferAttribute(kind, 1));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);

    const col = (h) => new THREE.Color(h);
    this.uniforms = {
      uTime: { value: 0 },
      uBreath: { value: 0 },
      uPulse: { value: 0 },
      uFlash: { value: 0 },
      uHeartMix: { value: 0 },
      uDim: { value: 1 },
      uSize: { value: size },
      uPixelRatio: { value: 1 },
      uDeep: { value: col(pal.deep) },
      uMid: { value: col(pal.mid) },
      uIce: { value: col(pal.ice) },
      uCore: { value: col(pal.core) },
      uHeart: { value: col(pal.heart) },
      uCenter: { value: new THREE.Vector3() },
      uStretch: { value: new THREE.Vector3() },
      uSpin: { value: new THREE.Vector3() },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: VERTEX,
      fragmentShader: POINT_FRAGMENT,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.frame = new THREE.Group(); // band orientation
    const body = new THREE.Group();
    body.rotation.x = -Math.PI / 2; // chip (x, y, z) -> three (x, z, -y)
    const points = new THREE.Points(geo, mat);
    points.frustumCulled = false; // the shader moves particles, so three's bounds don't apply
    body.add(points);
    this.frame.add(body);
    scene.add(this.frame);

    this.core = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: glowTexture(),
        color: col(pal.ice),
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    scene.add(this.core);
    this.coreColor = { ice: col(pal.ice), heart: col(pal.heart), white: col(pal.core) };
  }

  // q: band orientation [w, x, y, z], world Z up. offset / stretch / spin: scene-axis vectors.
  update({ time, breath, pulse, flash, heartMix, dim, q, offset, stretch, spin }) {
    const u = this.uniforms;
    u.uTime.value = time;
    u.uBreath.value = breath;
    u.uPulse.value = pulse;
    u.uFlash.value = flash;
    u.uHeartMix.value = heartMix;
    u.uDim.value = dim;
    // World Z-up -> three Y-up: rotation axis (x, y, z) becomes (x, z, -y).
    this.frame.quaternion.set(q[1], q[3], -q[2], q[0]);
    this.frame.position.set(...offset);
    u.uCenter.value.set(...offset);
    u.uStretch.value.set(...stretch);
    u.uSpin.value.set(...spin);

    // The core glow follows and stretches too (on screen: the camera looks almost straight on).
    const s = 2.2 * (1 + 0.12 * breath + 0.1 * pulse + 0.5 * flash);
    const st = Math.min(Math.hypot(stretch[0], stretch[1]), 0.8);
    this.core.position.set(...offset);
    this.core.material.rotation = Math.atan2(stretch[1], stretch[0]);
    this.core.scale.set(s * (1 + st), s / Math.sqrt(1 + st), 1);
    this.core.material.opacity = (0.1 + 0.06 * breath + 0.14 * pulse + 0.4 * flash) * dim;
    this.core.material.color
      .copy(this.coreColor.ice)
      .lerp(this.coreColor.heart, heartMix * pulse * 0.5)
      .lerp(this.coreColor.white, Math.min(1, flash));
  }
}
