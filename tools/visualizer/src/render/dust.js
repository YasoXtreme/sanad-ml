// Ambient dust around the blob: shows the last ~second of the band's motion.
// The blob stays centred, so the world streams past it: dust flows opposite to the band's velocity
// and counter-rotates with the gyro, leaving short trails. Each grain lives ~1-2 s and fades.
// An impact blows the dust outward.

import * as THREE from 'three';

import { POINT_FRAGMENT } from './scene.js';

const POINT_VERTEX = /* glsl */ `
  attribute float aAlpha;
  attribute float aSize;
  uniform float uSize, uPixelRatio;
  uniform vec3 uColor;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * aSize * uPixelRatio / -mv.z;
    vColor = uColor;
    vAlpha = aAlpha;
  }
`;

const R_MIN = 1.9;
const R_MAX = 6.5;

export class Dust {
  constructor(scene, cfg) {
    this.cfg = cfg.dust;
    const n = this.cfg.count;
    this.n = n;
    this.pos = new Float32Array(n * 3);
    this.burstV = new Float32Array(n * 3);
    this.age = new Float32Array(n);
    this.life = new Float32Array(n);
    this.alpha = new Float32Array(n);
    this.size = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.spawn(i);
      this.age[i] = Math.random() * this.life[i]; // don't all start together
    }

    const pg = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.pos, 3);
    this.alphaAttr = new THREE.BufferAttribute(this.alpha, 1);
    pg.setAttribute('position', this.posAttr);
    pg.setAttribute('aAlpha', this.alphaAttr);
    pg.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));
    pg.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 20);
    this.color = new THREE.Color(cfg.palette.dust);
    this.uniforms = {
      uSize: { value: this.cfg.size },
      uPixelRatio: { value: 1 },
      uColor: { value: this.color },
    };
    const pm = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: POINT_VERTEX,
      fragmentShader: POINT_FRAGMENT,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    scene.add(new THREE.Points(pg, pm));

    // Trails: one segment per grain, bright head -> dark tail.
    this.linePos = new Float32Array(n * 6);
    this.lineCol = new Float32Array(n * 6);
    const lg = new THREE.BufferGeometry();
    this.linePosAttr = new THREE.BufferAttribute(this.linePos, 3);
    this.lineColAttr = new THREE.BufferAttribute(this.lineCol, 3);
    lg.setAttribute('position', this.linePosAttr);
    lg.setAttribute('color', this.lineColAttr);
    lg.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 20);
    const lm = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    scene.add(new THREE.LineSegments(lg, lm));
  }

  spawn(i, rMin = R_MIN, rMax = R_MAX) {
    const z = Math.random() * 2 - 1;
    const t = Math.random() * Math.PI * 2;
    const s = Math.sqrt(1 - z * z);
    const r = rMin + (rMax - rMin) * Math.cbrt(Math.random());
    this.pos[i * 3] = r * s * Math.cos(t);
    this.pos[i * 3 + 1] = r * z;
    this.pos[i * 3 + 2] = r * s * Math.sin(t);
    this.burstV[i * 3] = this.burstV[i * 3 + 1] = this.burstV[i * 3 + 2] = 0;
    this.age[i] = 0;
    this.life[i] = 1.0 + Math.random() * 1.2;
    this.size[i] = 0.5 + Math.random();
  }

  // Impact: blow everything outward from the blob (at `center`); strength 0..1.
  burst(strength, center = [0, 0, 0]) {
    for (let i = 0; i < this.n; i++) {
      if (Math.random() < 0.45) {
        this.spawn(i, 1.4, 2.4); // a puff from the blob's surface
        this.pos[i * 3] += center[0];
        this.pos[i * 3 + 1] += center[1];
        this.pos[i * 3 + 2] += center[2];
        this.age[i] = 0.15;
      }
      const x = this.pos[i * 3] - center[0];
      const y = this.pos[i * 3 + 1] - center[1];
      const z = this.pos[i * 3 + 2] - center[2];
      const r = Math.hypot(x, y, z) || 1;
      const k = (strength * (5 + 7 * Math.random())) / r;
      this.burstV[i * 3] += x * k;
      this.burstV[i * 3 + 1] += y * k;
      this.burstV[i * 3 + 2] += z * k;
    }
  }

  // v: band velocity (m/s), w: band angular velocity (rad/s), both in three.js axes.
  update(dt, { v, w, time, dim }) {
    const { transGain, rotGain, trailS } = this.cfg;
    const flowX = -v[0] * transGain;
    const flowY = -v[1] * transGain;
    const flowZ = -v[2] * transGain;
    const wx = -w[0] * rotGain;
    const wy = -w[1] * rotGain;
    const wz = -w[2] * rotGain;
    const burstDecay = Math.exp(-dt / 0.6);
    const c = this.color;
    for (let i = 0; i < this.n; i++) {
      const j = i * 3;
      let x = this.pos[j];
      let y = this.pos[j + 1];
      let z = this.pos[j + 2];
      // flow = translation + counter-rotation + slow ambient drift + impact burst
      let ux = flowX + (wy * z - wz * y) + 0.1 * Math.sin(0.6 * y + 0.31 * time) - 0.04 * z;
      let uy = flowY + (wz * x - wx * z) + 0.1 * Math.sin(0.6 * z + 0.23 * time + 1);
      let uz = flowZ + (wx * y - wy * x) + 0.1 * Math.sin(0.6 * x + 0.27 * time + 2) + 0.04 * x;
      ux += this.burstV[j];
      uy += this.burstV[j + 1];
      uz += this.burstV[j + 2];
      this.burstV[j] *= burstDecay;
      this.burstV[j + 1] *= burstDecay;
      this.burstV[j + 2] *= burstDecay;

      x += ux * dt;
      y += uy * dt;
      z += uz * dt;
      this.pos[j] = x;
      this.pos[j + 1] = y;
      this.pos[j + 2] = z;
      this.age[i] += dt;
      const r = Math.hypot(x, y, z);
      if (this.age[i] > this.life[i] || r < 1.2 || r > 9) {
        this.spawn(i);
        x = this.pos[j];
        y = this.pos[j + 1];
        z = this.pos[j + 2];
        ux = uy = uz = 0;
      }

      const age = this.age[i];
      const life = this.life[i];
      const fade = Math.min(1, age / 0.3) * Math.min(1, Math.max(0, (life - age) / 0.5));
      const speed = Math.hypot(ux, uy, uz);
      const a = fade * (0.22 + 0.78 * Math.min(1, speed / 1.5)) * dim;
      this.alpha[i] = a;

      // trail: head at the grain, tail where it was ~trailS ago (capped length)
      const k = Math.min(trailS, 1.2 / Math.max(speed, 1e-3));
      const l = i * 6;
      this.linePos[l] = x;
      this.linePos[l + 1] = y;
      this.linePos[l + 2] = z;
      this.linePos[l + 3] = x - ux * k;
      this.linePos[l + 4] = y - uy * k;
      this.linePos[l + 5] = z - uz * k;
      const la = a * 0.55;
      this.lineCol[l] = c.r * la;
      this.lineCol[l + 1] = c.g * la;
      this.lineCol[l + 2] = c.b * la;
      this.lineCol[l + 3] = this.lineCol[l + 4] = this.lineCol[l + 5] = 0;
    }
    this.posAttr.needsUpdate = true;
    this.alphaAttr.needsUpdate = true;
    this.linePosAttr.needsUpdate = true;
    this.lineColAttr.needsUpdate = true;
  }
}
