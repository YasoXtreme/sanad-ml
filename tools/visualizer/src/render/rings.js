// Expanding light rings: rose for each heartbeat, ice-white shockwave for an impact.

import * as THREE from 'three';

import { easeOutCubic } from './anim.js';

const VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv * 2.0 - 1.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform float uAlpha, uWidth;
  varying vec2 vUv;
  void main() {
    float d = length(vUv);
    float ring = exp(-pow((d - 0.9) / uWidth, 2.0));
    float halo = 0.06 * exp(-pow((d - 0.9) / (uWidth * 3.0), 2.0));
    float a = (ring + halo) * uAlpha;
    if (a < 0.002) discard;
    gl_FragColor = vec4(uColor, a);
  }
`;

const KINDS = {
  heart: { dur: 1.5, from: 1.2, to: 3.0, width: 0.012, alpha: 0.3 },
  impact: { dur: 0.9, from: 1.0, to: 5.5, width: 0.022, alpha: 0.85 },
};

export class Rings {
  constructor(scene, camera, cfg, count = 6) {
    this.colors = { heart: new THREE.Color(cfg.palette.heart), impact: new THREE.Color(cfg.palette.impact) };
    this.rings = [];
    const geo = new THREE.PlaneGeometry(2, 2);
    for (let i = 0; i < count; i++) {
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color() },
          uAlpha: { value: 0 },
          uWidth: { value: 0.03 },
        },
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.visible = false;
      mesh.lookAt(camera.position); // the camera never moves: face it once
      scene.add(mesh);
      this.rings.push({ mesh, t: Infinity, kind: KINDS.heart, strength: 1 });
    }
    this.camera = camera;
  }

  spawn(kind, strength = 1, center = [0, 0, 0]) {
    let r = this.rings.find((x) => !x.mesh.visible);
    if (!r) r = this.rings.reduce((p, q) => (q.t > p.t ? q : p));
    r.t = 0;
    r.kind = KINDS[kind];
    r.strength = strength;
    r.mesh.material.uniforms.uColor.value.copy(this.colors[kind]);
    r.mesh.material.uniforms.uWidth.value = r.kind.width;
    r.mesh.position.set(...center); // rings start where the blob is now
    r.mesh.lookAt(this.camera.position);
    r.mesh.visible = true;
  }

  update(dt, dim) {
    for (const r of this.rings) {
      if (!r.mesh.visible) continue;
      r.t += dt;
      const u = r.t / r.kind.dur;
      if (u >= 1) {
        r.mesh.visible = false;
        continue;
      }
      const s = r.kind.from + (r.kind.to - r.kind.from) * easeOutCubic(u);
      r.mesh.scale.set(s, s, 1);
      r.mesh.material.uniforms.uAlpha.value = (1 - u) ** 1.6 * r.kind.alpha * r.strength * dim;
    }
  }
}
