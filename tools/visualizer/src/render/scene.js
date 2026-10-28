// Renderer, camera and the bloom pipeline. Pure black background (OLED).

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

export function createScene(canvas, cfg) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  let pixelRatio = dpr();
  renderer.setPixelRatio(pixelRatio);
  const prUniforms = []; // point-size uniforms that must follow the screen scale
  renderer.setClearColor(0x000000, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(256, 256),
    cfg.bloom.strength,
    cfg.bloom.radius,
    cfg.bloom.threshold,
  );
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  let sized = [0, 0, 0];
  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (!w || !h) return; // hidden tab / collapsed pane: keep the last size
    // The screen scale can change after load (projector plugged in, browser zoom, window moved to
    // another monitor): renderer, bloom buffers and point sizes must all follow it.
    pixelRatio = dpr();
    sized = [w, h, pixelRatio];
    renderer.setPixelRatio(pixelRatio);
    for (const u of prUniforms) u.value = pixelRatio;
    renderer.setSize(w, h, false);
    composer.setPixelRatio(pixelRatio);
    composer.setSize(w, h);
    camera.aspect = w / h;
    // Portrait screens: step back so the blob and its ring still fit.
    const dist = 7.4 * (camera.aspect < 1 ? (1 / camera.aspect) ** 0.85 : 1);
    camera.position.set(0, dist * 0.22, dist);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  // Renders unless the page has no size right now (zero-size framebuffers make WebGL complain).
  function render(dt) {
    if (!window.innerWidth || !window.innerHeight) return;
    if (window.innerWidth !== sized[0] || window.innerHeight !== sized[1] || dpr() !== sized[2]) resize();
    composer.render(dt);
  }

  // Register a {value} uniform that holds the pixel ratio (for point sizes).
  function trackPixelRatio(uniform) {
    uniform.value = pixelRatio;
    prUniforms.push(uniform);
  }

  return { renderer, scene, camera, composer, bloom, render, trackPixelRatio };
}

// Shared soft round sprite for point particles.
export const POINT_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float a = pow(1.0 - d * 2.0, 1.8);
    gl_FragColor = vec4(vColor, a * vAlpha);
  }
`;
