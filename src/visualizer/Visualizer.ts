import * as THREE from "three";
import {
  GPUComputationRenderer,
  type Variable,
} from "three/examples/jsm/misc/GPUComputationRenderer.js";
import { SIMPLEX3D, CURL, HASH, VNOISE } from "./glsl";
import type { FlowPlayer } from "../audio/flowmap";

// Particle count = TEX * TEX. 512² ≈ 262k. Drop to 256 on weak GPUs.
const TEX = 512;
const TAU = Math.PI * 2;

// Kaleidoscope families, cycled with the 'K' key.
//  kind: 0 off · 1 radial · 2 grid · 3 polar · 4 multipoint · 5 orbital
// NOTE the two decoupled axes:
//   segPerPoint = how many MIRROR reflections each center makes (the "kaleidoscope" count)
//   points/orbit*/drift*/bubbleRate = how the centers MOVE (motion, independent of mirrors)
type KaleidoMode = {
  name: string;
  kind: number;
  segments: number; // radial segs / grid tiles / polar segs
  rings: number; // polar radial mirror count
  points: number; // multipoint/orbital: number of moving mirror centers
  bubbleRate: number; // fade-in/out + respawn rate
  driftAmt: number; // perlin drift / jitter amount
  driftSpeed: number; // drift evolution rate
  rotSpeed: number; // per-center rotation rate
  segPerPoint: number; // MIRROR count per center (decoupled from motion)
  blendSharp: number; // voronoi hardness
  orbitRadius: number; // orbital: how far satellites sit from center
  orbitSpeed: number; // orbital: angular swirl speed
  centerPull: number; // orbital: gravity/weight of the center anchor (0 = none)
  bounceSpeed: number; // bounce: base travel speed of edge-ricocheting centers
  reactive: number; // 1 = music intensity scales the number of active centers
  auto?: "cycle" | "music" | "symmetry" | "surface"; // special: auto-choreography driver (no own geometry)
};
const K0 = { segments: 0, rings: 0, points: 0, bubbleRate: 0, driftAmt: 0, driftSpeed: 0, rotSpeed: 0, segPerPoint: 6, blendSharp: 8, orbitRadius: 0, orbitSpeed: 0, centerPull: 0, bounceSpeed: 0, reactive: 0 };
const KALEIDO_MODES: KaleidoMode[] = [
  // ── auto-choreography leads: these are the headline modes; symmetry is default ──
  { name: "surface", kind: -1, ...K0, auto: "surface" },
  { name: "auto · symmetry", kind: -1, ...K0, auto: "symmetry" },
  { name: "auto · music", kind: -1, ...K0, auto: "music" },
  { name: "auto · flow", kind: -1, ...K0, auto: "cycle" },
  { name: "off", kind: 0, ...K0 },
  { name: "radial 8×", kind: 1, ...K0, segments: 8 },
  { name: "radial 12×", kind: 1, ...K0, segments: 12 },
  { name: "grid 3×", kind: 2, ...K0, segments: 3 },
  { name: "polar 6×", kind: 3, ...K0, segments: 6, rings: 3 },
  // ── multipoint "bubbling moving mirrors" — scattered centers ──
  { name: "bubble 3 · drift", kind: 4, ...K0, points: 3, bubbleRate: 0.05, driftAmt: 0.13, driftSpeed: 0.08, rotSpeed: 0.06, segPerPoint: 6, blendSharp: 6 },
  { name: "bubble 5 · churn", kind: 4, ...K0, points: 5, bubbleRate: 0.10, driftAmt: 0.18, driftSpeed: 0.16, rotSpeed: 0.12, segPerPoint: 5, blendSharp: 9 },
  { name: "swirl 4 · evolve", kind: 4, ...K0, points: 4, bubbleRate: 0.07, driftAmt: 0.22, driftSpeed: 0.12, rotSpeed: 0.30, segPerPoint: 8, blendSharp: 5 },
  // ── orbital: center gravity + satellites swirling/bubbling around it ──
  // (mirror count varied independently of orbit/drift to show the two axes)
  { name: "orbit calm 6×", kind: 5, ...K0, points: 3, segPerPoint: 6, bubbleRate: 0.06, driftAmt: 0.04, driftSpeed: 0.10, rotSpeed: 0.05, blendSharp: 7, orbitRadius: 0.26, orbitSpeed: 0.10, centerPull: 1.4 },
  { name: "orbit swirl 8×", kind: 5, ...K0, points: 4, segPerPoint: 8, bubbleRate: 0.09, driftAmt: 0.06, driftSpeed: 0.16, rotSpeed: 0.10, blendSharp: 8, orbitRadius: 0.30, orbitSpeed: 0.28, centerPull: 1.0 },
  { name: "orbit chaos 5×", kind: 5, ...K0, points: 5, segPerPoint: 5, bubbleRate: 0.13, driftAmt: 0.12, driftSpeed: 0.22, rotSpeed: 0.18, blendSharp: 9, orbitRadius: 0.34, orbitSpeed: 0.40, centerPull: 0.7 },
  { name: "orbit bloom 12×", kind: 5, ...K0, points: 3, segPerPoint: 12, bubbleRate: 0.05, driftAmt: 0.05, driftSpeed: 0.08, rotSpeed: 0.04, blendSharp: 6, orbitRadius: 0.18, orbitSpeed: 0.08, centerPull: 1.8 },
  // ── bounce: swirl centers ricocheting off the screen edges, varied speeds ──
  { name: "bounce 2 · slow", kind: 6, ...K0, points: 2, segPerPoint: 6, rotSpeed: 0.05, blendSharp: 7, bounceSpeed: 0.06 },
  { name: "bounce 3 · mixed", kind: 6, ...K0, points: 3, segPerPoint: 8, rotSpeed: 0.08, blendSharp: 8, bounceSpeed: 0.11 },
  { name: "bounce 3 · fast", kind: 6, ...K0, points: 3, segPerPoint: 5, rotSpeed: 0.12, blendSharp: 9, bounceSpeed: 0.22 },
  { name: "bounce 2 · swirl", kind: 6, ...K0, points: 2, segPerPoint: 12, rotSpeed: 0.28, blendSharp: 6, bounceSpeed: 0.09 },
  // ── swarm: bounce centers whose COUNT tracks music intensity ──
  // crescendo → up to `points` swirls fill the screen; mellow → collapses to ~1.
  { name: "swarm 6 · slow", kind: 6, ...K0, points: 6, segPerPoint: 6, rotSpeed: 0.05, blendSharp: 7, bounceSpeed: 0.06, reactive: 1 },
  { name: "swarm 6 · mixed", kind: 6, ...K0, points: 6, segPerPoint: 8, rotSpeed: 0.08, blendSharp: 8, bounceSpeed: 0.11, reactive: 1 },
  { name: "swarm 6 · fast", kind: 6, ...K0, points: 6, segPerPoint: 5, rotSpeed: 0.12, blendSharp: 9, bounceSpeed: 0.20, reactive: 1 },
  { name: "swarm 5 · swirl", kind: 6, ...K0, points: 5, segPerPoint: 12, rotSpeed: 0.26, blendSharp: 6, bounceSpeed: 0.09, reactive: 1 },
  // ── symmetric: guaranteed N-fold dihedral; motion = spin + radius-swirl +
  //    radial breathe. seg=N, rotSpeed=spin, driftAmt=swirl, orbitRadius=breatheAmt,
  //    driftSpeed=breatheSpeed. No scattered centers, so always symmetrical. ──
  { name: "sym mirror 6×", kind: 7, ...K0, segments: 6, rotSpeed: 0.05 },
  { name: "sym orbit 8×", kind: 7, ...K0, segments: 8, rotSpeed: 0.10, driftAmt: 2.6, orbitRadius: 0.05, driftSpeed: 0.6 },
  { name: "sym balance 8×", kind: 7, ...K0, segments: 8, rotSpeed: 0.05, driftAmt: 1.2, orbitRadius: 0.16, driftSpeed: 0.45 },
  // symmetric multi-center: focal points split apart while staying mirror-symmetric
  { name: "sym split 2×", kind: 8, ...K0, points: 2, segPerPoint: 5, rotSpeed: 0.10, orbitRadius: 0.14, driftAmt: 0.22, orbitSpeed: 0.11 },
  { name: "sym split 4×", kind: 8, ...K0, points: 4, segPerPoint: 5, rotSpeed: 0.08, orbitRadius: 0.13, driftAmt: 0.18, orbitSpeed: 0.11 },
  // unified symMorph space (kind 9) — the morph-friendly auto·symmetry family.
  // slot map: splitX=orbitRadius, splitY=centerPull, spin=rotSpeed, swirl=driftAmt, moveAmp=bubbleRate, moveSpeed=orbitSpeed
  // kind-9 slots: points=numPoints, segments=localSeg, orbitRadius=focusR, rotSpeed=spin, driftAmt=swirl
  { name: "morph · two", kind: 9, ...K0, points: 2, segments: 6, orbitRadius: 0.24, rotSpeed: 0.09, driftAmt: 2.0 },
  { name: "morph · triangle", kind: 9, ...K0, points: 3, segments: 6, orbitRadius: 0.26, rotSpeed: 0.10, driftAmt: 2.4 },
  { name: "morph · eight", kind: 9, ...K0, points: 8, segments: 4, orbitRadius: 0.20, rotSpeed: 0.12, driftAmt: 2.8 },
];

// quick constructor for ad-hoc presets (low mirror counts not in the K cycle)
const km = (name: string, over: Partial<KaleidoMode> & { kind: number }): KaleidoMode => ({
  name,
  ...K0,
  ...over,
});
const findMode = (n: string): KaleidoMode => KALEIDO_MODES.find((m) => m.name === n)!;

// AUTO "flow": a curated wander — simple → complex → simple, looped. The list
// rises from a single/two-mirror look up through orbit/swirl/bounce, then eases
// back down, so even without music it feels like a natural progression.
const AUTO_PROGRAM: KaleidoMode[] = [
  km("two mirror", { kind: 1, segments: 2 }),
  km("single", { kind: 1, segments: 1 }),
  km("two mirror", { kind: 1, segments: 2 }),
  km("four mirror", { kind: 1, segments: 4 }),
  findMode("polar 6×"),
  findMode("orbit calm 6×"),
  findMode("orbit swirl 8×"),
  findMode("bounce 3 · mixed"),
  findMode("orbit chaos 5×"),
  findMode("swirl 4 · evolve"),
  findMode("orbit swirl 8×"),
  findMode("orbit calm 6×"),
  findMode("polar 6×"),
  km("four mirror", { kind: 1, segments: 4 }),
  km("two mirror", { kind: 1, segments: 2 }),
];

// AUTO "music": an energy ladder, simplest → busiest. Smoothed track energy
// picks the rung (one step at a time, with hysteresis) so quiet = few mirrors,
// crescendo = orbiting/swarming. Top rung is reactive (swirl count tracks volume).
const AUTO_LADDER: KaleidoMode[] = [
  km("single", { kind: 1, segments: 1 }),
  km("two mirror", { kind: 1, segments: 2 }),
  km("four mirror", { kind: 1, segments: 4 }),
  findMode("polar 6×"),
  findMode("orbit calm 6×"),
  findMode("orbit swirl 8×"),
  findMode("swarm 6 · mixed"),
];

// AUTO "symmetry" is a GENERATIVE journey (not fixed keyframes). It breathes
// between two pose types — CENTER (all points consolidated at the middle, a
// calm pause) and BLOOM (points spread into an N-fold ring that spins/circles).
// Each time it returns to center it rolls a NEW random bloom: a different point
// count N (incl. 3 → triangle), radius, spin direction, swirl, local fold, and
// dwell — so it explores 2/3/4/6/8/16 and never repeats. Because focusR=0 at
// center collapses all N to one spot, N can change invisibly while consolidated.
//
// A "pose" is the target the journey eases toward; SymPose holds the kind-9
// params. numPoints = focal points in the ring; localSeg = each region's own
// kaleido fold (regional symmetry); circle = rigid-body rotation (turns/sec).
type SymPose = {
  numPoints: number;
  focusR: number;
  spin: number;
  swirl: number;
  circle: number;
  localSeg: number;
};
// Allowed point counts for blooms — even spacing guaranteed for any N; 3 gives a
// triangle. Weighted toward the prettier low counts but reaching up to 16.
const SYM_POINT_COUNTS = [2, 3, 4, 4, 6, 8, 16];

// SURFACE (water) mode pose: N swirl SOURCES on a ring in the flow field. The
// field is the sum of N identical evenly-spaced sources, so it is exactly N-fold
// symmetric and particles flow through it (no post-process mirror). radius=0
// pulls all sources to the center (consolidated) — where N can change invisibly.
type SurfPose = {
  numPoints: number;
  radius: number; // ring radius of the sources
  swirl: number; // tangential strength
  pull: number; // inward pull strength
  falloff: number; // locality of each source
};
const SURF_POINT_COUNTS = [2, 3, 4, 4, 5, 6, 8];

// Cosine-gradient palettes (Inigo Quilez form): color(t) = a + b*cos(2π(c*t + d)).
type Palette = {
  name: string;
  a: [number, number, number];
  b: [number, number, number];
  c: [number, number, number];
  d: [number, number, number];
};
const PALETTES: Palette[] = [
  { name: "Spectrum", a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.33, 0.67] },
  { name: "Aurora", a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 0.5], d: [0.8, 0.9, 0.3] },
  { name: "Ice", a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.1, 0.2] },
  { name: "Ember", a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.3, 0.2, 0.2] },
  { name: "Lagoon", a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 0.7, 0.4], d: [0.0, 0.15, 0.2] },
  { name: "Magma", a: [0.8, 0.5, 0.4], b: [0.2, 0.4, 0.2], c: [2, 1, 1], d: [0.0, 0.25, 0.25] },
];

const velocityFrag = /* glsl */ `
  uniform float uTime;
  uniform float uFieldScale;
  uniform float uFieldSpeed;
  uniform float uInertia;
  uniform float uCurlStrength;
  // ── surface (water) mode: symmetry lives in the FORCE FIELD ──
  uniform float uSurface;     // 0 = legacy curl, 1 = surface flow
  uniform float uSrcCount;    // N swirl sources on the ring (1..16)
  uniform float uSrcRadius;   // ring radius of the sources (0 = all at center)
  uniform float uSrcRot;      // accumulated ring rotation (radians)
  uniform float uSrcSwirl;    // tangential (swirl) strength per source
  uniform float uSrcPull;     // inward pull strength per source
  uniform float uSrcFalloff;  // how tightly each source's influence is localized
  uniform float uUpdraft;     // vertical lift near a source (bubbling up)
  uniform float uSettle;      // pull of z back toward the surface plane
  uniform float uSwirlNoise;  // small symmetric noise for organic wobble

  ${SIMPLEX3D}
  ${CURL}

  const float TAU_V = 6.28318530718;

  void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec3 pos = texture2D(texturePosition, uv).xyz;
    vec3 vel = texture2D(textureVelocity, uv).xyz;

    if (uSurface > 0.5) {
      // Sum of N identical, evenly-spaced swirl sources. Rotating space by
      // TAU/N permutes the sources, leaving this sum invariant → the field is
      // EXACTLY N-fold rotationally symmetric, so particles flow symmetrically
      // without any mirror fold. Each source swirls (tangential) + draws inward.
      vec2 acc = vec2(0.0);
      float fall = 0.0;
      float N = max(uSrcCount, 1.0);
      for (int k = 0; k < 16; k++) {
        if (float(k) >= N) break;
        float ang = uSrcRot + TAU_V * float(k) / N;
        vec2 sc = uSrcRadius * vec2(cos(ang), sin(ang));
        vec2 d = pos.xy - sc;
        float dist = length(d) + 1e-3;
        float f = exp(-dist * uSrcFalloff);
        vec2 tang = vec2(-d.y, d.x) / dist;   // swirl around source
        vec2 rad  = -d / dist;                // pull toward source
        acc += (tang * uSrcSwirl + rad * uSrcPull) * f;
        fall += f;
      }
      acc += -pos.xy * 0.15;                  // gentle framing pull to center
      // vertical: rise near sources (bubble up), relax back toward the surface
      float vz = uUpdraft * fall - uSettle * pos.z;
      vec3 target = vec3(acc, vz);
      vel = mix(target, vel, uInertia);
      gl_FragColor = vec4(vel, 1.0);
      return;
    }

    vec3 field = curlNoise(pos * uFieldScale + vec3(0.0, 0.0, uTime * uFieldSpeed));
    field *= uCurlStrength;
    // gentle pull toward the origin so the swarm stays framed
    field += -pos * 0.18;

    vel = mix(field, vel, uInertia);
    gl_FragColor = vec4(vel, 1.0);
  }
`;

const positionFrag = /* glsl */ `
  uniform float uTime;
  uniform float uDt;
  uniform float uSpeed;
  uniform float uLifeDecay;

  ${HASH}

  void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec4 posLife = texture2D(texturePosition, uv);
    vec3 pos = posLife.xyz;
    float life = posLife.w;
    vec3 vel = texture2D(textureVelocity, uv).xyz;

    pos += vel * uDt * uSpeed;
    // each particle has its own lifespan (stable per texel) → staggered death
    float lifeDecay = uLifeDecay * (0.45 + hash12(uv * 7.13) * 1.6);
    life -= uDt * lifeDecay;

    if (life <= 0.0 || length(pos) > 3.4) {
      float t = uTime * 0.37;
      vec3 r = vec3(
        hash12(uv + t),
        hash12(uv * 1.7 + t + 11.3),
        hash12(uv * 2.3 + t + 71.9)
      );
      pos = (r - 0.5) * 2.6;
      life = 1.0; // reborn at full life; fades in via the size/alpha envelope
    }

    gl_FragColor = vec4(pos, life);
  }
`;

const particleVert = /* glsl */ `
  uniform sampler2D texturePosition;
  uniform sampler2D textureVelocity;
  uniform float uSizeMin;
  uniform float uSizeMax;
  uniform float uSizeSpeed;
  uniform float uDpr;
  uniform float uTime;
  uniform float uSurfaceDraw; // 1 = depth-fade (water surface), 0 = life envelope
  uniform float uSurfaceBand; // half-thickness of the bright surface slab
  attribute vec2 ref;
  varying float vSpeed;
  varying float vLife;
  varying float vSeed;
  varying float vSurf;        // surface proximity 0..1 (1 = right at the surface)

  ${HASH}
  ${VNOISE}

  void main() {
    vec4 posLife = texture2D(texturePosition, ref);
    vec3 vel = texture2D(textureVelocity, ref).xyz;
    vLife = posLife.w;
    vSpeed = length(vel);
    float seed = hash12(ref * vec2(91.7, 13.3));
    vSeed = seed;

    vec4 mv = modelViewMatrix * vec4(posLife.xyz, 1.0);
    gl_Position = projectionMatrix * mv;

    // SURFACE mode: a particle is brightest at the surface plane (z≈0) and fades
    // as it rises/sinks away — so it bubbles up into view and sinks back out,
    // no hard birth/death pop. Otherwise use the legacy life envelope.
    float surf = exp(-(posLife.z * posLife.z) / max(uSurfaceBand * uSurfaceBand, 1e-4));
    vSurf = surf;
    float env = mix(
      smoothstep(0.0, 0.18, vLife) * smoothstep(1.0, 0.8, vLife),
      surf,
      uSurfaceDraw
    );
    // per-particle Perlin breathing between uSizeMin and uSizeMax
    float ns = vnoise(uTime * uSizeSpeed + seed * 97.0);
    float sizePx = mix(uSizeMin, uSizeMax, ns) * env;
    // normalize so size ~= sizePx CSS px at the camera's nominal distance
    gl_PointSize = sizePx * uDpr * (4.2 / max(0.2, -mv.z));
  }
`;

const particleFrag = /* glsl */ `
  precision highp float;
  varying float vSpeed;
  varying float vLife;
  varying float vSeed;
  varying float vSurf;
  uniform float uIntensity;
  uniform float uColorPhase;
  uniform float uColorSpread;
  uniform float uSurfaceDraw;
  uniform vec3 uPalA;
  uniform vec3 uPalB;
  uniform vec3 uPalC;
  uniform vec3 uPalD;

  // Inigo Quilez cosine palette, coefficients supplied as uniforms so the
  // whole palette can crossfade between presets at runtime.
  vec3 palette(float t) {
    return uPalA + uPalB * cos(6.28318 * (uPalC * t + uPalD));
  }

  void main() {
    vec2 pc = gl_PointCoord - 0.5;
    float d = length(pc);
    float alpha = smoothstep(0.5, 0.0, d);
    // brightness envelope: surface-depth fade (water) or life envelope (legacy)
    float env = mix(
      smoothstep(0.0, 0.18, vLife) * smoothstep(1.0, 0.8, vLife),
      vSurf,
      uSurfaceDraw
    );
    // palette position: speed + per-particle offset + rotating global phase
    float t = 0.1 + clamp(vSpeed, 0.0, 1.6) * 0.35 + vSeed * uColorSpread + uColorPhase;
    vec3 col = palette(t);
    float a = alpha * env * uIntensity;
    gl_FragColor = vec4(col * a, a);
  }
`;

const fadeFrag = /* glsl */ `
  precision highp float;
  uniform sampler2D tPrev;
  uniform float uDecay;
  varying vec2 vUv;
  void main() {
    gl_FragColor = texture2D(tPrev, vUv) * uDecay;
  }
`;

const displayFrag = /* glsl */ `
  precision highp float;
  #define MAX_POINTS 6
  uniform sampler2D tTrail;
  uniform vec2 uResolution;
  uniform float uTime;
  uniform float uContrast;    // music-driven brightness contrast (1 = neutral)
  uniform float uExposure;    // auto-exposure: pulled down on loud parts (1 = neutral)
  uniform float uMix;         // 0 = show set A, 1 = show set B (crossfade dissolve)
  // Two full parameter sets so we can crossfade between any two presets.
  // Layout: 0 kind, 1 segments, 2 rings, 3 points, 4 bubbleRate, 5 driftAmt,
  // 6 driftSpeed, 7 rotSpeed, 8 segPerPoint, 9 blendSharp, 10 orbitRadius,
  // 11 orbitSpeed, 12 centerPull, 13 bounceSpeed, 14 activePoints.
  uniform float uA[15];
  uniform float uB[15];
  varying vec2 vUv;

  ${SIMPLEX3D}

  const float TAU = 6.28318530718;
  const float PI  = 3.14159265359;

  vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
  }

  // triangle wave 0..1 — a point sweeping this bounces off both ends
  float tri(float x) { return abs(fract(x * 0.5) * 2.0 - 1.0); }

  // classic radial kaleidoscope around the screen center
  vec2 radial(vec2 uv, float seg, float rot, float aspect) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x) + rot;
    float seg_a = TAU / seg;
    a = mod(a, seg_a);
    a = abs(a - seg_a * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r;
    q.x /= aspect;
    return q + 0.5;
  }

  // mirrored grid tiling (triangle wave folds the image into a grid of mirrors)
  vec2 gridMirror(vec2 uv, float tiles) {
    vec2 t = uv * tiles;
    return abs(fract(t * 0.5) * 2.0 - 1.0);
  }

  // polar: angular segments + radial ring mirroring
  vec2 polar(vec2 uv, float seg, float rings, float rot, float aspect) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x) + rot;
    float seg_a = TAU / seg;
    a = mod(a, seg_a);
    a = abs(a - seg_a * 0.5);
    r = abs(fract(r * rings) * 2.0 - 1.0) * 0.5;
    vec2 q = vec2(cos(a), sin(a)) * r;
    q.x /= aspect;
    return q + 0.5;
  }

  // radial fold of uv around an arbitrary moving center c
  vec2 foldAround(vec2 uv, vec2 c, float seg, float rot, float aspect) {
    vec2 q = uv - c;
    q.x *= aspect;
    float r = length(q);
    float a = atan(q.y, q.x) + rot;
    float seg_a = TAU / seg;
    a = mod(a, seg_a);
    a = abs(a - seg_a * 0.5);
    q = vec2(cos(a), sin(a)) * r;
    q.x /= aspect;
    return c + q;
  }

  // several mirror centers that drift on a perlin field, each fading in/out
  // (bubbling) and respawning elsewhere. Soft-voronoi blend of their folds.
  vec2 multipoint(vec2 uv, float aspect, float pts, float bubbleRate, float driftAmt,
                  float driftSpeed, float rotSpeed, float segPerPoint, float blendSharp,
                  float activePoints) {
    vec2 acc = vec2(0.0);
    float wsum = 0.0;
    for (int i = 0; i < MAX_POINTS; i++) {
      if (float(i) >= pts) break;
      float fi = float(i);
      float phase = uTime * bubbleRate + fi * 0.37;
      float cyc = floor(phase);
      float ph  = fract(phase);
      float life = max(sin(ph * PI), 0.0);       // 0 -> 1 -> 0 bubble envelope
      vec2 base = 0.18 + 0.64 * hash22(vec2(fi * 7.0 + cyc, fi * 3.0 - cyc));
      float dz = uTime * driftSpeed;
      vec2 drift = vec2(
        snoise(vec3(base * 3.0, dz + fi)),
        snoise(vec3(base * 3.0 + 5.0, dz + fi + 11.0))
      ) * driftAmt;
      vec2 c = base + drift;
      float rot = uTime * rotSpeed * (0.6 + 0.8 * hash22(vec2(fi, 1.0)).x) + cyc * 1.3;
      vec2 folded = foldAround(uv, c, segPerPoint, rot, aspect);
      vec2 dv = (uv - c) * vec2(aspect, 1.0);
      float w = life * exp(-length(dv) * blendSharp) * clamp(activePoints - fi, 0.0, 1.0);
      acc += folded * w;
      wsum += w;
    }
    if (wsum < 1e-4) return uv;
    return acc / wsum;
  }

  // center-anchored: a (slightly moving) center mirror provides gravity, while
  // satellite mirror points orbit around it (swirl) with perlin jitter + bubble
  // life. The center keeps the composition framed; satellites off-shoot from it.
  vec2 orbital(vec2 uv, float aspect, float pts, float bubbleRate, float driftAmt,
               float driftSpeed, float rotSpeed, float segPerPoint, float blendSharp,
               float orbitRadius, float orbitSpeed, float centerPull, float activePoints) {
    vec2 acc = vec2(0.0);
    float wsum = 0.0;

    float cz = uTime * driftSpeed;
    vec2 c0 = vec2(0.5) + vec2(snoise(vec3(0.0, 0.0, cz)), snoise(vec3(7.3, 0.0, cz))) * (driftAmt * 0.5);

    {
      vec2 folded = foldAround(uv, c0, segPerPoint, uTime * rotSpeed, aspect);
      vec2 dv = (uv - c0) * vec2(aspect, 1.0);
      float w = centerPull * exp(-length(dv) * (blendSharp * 0.5));
      acc += folded * w;
      wsum += w;
    }

    for (int i = 0; i < MAX_POINTS; i++) {
      if (float(i) >= pts) break;
      float fi = float(i);
      float baseAng = fi / max(pts, 1.0) * TAU;
      float jz = uTime * driftSpeed + fi * 4.0;
      float ang = baseAng + uTime * orbitSpeed + snoise(vec3(fi + 3.0, 0.0, jz)) * driftAmt * 3.0;
      float rad = orbitRadius * (0.75 + 0.45 * snoise(vec3(fi, 0.0, jz)));
      vec2 c = c0 + vec2(cos(ang) / aspect, sin(ang)) * rad;
      float phase = uTime * bubbleRate + fi * 0.41;
      float life = max(sin(fract(phase) * PI), 0.0);
      float rot = uTime * rotSpeed + ang;
      vec2 folded = foldAround(uv, c, segPerPoint, rot, aspect);
      vec2 dv = (uv - c) * vec2(aspect, 1.0);
      float w = life * exp(-length(dv) * blendSharp) * clamp(activePoints - fi, 0.0, 1.0);
      acc += folded * w;
      wsum += w;
    }
    if (wsum < 1e-4) return uv;
    return acc / wsum;
  }

  // swirl centers that travel and ricochet off the four screen edges, each at a
  // different speed/direction (deterministic via triangle waves, no state).
  vec2 bounce(vec2 uv, float aspect, float pts, float rotSpeed, float segPerPoint,
              float blendSharp, float bounceSpeed, float activePoints) {
    vec2 acc = vec2(0.0);
    float wsum = 0.0;
    float m = 0.12;
    for (int i = 0; i < MAX_POINTS; i++) {
      if (float(i) >= pts) break;
      float fi = float(i);
      vec2 h = hash22(vec2(fi * 1.7 + 1.0, fi * 2.3 + 5.0));
      vec2 ph = hash22(vec2(fi * 3.1 + 2.0, fi * 0.7 + 8.0));
      float sp = bounceSpeed * (0.5 + h.x * 1.5);
      float spx = sp * (0.7 + 0.6 * h.y);
      float spy = sp * (0.7 + 0.6 * hash22(vec2(fi, 9.0)).x);
      float px = m + (1.0 - 2.0 * m) * tri(uTime * spx + ph.x * 4.0);
      float py = m + (1.0 - 2.0 * m) * tri(uTime * spy + ph.y * 4.0);
      vec2 c = vec2(px, py);
      float rot = uTime * rotSpeed + fi * 1.3;
      vec2 folded = foldAround(uv, c, segPerPoint, rot, aspect);
      vec2 dv = (uv - c) * vec2(aspect, 1.0);
      float w = exp(-length(dv) * blendSharp) * clamp(activePoints - fi, 0.0, 1.0);
      acc += folded * w;
      wsum += w;
    }
    if (wsum < 1e-4) return uv;
    return acc / wsum;
  }

  // fully symmetric (dihedral N-fold) kaleidoscope. Spin, radius-dependent
  // swirl (orbiting arms), and radial breathing (expand/contract) are ALL
  // functions of (radius, foldedAngle), so the output stays perfectly N-fold
  // symmetric — no scattered centers to break it.
  vec2 symmetric(vec2 uv, float aspect, float seg, float spinSpeed, float swirlAmt,
                 float breatheAmt, float breatheSpeed) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x);
    r *= 1.0 + breatheAmt * sin(uTime * breatheSpeed);   // expand / contract
    a += uTime * spinSpeed + swirlAmt * r;               // spin + orbiting swirl
    float seg_a = TAU / max(seg, 1.0);
    a = mod(a, seg_a);
    a = abs(a - seg_a * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r;
    q.x /= aspect;
    return q + 0.5;
  }

  // symmetric MULTI-center: fold screen space with abs() first (mirroring it
  // into a half-plane for 2 points or a quadrant for 4), then place ONE moving
  // swirl center in that fundamental domain. Its mirror image(s) appear for free
  // and move in perfect symmetric lockstep — so the focal points split apart /
  // drift independently while the whole frame stays mirror-symmetric.
  vec2 symSplit(vec2 uv, float aspect, float splitCount, float segPerPoint,
                float spinSpeed, float spreadBase, float spreadAmp, float spreadSpeed) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    p.x = abs(p.x);                          // mirror L/R  → 2 focal points
    if (splitCount >= 3.5) p.y = abs(p.y);   // also mirror T/B → 4 focal points
    // moving focal center within the domain (parts outward then back)
    float cx = spreadBase + spreadAmp * (0.5 + 0.5 * sin(uTime * spreadSpeed));
    float cy = 0.0;
    if (splitCount >= 3.5)
      cy = spreadBase + spreadAmp * (0.5 + 0.5 * sin(uTime * spreadSpeed * 0.83 + 1.7));
    vec2 c = vec2(cx, cy);
    // radial kaleido fold around the focal center (each point is its own swirl)
    vec2 q = p - c;
    float r = length(q);
    float a = atan(q.y, q.x) + uTime * spinSpeed;
    float seg_a = TAU / max(segPerPoint, 1.0);
    a = mod(a, seg_a);
    a = abs(a - seg_a * 0.5);
    q = vec2(cos(a), sin(a)) * r;
    p = c + q;
    p.x /= aspect;
    return p + 0.5;
  }

  // UNIFIED symmetric morph space (the auto·symmetry family). "Intelligent
  // symmetric points": symmetry comes from the CALCULATION, not fixed screen
  // mirrors. We rigidly ROTATE the whole figure by globalPhase (so it circles),
  // abs()-fold for 4-fold dihedral symmetry about the rotated axes, then place
  // ONE focal point in polar coords (focusR, focusAng):
  //   focusR = 0          → one point at the center (consolidated)
  //   focusR>0, ang = 0   → 2 points;  ang = 45° → 4 points (grid)
  // Because we rotate before folding, the constellation can orbit the center as
  // a rigid body and still stay perfectly symmetric. spinRate spins each point's
  // local kaleido fold; swirl adds a radius-dependent twist (orbiting arms).
  // N-fold dihedral fold: numPoints focal points arranged in an evenly-spaced
  // ring (works for any N — 2, 3 triangle, 4, 6, 8, 16…). We fold the plane into
  // one mirrored sector of angle TAU/N, place ONE focal point on the sector
  // midline at radius focusR, and the N-fold fold replicates it into a perfectly
  // symmetric ring. focusR=0 → all N coincide at the center (one point), so the
  // point COUNT can change invisibly while consolidated. globalPhase rotates the
  // whole ring (circling); each focal region has its own local kaleido fold
  // (localSeg) that spins (spinRate) + twists with radius (swirl).
  vec2 symMorph(vec2 uv, float aspect, float numPoints, float focusR,
                float globalPhase, float spinPhase, float swirl, float localSeg) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x) - globalPhase;       // rotate whole figure
    float sector = TAU / max(numPoints, 1.0);
    a = mod(a, sector);
    a = abs(a - sector * 0.5);                     // mirror within sector → midline at a=0
    p = r * vec2(cos(a), sin(a));
    vec2 c = vec2(focusR, 0.0);                     // focal point on the sector midline
    vec2 q = p - c;
    float rr = length(q);
    // spinPhase is an ACCUMULATED angle (not rate*uTime) so changing the spin
    // speed never causes a jump — the regional pattern's rotation stays C1-smooth.
    float aa = atan(q.y, q.x) + spinPhase + swirl * rr; // regional spin + swirl
    float lseg = TAU / max(localSeg, 1.0);
    aa = mod(aa, lseg);
    aa = abs(aa - lseg * 0.5);
    q = rr * vec2(cos(aa), sin(aa));
    p = c + q;
    p.x /= aspect;
    return p + 0.5;
  }

  // dispatch one parameter set to its kaleidoscope mapping
  vec2 mapK(vec2 uv, float aspect,
            float kind, float seg, float rings, float pts, float bubbleRate,
            float driftAmt, float driftSpeed, float rotSpeed, float segPerPoint,
            float blendSharp, float orbitRadius, float orbitSpeed, float centerPull,
            float bounceSpeed, float activePoints) {
    int k = int(kind + 0.5);
    float rot = uTime * 0.04; // slow global spin for the static families
    if (k == 1) return radial(uv, seg, rot, aspect);
    else if (k == 2) return gridMirror(uv, seg);
    else if (k == 3) return polar(uv, seg, rings, rot, aspect);
    else if (k == 4) return multipoint(uv, aspect, pts, bubbleRate, driftAmt, driftSpeed, rotSpeed, segPerPoint, blendSharp, activePoints);
    else if (k == 5) return orbital(uv, aspect, pts, bubbleRate, driftAmt, driftSpeed, rotSpeed, segPerPoint, blendSharp, orbitRadius, orbitSpeed, centerPull, activePoints);
    else if (k == 6) return bounce(uv, aspect, pts, rotSpeed, segPerPoint, blendSharp, bounceSpeed, activePoints);
    // symmetric: seg, spin=rotSpeed, swirl=driftAmt, breatheAmt=orbitRadius, breatheSpeed=driftSpeed
    else if (k == 7) return symmetric(uv, aspect, seg, rotSpeed, driftAmt, orbitRadius, driftSpeed);
    // symSplit: splitCount=pts, segPerPoint, spin=rotSpeed, spreadBase=orbitRadius, spreadAmp=driftAmt, spreadSpeed=orbitSpeed
    else if (k == 8) return symSplit(uv, aspect, pts, segPerPoint, rotSpeed, orbitRadius, driftAmt, orbitSpeed);
    // symMorph (unified morph space): numPoints=pts, focusR=orbitRadius,
    //   globalPhase=bubbleRate, spinRate=rotSpeed, swirl=driftAmt, localSeg=seg
    else if (k == 9) return symMorph(uv, aspect, pts, orbitRadius, bubbleRate, rotSpeed, driftAmt, seg);
    return uv; // 0 = off
  }

  void main() {
    float aspect = uResolution.x / uResolution.y;
    float m = smoothstep(0.0, 1.0, uMix);

    vec3 col;
    bool bothMorph = int(uA[0] + 0.5) == 9 && int(uB[0] + 0.5) == 9;
    if (uMix > 0.001 && bothMorph) {
      // PARAMETER morph: lerp the symMorph params and sample the trail ONCE, so
      // the same particle field continuously reshapes (focus radius slides,
      // figure rotates) instead of one image cross-dissolving over another.
      float npts   = mix(uA[3],  uB[3],  m);
      float focusR = mix(uA[10], uB[10], m);
      float gPhase = mix(uA[4],  uB[4],  m);
      float spin   = mix(uA[7],  uB[7],  m);
      float swirl  = mix(uA[5],  uB[5],  m);
      float lseg   = mix(uA[1],  uB[1],  m);
      vec2 uv = symMorph(vUv, aspect, npts, focusR, gPhase, spin, swirl, lseg);
      col = texture2D(tTrail, uv).rgb;
    } else {
      vec2 uvA = mapK(vUv, aspect, uA[0],uA[1],uA[2],uA[3],uA[4],uA[5],uA[6],uA[7],uA[8],uA[9],uA[10],uA[11],uA[12],uA[13],uA[14]);
      col = texture2D(tTrail, uvA).rgb;
      if (uMix > 0.001) {
        vec2 uvB = mapK(vUv, aspect, uB[0],uB[1],uB[2],uB[3],uB[4],uB[5],uB[6],uB[7],uB[8],uB[9],uB[10],uB[11],uB[12],uB[13],uB[14]);
        vec3 colB = texture2D(tTrail, uvB).rgb;
        col = mix(col, colB, m); // cross-family fallback: ease-in-out dissolve
      }
    }

    col *= uExposure;                            // auto-exposure (tames loud blowouts)
    col = col / (col + vec3(1.0));               // Reinhard tone map
    float vig = smoothstep(1.1, 0.25, length(vUv - 0.5));
    col *= mix(0.55, 1.0, vig);                   // vignette
    col = pow(col, vec3(0.4545));                 // gamma
    // music-driven contrast around mid-gray: louder = punchier, mellow = softer
    col = clamp((col - 0.5) * uContrast + 0.5, 0.0, 1.0);
    gl_FragColor = vec4(col, 1.0);
  }
`;

export class Visualizer {
  onKaleidoChange: (name: string) => void = () => {};
  onPaletteChange: (name: string) => void = () => {};
  onFieldSpeedChange: (speed: number) => void = () => {};
  onBreatheChange: (rate: number, peak: number) => void = () => {};

  private canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private camera: THREE.PerspectiveCamera;
  private particleScene: THREE.Scene;
  private quadCamera: THREE.OrthographicCamera;
  private fadeScene: THREE.Scene;
  private displayScene: THREE.Scene;

  private gpu!: GPUComputationRenderer;
  private positionVar!: Variable;
  private velocityVar!: Variable;

  private points!: THREE.Points;
  private pointsMat!: THREE.ShaderMaterial;
  private fadeMat!: THREE.ShaderMaterial;
  private displayMat!: THREE.ShaderMaterial;

  private trailA!: THREE.WebGLRenderTarget;
  private trailB!: THREE.WebGLRenderTarget;

  private clock = new THREE.Clock();
  private raf = 0;
  private kaleidoIndex = 0;
  private running = false;

  // palette crossfade state (current coeffs lerp toward target coeffs)
  // how fast the flow field's vector directions evolve over time
  private fieldSpeed = 0.06;
  // particle breathing: cycle rate + peak size (px). min stays 1.
  private breatheRate = 0.6;
  private breathePeak = 4;
  // base brightness; music modulates around it
  private intensity = 0.4;

  // kaleidoscope crossfade: uAVals shown when mix==0, dissolving toward uBVals.
  private uAVals: number[] = new Array(15).fill(0);
  private uBVals: number[] = new Array(15).fill(0);
  private modeA: KaleidoMode = KALEIDO_MODES[0];
  private modeB: KaleidoMode | null = null;
  private mix = 0;
  private mixDur = 1.2;
  // auto-choreography driver
  private autoMode: "cycle" | "music" | "symmetry" | "surface" | null = null;
  private autoLabel = "";
  private autoTimer = 0; // cycle: time on current step
  private autoStep = 0; // cycle: index into AUTO_PROGRAM
  private autoEnergy = 0; // music: slow-smoothed track energy
  private autoTier = 0; // music: current rung in AUTO_LADDER
  private autoCooldown = 0; // music: min dwell before next rung change
  // symmetry journey driver (generative pose machine)
  private symFrom: SymPose = { numPoints: 2, focusR: 0, spin: 0, swirl: 0, circle: 0, localSeg: 6 };
  private symTo: SymPose = { numPoints: 2, focusR: 0, spin: 0, swirl: 0, circle: 0, localSeg: 6 };
  private symCur: SymPose = { numPoints: 2, focusR: 0, spin: 0, swirl: 0, circle: 0, localSeg: 6 };
  private symT = 1; // 0..1 progress of the current ease (1 = arrived)
  private symHold = 0; // seconds left to dwell at the target before retargeting
  private symMoveDur = 4; // seconds for the current ease
  private symAtBloom = false; // is the current target a bloom (vs center)?
  private symPendingN = 2; // point count chosen at center for the next bloom
  private symPhase = 0; // accumulated whole-figure rotation (radians)
  private symSpin = 0; // accumulated regional spin (radians)

  // surface (water) driver — symmetry lives in the force field; the display
  // kaleido fold is OFF. Pose machine like the journey, but it drives the
  // velocity-shader source ring instead of a UV fold.
  private surfActive = false;
  private surfFrom: SurfPose = { numPoints: 3, radius: 0, swirl: 1.2, pull: 0.5, falloff: 2.4 };
  private surfTo: SurfPose = { numPoints: 3, radius: 0, swirl: 1.2, pull: 0.5, falloff: 2.4 };
  private surfCur: SurfPose = { numPoints: 3, radius: 0, swirl: 1.2, pull: 0.5, falloff: 2.4 };
  private surfT = 1;
  private surfHold = 0;
  private surfMoveDur = 5;
  private surfAtBloom = false;
  private surfPendingN = 3;
  private surfRot = 0; // accumulated ring rotation (radians)

  // optional music reactivity
  private audioEl: HTMLAudioElement | null = null;
  private flow: FlowPlayer | null = null;
  private paletteIndex = 0;
  private paletteTimer = 0;
  private curA = new THREE.Vector3();
  private curB = new THREE.Vector3();
  private curC = new THREE.Vector3();
  private curD = new THREE.Vector3();
  private tgtA = new THREE.Vector3();
  private tgtB = new THREE.Vector3();
  private tgtC = new THREE.Vector3();
  private tgtD = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.autoClear = true;

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
    this.camera.position.set(0, 0, 4.2);

    this.particleScene = new THREE.Scene();
    this.fadeScene = new THREE.Scene();
    this.displayScene = new THREE.Scene();
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this.initCompute();
    this.initParticles();
    this.initPasses();
    this.fillMode(this.modeA, this.uAVals); // seed set A with the initial mode
    this.setPalette(0, true);
    this.resize();

    window.addEventListener("resize", this.resize);
    window.addEventListener("keydown", this.onKey);
  }

  private initCompute() {
    this.gpu = new GPUComputationRenderer(TEX, TEX, this.renderer);

    const pos = this.gpu.createTexture();
    const vel = this.gpu.createTexture();
    this.seedTextures(pos, vel);

    this.positionVar = this.gpu.addVariable("texturePosition", positionFrag, pos);
    this.velocityVar = this.gpu.addVariable("textureVelocity", velocityFrag, vel);

    this.gpu.setVariableDependencies(this.positionVar, [
      this.positionVar,
      this.velocityVar,
    ]);
    this.gpu.setVariableDependencies(this.velocityVar, [
      this.positionVar,
      this.velocityVar,
    ]);

    Object.assign(this.positionVar.material.uniforms, {
      uTime: { value: 0 },
      uDt: { value: 0.016 },
      uSpeed: { value: 1.0 },
      uLifeDecay: { value: 0.22 },
    });
    Object.assign(this.velocityVar.material.uniforms, {
      uTime: { value: 0 },
      uFieldScale: { value: 0.9 },
      uFieldSpeed: { value: this.fieldSpeed },
      uInertia: { value: 0.86 },
      uCurlStrength: { value: 1.0 },
      // surface (water) mode
      uSurface: { value: 0 },
      uSrcCount: { value: 3 },
      uSrcRadius: { value: 0.0 },
      uSrcRot: { value: 0.0 },
      uSrcSwirl: { value: 1.4 },
      uSrcPull: { value: 0.5 },
      uSrcFalloff: { value: 2.2 },
      uUpdraft: { value: 0.5 },
      uSettle: { value: 0.8 },
      uSwirlNoise: { value: 0.0 },
    });

    const err = this.gpu.init();
    if (err) console.error("[Visualizer] GPUComputationRenderer init:", err);
  }

  private seedTextures(pos: THREE.DataTexture, vel: THREE.DataTexture) {
    const p = pos.image.data as Float32Array;
    const v = vel.image.data as Float32Array;
    for (let i = 0; i < p.length; i += 4) {
      // random point in a cube, normalized-ish spread
      p[i + 0] = (Math.random() - 0.5) * 2.6;
      p[i + 1] = (Math.random() - 0.5) * 2.6;
      p[i + 2] = (Math.random() - 0.5) * 2.6;
      p[i + 3] = Math.random(); // life
      v[i + 0] = 0;
      v[i + 1] = 0;
      v[i + 2] = 0;
      v[i + 3] = 1;
    }
  }

  private initParticles() {
    const count = TEX * TEX;
    const positions = new Float32Array(count * 3);
    const refs = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      refs[i * 2 + 0] = (i % TEX) / TEX;
      refs[i * 2 + 1] = Math.floor(i / TEX) / TEX;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("ref", new THREE.BufferAttribute(refs, 2));

    this.pointsMat = new THREE.ShaderMaterial({
      uniforms: {
        texturePosition: { value: null },
        textureVelocity: { value: null },
        uSizeMin: { value: 1.0 },
        uSizeMax: { value: this.breathePeak },
        uSizeSpeed: { value: this.breatheRate },
        uDpr: { value: this.renderer.getPixelRatio() },
        uTime: { value: 0 },
        uIntensity: { value: this.intensity },
        uColorPhase: { value: 0 },
        uColorSpread: { value: 0.4 },
        uSurfaceDraw: { value: 0 },
        uSurfaceBand: { value: 0.7 },
        // palette uniform values ARE the cur* vectors, lerped in place each frame
        uPalA: { value: this.curA },
        uPalB: { value: this.curB },
        uPalC: { value: this.curC },
        uPalD: { value: this.curD },
      },
      vertexShader: particleVert,
      fragmentShader: particleFrag,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geo, this.pointsMat);
    this.points.frustumCulled = false;
    this.particleScene.add(this.points);
  }

  private initPasses() {
    const rtOpts = {
      type: THREE.HalfFloatType,
      depthBuffer: false,
      stencilBuffer: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    };
    this.trailA = new THREE.WebGLRenderTarget(2, 2, rtOpts);
    this.trailB = new THREE.WebGLRenderTarget(2, 2, rtOpts);

    const quad = new THREE.PlaneGeometry(2, 2);

    this.fadeMat = new THREE.ShaderMaterial({
      uniforms: {
        tPrev: { value: null },
        uDecay: { value: 0.93 },
      },
      vertexShader: QUAD_VERT,
      fragmentShader: fadeFrag,
      depthTest: false,
      depthWrite: false,
    });
    this.fadeScene.add(new THREE.Mesh(quad, this.fadeMat));

    this.displayMat = new THREE.ShaderMaterial({
      uniforms: {
        tTrail: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uTime: { value: 0 },
        uContrast: { value: 1.0 },
        uExposure: { value: 1.0 },
        uMix: { value: 0 },
        uA: { value: this.uAVals },
        uB: { value: this.uBVals },
      },
      vertexShader: QUAD_VERT,
      fragmentShader: displayFrag,
      depthTest: false,
      depthWrite: false,
    });
    this.displayScene.add(new THREE.Mesh(quad, this.displayMat));
  }

  private resize = () => {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    const dpr = this.renderer.getPixelRatio();
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.trailA.setSize(w * dpr, h * dpr);
    this.trailB.setSize(w * dpr, h * dpr);
    this.displayMat.uniforms.uResolution.value.set(w * dpr, h * dpr);
  };

  private onKey = (e: KeyboardEvent) => {
    if (e.key === "k" || e.key === "K") {
      this.setKaleido(this.kaleidoIndex + 1);
    } else if (e.key === "p" || e.key === "P") {
      this.paletteTimer = 0;
      this.setPalette(this.paletteIndex + 1);
    } else if (e.key === "]") {
      this.setFieldSpeed(this.fieldSpeed * 1.5);
    } else if (e.key === "[") {
      this.setFieldSpeed(this.fieldSpeed / 1.5);
    } else if (e.key === "\\") {
      this.setFieldSpeed(0); // freeze the field's evolution
    } else if (e.key === ".") {
      this.setBreatheRate(this.breatheRate * 1.4); // breathe faster
    } else if (e.key === ",") {
      this.setBreatheRate(this.breatheRate / 1.4); // breathe slower
    } else if (e.key === "=" || e.key === "+") {
      this.setBreathePeak(this.breathePeak + 1); // bigger swell
    } else if (e.key === "-" || e.key === "_") {
      this.setBreathePeak(this.breathePeak - 1); // smaller swell
    }
  };

  /** Switch kaleidoscope entry by index (wraps). Auto entries start a driver. */
  private setKaleido(index: number) {
    this.kaleidoIndex = ((index % KALEIDO_MODES.length) + KALEIDO_MODES.length) % KALEIDO_MODES.length;
    const m = KALEIDO_MODES[this.kaleidoIndex];
    if (m.auto) {
      this.startAuto(m.auto, m.name);
      return;
    }
    this.autoMode = null;
    this.setSurfaceMode(false); // leaving surface → restore legacy field + draw
    this.transitionTo(m, 1.2); // smooth manual switch
    this.emitKaleidoLabel();
  }

  /** Write a preset's params into a 15-slot parameter array (set A or B). */
  private fillMode(m: KaleidoMode, a: number[]) {
    a[0] = m.kind; a[1] = m.segments; a[2] = m.rings; a[3] = m.points;
    a[4] = m.bubbleRate; a[5] = m.driftAmt; a[6] = m.driftSpeed; a[7] = m.rotSpeed;
    a[8] = m.segPerPoint; a[9] = m.blendSharp; a[10] = m.orbitRadius; a[11] = m.orbitSpeed;
    a[12] = m.centerPull; a[13] = m.bounceSpeed; a[14] = m.points; // activePoints = full
  }

  /** Begin a smooth crossfade from the current look to `m` over `dur` seconds. */
  private transitionTo(m: KaleidoMode, dur: number) {
    if (this.modeB) {
      // commit any in-progress transition instantly so we always blend two looks
      this.modeA = this.modeB;
      this.fillMode(this.modeA, this.uAVals);
    }
    this.modeB = m;
    this.fillMode(m, this.uBVals);
    this.mix = 0;
    this.mixDur = Math.max(0.1, dur);
    this.displayMat.uniforms.uMix.value = 0;
  }

  /** Push the current kaleido label to the HUD (includes auto target if active). */
  private emitKaleidoLabel() {
    const cur = (this.modeB || this.modeA).name;
    this.onKaleidoChange(this.autoMode ? `${this.autoLabel} → ${cur}` : cur);
  }

  /** Enter an auto-choreography mode (timed wander, energy ladder, or journey). */
  private startAuto(kind: "cycle" | "music" | "symmetry" | "surface", label: string) {
    this.autoMode = kind;
    this.autoLabel = label;
    this.autoTimer = 0;
    this.autoStep = 0;
    this.autoEnergy = 0;
    this.autoTier = 0;
    this.autoCooldown = 0;
    // surface mode puts symmetry in the FORCE FIELD; turn off the display fold.
    const surf = kind === "surface";
    this.setSurfaceMode(surf);
    if (surf) {
      // no display fold (kind 0 passthrough); the velocity shader makes symmetry
      this.modeB = null;
      this.mix = 0;
      this.displayMat.uniforms.uMix.value = 0;
      this.modeA = km("surface", { kind: 0 });
      this.fillMode(this.modeA, this.uAVals);
      this.surfRot = 0;
      this.surfFrom = this.surfCenterPose();
      this.surfCur = { ...this.surfFrom };
      this.surfTo = this.surfFrom;
      this.surfAtBloom = false;
      this.surfT = 1;
      this.surfHold = 0;
      this.updateSurface(0);
      this.emitKaleidoLabel();
      return;
    }
    if (kind === "symmetry") {
      // one continuously-morphing kind-9 field driven by updateSymJourney();
      // no A/B crossfade needed (params animate every frame). Seed set A.
      this.symPhase = 0;
      this.symSpin = 0;
      this.modeB = null;
      this.mix = 0;
      this.displayMat.uniforms.uMix.value = 0;
      this.modeA = km("auto · symmetry", { kind: 9, segments: 8 });
      this.fillMode(this.modeA, this.uAVals);
      // start consolidated at center, then immediately roll a first bloom target
      this.symFrom = this.centerPose();
      this.symCur = { ...this.symFrom };
      this.symTo = this.symFrom;
      this.symAtBloom = false; // next arrival is a bloom
      this.symT = 1; // force an immediate retarget on first update
      this.symHold = 0;
      this.updateSymJourney(0); // write initial params
    } else {
      const first = kind === "cycle" ? AUTO_PROGRAM[0] : AUTO_LADDER[0];
      this.transitionTo(first, 2.0);
    }
    this.emitKaleidoLabel();
  }

  /** A calm consolidated pose: all points pulled to the center. Point count is
   *  carried through so it can change invisibly while collapsed. */
  private centerPose(numPoints = 2): SymPose {
    return { numPoints, focusR: 0.0, spin: 0.04, swirl: 0.5, circle: 0.0, localSeg: 6 };
  }

  /** Roll a fresh random bloom pose — new N (incl. 3=triangle, up to 16),
   *  radius, spin direction, swirl, regional fold, and rotation. Never repeats. */
  private randomBloom(): SymPose {
    const rnd = Math.random;
    const N = SYM_POINT_COUNTS[Math.floor(rnd() * SYM_POINT_COUNTS.length)];
    // higher N looks better a little tighter so the ring doesn't overlap itself
    const focusR = (N <= 3 ? 0.26 : N <= 6 ? 0.23 : 0.20) + rnd() * 0.06;
    const dir = rnd() < 0.5 ? -1 : 1; // sometimes rotate the other way
    return {
      numPoints: N,
      focusR,
      spin: (0.05 + rnd() * 0.12) * (rnd() < 0.5 ? -1 : 1),
      swirl: 0.8 + rnd() * 2.8,
      circle: dir * (0.02 + rnd() * 0.09),
      // regional fold per focal point: sometimes simple mirror, sometimes ornate
      localSeg: [2, 3, 4, 5, 6, 8][Math.floor(rnd() * 6)],
    };
  }

  /** Generative symmetry journey. Eases the current pose toward a target; on
   *  arrival it dwells (hold), then retargets — alternating CENTER (calm pause)
   *  and a freshly-rolled random BLOOM. The whole figure also rotates (symPhase)
   *  so the constellation circles. Music energy modulates pace, spread, dwell. */
  private updateSymJourney(dt: number) {
    // music energy (or a slow idle drift) → pace + spread + dwell
    let e: number;
    if (this.audioEl && this.flow && !this.audioEl.paused && !this.audioEl.ended) {
      const s = this.flow.sample(this.audioEl.currentTime);
      e = Math.min(1, s.loudness * 1.2 + s.bass * 0.2);
    } else {
      e = 0.5 + 0.5 * Math.sin(this.clock.elapsedTime * 0.06);
    }
    this.autoEnergy += (e - this.autoEnergy) * (1 - Math.exp(-dt * 0.5));

    // advance the eased transition toward the target pose (slower = more graceful)
    const pace = 0.5 + this.autoEnergy * 0.8;
    if (this.symT < 1) {
      this.symT = Math.min(1, this.symT + (dt / this.symMoveDur) * pace);
    } else {
      // arrived: dwell, then retarget (alternating CENTER ↔ a fresh random BLOOM)
      this.symHold -= dt;
      if (this.symHold <= 0) {
        this.symFrom = { ...this.symCur };
        if (this.symAtBloom) {
          // return to center; carry the NEXT bloom's N so the count change is hidden
          this.symPendingN = SYM_POINT_COUNTS[Math.floor(Math.random() * SYM_POINT_COUNTS.length)];
          this.symTo = this.centerPose(this.symPendingN);
          this.symMoveDur = 3.5 + Math.random() * 2.5;
          this.symHold = 3 + Math.random() * 6; // pause at center (longer sometimes)
          this.symAtBloom = false;
        } else {
          const bloom = this.randomBloom();
          bloom.numPoints = this.symPendingN || bloom.numPoints; // N chosen at center
          this.symTo = bloom;
          this.symMoveDur = 4 + Math.random() * 3;
          this.symHold = 4 + Math.random() * 5; // dwell in the bloom
          this.symAtBloom = true;
        }
        this.symT = 0;
      }
    }

    // smoothstep-eased interpolation from symFrom → symTo
    const t = this.symT;
    const m = t * t * (3 - 2 * t);
    const L = (x: number, y: number) => x + (y - x) * m;
    const f = this.symFrom, g = this.symTo;
    const cur = this.symCur;
    // DISCRETE fold counts (N, localSeg) can't be lerped — fractional folds make
    // a sweeping seam and N pops. The journey always passes through the
    // consolidated center (focusR=0, all points coincident), so we hold these
    // constant per-bloom and let them switch only there, where it's invisible.
    // Show the bloom-leg's values: center→bloom uses the target bloom (starts at
    // R=0); bloom→center keeps the leaving bloom (ends at R=0).
    const blo = this.symAtBloom ? g : f;
    cur.numPoints = blo.numPoints;
    cur.localSeg = blo.localSeg;
    // CONTINUOUS params flow smoothly — these are what visibly morphs:
    cur.focusR = L(f.focusR, g.focusR) * (1 + this.autoEnergy * 0.22); // open/close
    cur.swirl = L(f.swirl, g.swirl);   // swirl*radius, no uTime amplification → safe
    cur.circle = L(f.circle, g.circle); // rate fed into accumulated symPhase → safe
    cur.spin = L(f.spin, g.spin);       // rate fed into accumulated symSpin → safe

    // advance rotation PHASES from their rates (accumulation makes rate changes
    // jump-free): symPhase = whole-figure circling, symSpin = regional spin.
    this.symPhase += dt * cur.circle * TAU;
    this.symSpin += dt * cur.spin * TAU;

    // write kind-9 slots: numPoints=pts[3], focusR=orbitRadius[10],
    // globalPhase=bubbleRate[4], spinPhase=rotSpeed[7], swirl=driftAmt[5], localSeg=seg[1]
    const u = this.uAVals;
    u[0] = 9;
    u[3] = cur.numPoints;
    u[1] = cur.localSeg;
    u[10] = cur.focusR;
    u[4] = this.symPhase;
    u[7] = this.symSpin;
    u[5] = cur.swirl;
  }

  /** Toggle the water-surface simulation (symmetry in the force field). When on,
   *  the display kaleido fold is bypassed and particles draw with a depth fade. */
  private setSurfaceMode(on: boolean) {
    this.surfActive = on;
    this.velocityVar.material.uniforms.uSurface.value = on ? 1 : 0;
    this.pointsMat.uniforms.uSurfaceDraw.value = on ? 1 : 0;
  }

  private surfCenterPose(numPoints = 3): SurfPose {
    return { numPoints, radius: 0.0, swirl: 1.0, pull: 0.6, falloff: 2.6 };
  }

  /** Roll a fresh random water bloom: new source count, ring radius, swirl. */
  private randomSurfBloom(): SurfPose {
    const r = Math.random;
    const N = SURF_POINT_COUNTS[Math.floor(r() * SURF_POINT_COUNTS.length)];
    return {
      numPoints: N,
      radius: (N <= 3 ? 1.1 : N <= 5 ? 0.95 : 0.8) + r() * 0.4,
      swirl: (0.8 + r() * 1.8) * (r() < 0.5 ? -1 : 1), // sometimes counter-swirl
      pull: 0.35 + r() * 0.5,
      falloff: 1.6 + r() * 1.8,
    };
  }

  /** Water-surface journey: eases the source ring between CENTER (consolidated)
   *  and a fresh random BLOOM, driving the velocity shader. The ring also slowly
   *  rotates. N changes only at center (radius≈0) where sources coincide, so the
   *  count change is a physical bubbling-in, never a snap. */
  private updateSurface(dt: number) {
    let e: number;
    if (this.audioEl && this.flow && !this.audioEl.paused && !this.audioEl.ended) {
      const s = this.flow.sample(this.audioEl.currentTime);
      e = Math.min(1, s.loudness * 1.2 + s.bass * 0.2);
    } else {
      e = 0.5 + 0.5 * Math.sin(this.clock.elapsedTime * 0.06);
    }
    this.autoEnergy += (e - this.autoEnergy) * (1 - Math.exp(-dt * 0.5));

    const pace = 0.5 + this.autoEnergy * 0.8;
    if (this.surfT < 1) {
      this.surfT = Math.min(1, this.surfT + (dt / this.surfMoveDur) * pace);
    } else {
      this.surfHold -= dt;
      if (this.surfHold <= 0) {
        this.surfFrom = { ...this.surfCur };
        if (this.surfAtBloom) {
          this.surfPendingN = SURF_POINT_COUNTS[Math.floor(Math.random() * SURF_POINT_COUNTS.length)];
          this.surfTo = this.surfCenterPose(this.surfPendingN);
          this.surfMoveDur = 4 + Math.random() * 3;
          this.surfHold = 3 + Math.random() * 5; // pause, consolidated at center
          this.surfAtBloom = false;
        } else {
          const bloom = this.randomSurfBloom();
          bloom.numPoints = this.surfPendingN || bloom.numPoints;
          this.surfTo = bloom;
          this.surfMoveDur = 5 + Math.random() * 4; // slow, graceful spread
          this.surfHold = 5 + Math.random() * 6; // dwell in the bloom
          this.surfAtBloom = true;
        }
        this.surfT = 0;
      }
    }

    const t = this.surfT;
    const m = t * t * (3 - 2 * t);
    const L = (x: number, y: number) => x + (y - x) * m;
    const f = this.surfFrom, g = this.surfTo;
    const cur = this.surfCur;
    // N is discrete: hold the bloom-leg's count so it only changes at radius≈0.
    cur.numPoints = (this.surfAtBloom ? g : f).numPoints;
    cur.radius = L(f.radius, g.radius);
    cur.swirl = L(f.swirl, g.swirl);
    cur.pull = L(f.pull, g.pull);
    cur.falloff = L(f.falloff, g.falloff);

    // slow ring rotation, scaled a touch by energy
    this.surfRot += dt * (0.12 + this.autoEnergy * 0.18);

    const u = this.velocityVar.material.uniforms;
    u.uSrcCount.value = cur.numPoints;
    u.uSrcRadius.value = cur.radius;
    u.uSrcRot.value = this.surfRot;
    u.uSrcSwirl.value = cur.swirl;
    u.uSrcPull.value = cur.pull;
    u.uSrcFalloff.value = cur.falloff;
  }

  /** Advance whichever auto driver is active; may trigger a crossfade. */
  private updateAuto(dt: number) {
    if (!this.autoMode) return;
    this.autoCooldown = Math.max(0, this.autoCooldown - dt);

    if (this.autoMode === "surface") {
      this.updateSurface(dt);
      return;
    }

    if (this.autoMode === "symmetry") {
      this.updateSymJourney(dt);
      return;
    }

    if (this.autoMode === "cycle") {
      this.autoTimer += dt;
      if (this.autoTimer > 9 && !this.modeB) {
        this.autoTimer = 0;
        this.autoStep = (this.autoStep + 1) % AUTO_PROGRAM.length;
        this.transitionTo(AUTO_PROGRAM[this.autoStep], 3.5);
        this.emitKaleidoLabel();
      }
      return;
    }

    // music: pick the ladder rung from slow-smoothed track energy.
    // When nothing is playing, drift the energy on a slow sine so it still
    // wanders the ladder (nice as an idle screensaver without a track).
    let e: number;
    if (this.audioEl && this.flow && !this.audioEl.paused && !this.audioEl.ended) {
      const s = this.flow.sample(this.audioEl.currentTime);
      e = Math.min(1, s.loudness * 1.2 + s.bass * 0.2);
    } else {
      e = 0.5 + 0.5 * Math.sin(this.clock.elapsedTime * 0.06);
    }
    this.autoEnergy += (e - this.autoEnergy) * (1 - Math.exp(-dt * 0.5)); // ~2s smoothing
    const ladder = AUTO_LADDER;
    const top = ladder.length - 1;
    const lvl = Math.max(0, Math.min(1, this.autoEnergy)) * top;
    if (!this.modeB && this.autoCooldown <= 0) {
      let desired = this.autoTier;
      if (lvl > this.autoTier + 0.6) desired = this.autoTier + 1; // step up one rung
      else if (lvl < this.autoTier - 0.6) desired = this.autoTier - 1; // step down one
      desired = Math.max(0, Math.min(top, desired));
      if (desired !== this.autoTier) {
        this.autoTier = desired;
        this.transitionTo(ladder[desired], 3.0);
        this.autoCooldown = 4; // min seconds between rung changes
        this.emitKaleidoLabel();
      }
    }
  }

  /** Set how fast the flow field's vector directions change over time. */
  private setFieldSpeed(speed: number) {
    // clamp to a sane range; 0 = frozen field, ~0.9 = churning fast
    this.fieldSpeed = speed <= 0 ? 0 : Math.min(0.9, Math.max(0.004, speed));
    this.velocityVar.material.uniforms.uFieldSpeed.value = this.fieldSpeed;
    this.onFieldSpeedChange(this.fieldSpeed);
  }

  /** How fast each particle cycles between min and peak size. */
  private setBreatheRate(rate: number) {
    this.breatheRate = Math.min(3.0, Math.max(0.05, rate));
    this.pointsMat.uniforms.uSizeSpeed.value = this.breatheRate;
    this.onBreatheChange(this.breatheRate, this.breathePeak);
  }

  /** Peak particle size (px) at full swell; depth of the breath above the 1px floor. */
  private setBreathePeak(peak: number) {
    this.breathePeak = Math.min(14, Math.max(1, peak));
    this.pointsMat.uniforms.uSizeMax.value = this.breathePeak;
    this.onBreatheChange(this.breatheRate, this.breathePeak);
  }

  /** Switch to a palette by index; crossfades unless `instant`. */
  private setPalette(index: number, instant = false) {
    this.paletteIndex = ((index % PALETTES.length) + PALETTES.length) % PALETTES.length;
    const p = PALETTES[this.paletteIndex];
    this.tgtA.fromArray(p.a);
    this.tgtB.fromArray(p.b);
    this.tgtC.fromArray(p.c);
    this.tgtD.fromArray(p.d);
    if (instant) {
      this.curA.copy(this.tgtA);
      this.curB.copy(this.tgtB);
      this.curC.copy(this.tgtC);
      this.curD.copy(this.tgtD);
    }
    this.onPaletteChange(p.name);
  }

  /** Begin reacting to a track. `audioEl` is the playback clock. */
  attachAudio(audioEl: HTMLAudioElement, flow: FlowPlayer) {
    this.audioEl = audioEl;
    this.flow = flow;
    flow.reset();
  }

  detachAudio() {
    this.audioEl = null;
    this.flow = null;
    this.writeBaseUniforms();
  }

  /** Reset the music-modulated uniforms to their manual base values. */
  private writeBaseUniforms() {
    this.pointsMat.uniforms.uIntensity.value = this.intensity;
    this.pointsMat.uniforms.uSizeMax.value = this.breathePeak;
    this.velocityVar.material.uniforms.uCurlStrength.value = 1.0;
    this.positionVar.material.uniforms.uSpeed.value = 1.0;
    this.velocityVar.material.uniforms.uFieldSpeed.value = this.fieldSpeed;
    this.displayMat.uniforms.uContrast.value = 1.0; // neutral when no track plays
    this.displayMat.uniforms.uExposure.value = 1.0;
    this.uAVals[14] = this.modeA.points; // show all centers when idle
    if (this.modeB) this.uBVals[14] = this.modeB.points;
  }

  /** Map sampled music signals onto the engine, modulating around base values. */
  private applyAudio(dt: number) {
    if (!this.audioEl || !this.flow) return;
    const s = this.flow.sample(this.audioEl.currentTime);
    // brightness pulses with the beat + overall loudness
    this.pointsMat.uniforms.uIntensity.value =
      this.intensity * (0.5 + s.loudness * 0.7 + s.pulse * 1.1);
    // particles swell on the kick
    this.pointsMat.uniforms.uSizeMax.value =
      this.breathePeak * (1.0 + s.pulse * 0.7 + s.bass * 0.4);
    // bass surges the flow field strength + particle speed
    this.velocityVar.material.uniforms.uCurlStrength.value = 1.0 + s.bass * 1.5;
    this.positionVar.material.uniforms.uSpeed.value = 1.0 + s.bass * 0.7 + s.pulse * 0.5;
    // overall energy nudges how fast the field evolves
    this.velocityVar.material.uniforms.uFieldSpeed.value = this.fieldSpeed * (1.0 + s.loudness * 0.6);
    // treble shimmer drifts the palette a touch faster
    this.pointsMat.uniforms.uColorPhase.value += s.treble * dt * 0.05;
    // brightness contrast tracks volume: mellow = soft/flat, loud = punchy.
    // ease toward the target so it swells with the music rather than flickering.
    const targetContrast = 0.95 + s.loudness * 0.55 + s.pulse * 0.12;
    const cu = this.displayMat.uniforms.uContrast;
    cu.value += (targetContrast - cu.value) * (1 - Math.exp(-dt * 6));
    // auto-exposure: as the scene gets loud/dense it would blow out to flat
    // white (additive trails saturate, Reinhard clamps). Pull exposure DOWN with
    // loudness so loud reads as punchier contrast, not a washed-out glare.
    const targetExposure = 1.0 / (1.0 + s.loudness * 1.3 + s.pulse * 0.5);
    const eu = this.displayMat.uniforms.uExposure;
    eu.value += (targetExposure - eu.value) * (1 - Math.exp(-dt * 5));
    // intensity scales how many swirl centers are alive: crescendo fills the
    // screen with swirls, mellow collapses toward a single dim center. Applies
    // to whichever of the two crossfade sets are reactive presets.
    const energy = Math.min(1, s.loudness * 1.15 + s.bass * 0.25);
    const k2 = 1 - Math.exp(-dt * 4); // ease w/ phrasing
    if (this.modeA.reactive) {
      const tgt = 0.6 + energy * (this.modeA.points - 0.6);
      this.uAVals[14] += (tgt - this.uAVals[14]) * k2;
    }
    if (this.modeB && this.modeB.reactive) {
      const tgt = 0.6 + energy * (this.modeB.points - 0.6);
      this.uBVals[14] += (tgt - this.uBVals[14]) * k2;
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    // default to the headline surface (water) mode
    this.setKaleido(0);
    // push current state so the HUD reflects it immediately
    this.emitKaleidoLabel();
    this.onPaletteChange(PALETTES[this.paletteIndex].name);
    this.onFieldSpeedChange(this.fieldSpeed);
    this.onBreatheChange(this.breatheRate, this.breathePeak);
    this.clock.start();
    this.loop();
  }

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(this.clock.getDelta(), 0.033);
    const t = this.clock.elapsedTime;

    // advance simulation
    const pu = this.positionVar.material.uniforms;
    const vu = this.velocityVar.material.uniforms;
    pu.uTime.value = t;
    pu.uDt.value = dt;
    vu.uTime.value = t;
    this.gpu.compute();

    this.pointsMat.uniforms.texturePosition.value =
      this.gpu.getCurrentRenderTarget(this.positionVar).texture;
    this.pointsMat.uniforms.textureVelocity.value =
      this.gpu.getCurrentRenderTarget(this.velocityVar).texture;

    // dynamic sizing clock + continuous color rotation
    this.pointsMat.uniforms.uTime.value = t;
    this.pointsMat.uniforms.uColorPhase.value += dt * 0.015;

    // smooth palette crossfade (uniform values are the cur* vectors)
    const k = 1 - Math.exp(-dt * 1.6);
    this.curA.lerp(this.tgtA, k);
    this.curB.lerp(this.tgtB, k);
    this.curC.lerp(this.tgtC, k);
    this.curD.lerp(this.tgtD, k);

    // auto-cycle palettes every ~24s (manual switch with 'P' resets the timer)
    this.paletteTimer += dt;
    if (this.paletteTimer > 24) {
      this.paletteTimer = 0;
      this.setPalette(this.paletteIndex + 1);
    }

    // music reactivity: modulate around base values while a track plays,
    // otherwise keep the manual (keyboard) base values.
    if (this.audioEl && this.flow && !this.audioEl.paused && !this.audioEl.ended) {
      this.applyAudio(dt);
    } else if (this.audioEl) {
      this.writeBaseUniforms();
    }

    // auto-choreography + kaleidoscope crossfade advance
    this.updateAuto(dt);
    if (this.modeB) {
      this.mix += dt / this.mixDur;
      if (this.mix >= 1) {
        this.modeA = this.modeB;
        this.modeB = null;
        this.mix = 0;
        this.fillMode(this.modeA, this.uAVals);
        this.displayMat.uniforms.uMix.value = 0;
        this.emitKaleidoLabel();
      } else {
        this.displayMat.uniforms.uMix.value = this.mix;
      }
    }

    // slow camera drift for parallax/depth
    this.camera.position.x = Math.sin(t * 0.05) * 0.6;
    this.camera.position.y = Math.cos(t * 0.037) * 0.4;
    this.camera.lookAt(0, 0, 0);

    // 1. fade previous trail into write target
    this.fadeMat.uniforms.tPrev.value = this.trailA.texture;
    this.renderer.autoClear = true;
    this.renderer.setRenderTarget(this.trailB);
    this.renderer.render(this.fadeScene, this.quadCamera);

    // 2. additively draw particles on top (no clear)
    this.renderer.autoClear = false;
    this.renderer.render(this.particleScene, this.camera);
    this.renderer.autoClear = true;

    // 3. composite + kaleidoscope to screen
    this.displayMat.uniforms.tTrail.value = this.trailB.texture;
    this.displayMat.uniforms.uTime.value = t;
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.displayScene, this.quadCamera);

    // swap trail buffers
    const tmp = this.trailA;
    this.trailA = this.trailB;
    this.trailB = tmp;
  };

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.resize);
    window.removeEventListener("keydown", this.onKey);
    this.trailA.dispose();
    this.trailB.dispose();
    this.points.geometry.dispose();
    this.pointsMat.dispose();
    this.fadeMat.dispose();
    this.displayMat.dispose();
    this.gpu.dispose();
    this.renderer.dispose();
  }
}

const QUAD_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;
