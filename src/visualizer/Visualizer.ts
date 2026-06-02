/**
 * KaleidoFlow rendering engine.
 *
 * A GPGPU particle flow field (curl-noise advection in ping-pong float
 * textures) drawn additively into feedback/trail buffers, then composited
 * through a kaleidoscope + tone-map + cosine palette display pass.
 *
 * Symmetry is a post-process FOLD of the trail in `displayFrag`. The several
 * fold functions (radial, triangle kaleidoscope, N-fold dihedral `symMorph`,
 * the `divideMove` grid fold, the force-field `surface` mode…) are selected by
 * the active mode. The four menu modes are auto-choreographed "drivers"
 * (`updateSymJourney`, `updateDivide`, `updateSurface`) that ease a set of fold
 * parameters over time; transitions emerge/collapse seams through a clean
 * symmetric base so structure changes are seamless.
 *
 * Music (sampled from the precomputed FlowMap each frame in `applyAudio`) drives
 * only LIGHT and COLOUR — never particle motion — so the flow stays smooth.
 */
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
  auto?: "cycle" | "music" | "symmetry" | "surface" | "pinwheel" | "divide" | "dividev2" | "radialdiv"; // special: auto-choreography driver (no own geometry)
};
const K0 = { segments: 0, rings: 0, points: 0, bubbleRate: 0, driftAmt: 0, driftSpeed: 0, rotSpeed: 0, segPerPoint: 6, blendSharp: 8, orbitRadius: 0, orbitSpeed: 0, centerPull: 0, bounceSpeed: 0, reactive: 0 };
// The curated menu — four polished evolving modes. (The shader still contains
// the other fold kinds 1-9 used internally by these modes; only the menu is
// trimmed.) "evolving symmetry" is the headline/default.
const KALEIDO_MODES: KaleidoMode[] = [
  { name: "evolving symmetry", kind: -1, ...K0, auto: "radialdiv" }, // radial home + grids/triangles
  { name: "evolving surface", kind: -1, ...K0, auto: "surface" },    // water flow field
  { name: "evolving morph", kind: -1, ...K0, auto: "symmetry" },     // continuous symMorph journey
  { name: "evolving pinwheel", kind: -1, ...K0, auto: "pinwheel" },  // seamless rotational/spiral
];

// quick constructor for ad-hoc presets (low mirror counts not in the K cycle)
const km = (name: string, over: Partial<KaleidoMode> & { kind: number }): KaleidoMode => ({
  name,
  ...K0,
  ...over,
});
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
// Allowed focal-point counts for blooms — even spacing guaranteed for any N.
// FLOOR OF 4: counts of 2 (a single mirror line) and 3 read as a lopsided
// "two-way split", not a kaleidoscope — so the lowest count is 4 (a clean
// four-way split). Weighted (repeats) toward the headline counts 4/6/8.
const SYM_POINT_COUNTS = [4, 4, 5, 6, 6, 8, 8, 9, 9, 12, 16];

// DIVIDE-AND-MOVE pose (auto · divide). Per-axis split spread (dx,dy) + per-axis
// bisect(0)/trisect(1) mode (kx,ky). Grid count = (dx>0 ? (kx?3:2) : 1) ×
// (dy>0 ? (ky?3:2) : 1). All continuous: at d=0 an axis is a single center;
// growing d divides it and slides the parts apart. Mode flips (kx/ky) happen
// only at d≈0 where bisect==trisect, so they're invisible.
type DivPose = {
  dx: number; dy: number;
  kx: number; ky: number;
  spin: number; swirl: number;
  // symCells (auto · split) extras: layout 0=grid/1=ring; nx/ny axis divisions
  // (grid) or ring point count (nx); seg = rosette fold per focal point
  layout: number; nx: number; ny: number; seg: number;
  // v1 divide: bloomType 0=grid, 1=triangle/ring, 2=radial; radSeg = radial
  // segment count; seam = rest seam strength (radial home rests partly folded)
  bloomType: number; radSeg: number; seam: number;
};

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
  uniform float uWorldStretch; // horizontal world scale so content fills a wide screen
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

    // stretch the world horizontally at draw time so the (isotropic) field fills
    // a wide screen — the simulation/symmetry stays perfectly circular; only the
    // rendered shape widens. Keeps z intact for the depth/surface model.
    vec3 wp = vec3(posLife.x * uWorldStretch, posLife.y, posLife.z);
    vec4 mv = modelViewMatrix * vec4(wp, 1.0);
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
  uniform float uTrailBlur;   // 0 = crisp streaks, >0 = isotropic softening (px)
  uniform vec2 uTexel;        // 1/resolution
  varying vec2 vUv;
  void main() {
    vec4 c;
    if (uTrailBlur < 0.01) {
      c = texture2D(tPrev, vUv);
    } else {
      // cheap separable-ish 5-tap isotropic blur: rounds long directional streaks
      // into softer ink so there are fewer linear features to reflect at seams.
      vec2 o = uTexel * uTrailBlur;
      c  = texture2D(tPrev, vUv) * 0.4;
      c += texture2D(tPrev, vUv + vec2( o.x, 0.0)) * 0.15;
      c += texture2D(tPrev, vUv + vec2(-o.x, 0.0)) * 0.15;
      c += texture2D(tPrev, vUv + vec2(0.0,  o.y)) * 0.15;
      c += texture2D(tPrev, vUv + vec2(0.0, -o.y)) * 0.15;
    }
    gl_FragColor = c * uDecay;
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
  uniform float uSeamSoft;    // 0 = hard mirror seams, 1 = strongly feathered seams
  uniform float uRotational;  // 0 = mirror fold (symMorph), 1 = rotational (rotMorph, no seam)
  uniform float uDivide;      // 1 = divide-and-move fold (auto · divide), uses uA slots
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
  // Rounded mirror fold. Replaces a hard abs() (which makes a sharp derivative
  // kink → a high-contrast seam) with sqrt(x²+k²)−k: the crease at the fold line
  // is rounded, and because the folded value tops out slightly below the half-
  // sector, content eases off APPROACHING the seam instead of meeting its mirror
  // head-on. k=0 → identical to abs() (hard seam). k scales with the sector.
  float softFold(float x, float k) {
    return sqrt(x * x + k * k) - k;
  }

  // distance (in [0..1] of a half-sector) from the nearest fold line of a fold
  // whose half-angle measure is halfAng. 0 = right on the seam, 1 = mid-wedge.
  float seamProximity(float foldedAbs, float halfAng) {
    return clamp(foldedAbs / max(halfAng, 1e-4), 0.0, 1.0);
  }

  // symMorph now also returns a brightness weight via outW (1 mid-wedge, dips
  // toward the seams) so the doubled-up reflected streak fades into darkness
  // instead of glowing as a hard line. uSeamSoft drives both the wavy warp and
  // the feather depth.
  vec2 symMorph(vec2 uv, float aspect, float numPoints, float focusR,
                float globalPhase, float spinPhase, float swirl, float localSeg,
                out float outW) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x) - globalPhase;       // rotate whole figure
    float sector = TAU / max(numPoints, 1.0);
    // (B) WAVY SEAM: perturb the angle with noise that is PERIODIC over the
    // sector (so the result stays N-fold symmetric) — the seam wanders instead of
    // being a ruler-straight line. Amplitude scales with uSeamSoft + radius.
    float wob = sin(a * numPoints + uTime * 0.3) * snoise(vec3(cos(a)*1.7, sin(a)*1.7, uTime*0.05));
    a += wob * uSeamSoft * 0.10;
    a = mod(a, sector);
    float kS = uSeamSoft * 0.16 * sector;          // rounded crease feather
    float foldedS = softFold(a - sector * 0.5, kS);
    float seamW = seamProximity(foldedS, sector * 0.5);
    p = r * vec2(cos(foldedS), sin(foldedS));
    vec2 c = vec2(focusR, 0.0);                     // focal point on the sector midline
    vec2 q = p - c;
    float rr = length(q);
    float aa = atan(q.y, q.x) + spinPhase + swirl * rr; // regional spin + swirl
    float lseg = TAU / max(localSeg, 1.0);
    aa = mod(aa, lseg);
    float kL = uSeamSoft * 0.16 * lseg;
    float foldedL = softFold(aa - lseg * 0.5, kL);
    float seamWL = seamProximity(foldedL, lseg * 0.5);
    q = rr * vec2(cos(foldedL), sin(foldedL));
    p = c + q;
    p.x /= aspect;
    // (A) FEATHER: dip brightness toward each seam. smoothstep over a band whose
    // width is uSeamSoft; 1 mid-wedge → dips near the line. Combine both folds.
    float band = mix(0.0, 0.5, uSeamSoft);
    float wS = smoothstep(0.0, band + 1e-3, seamW);
    float wL = smoothstep(0.0, band + 1e-3, seamWL);
    outW = mix(1.0, wS * wL, clamp(uSeamSoft, 0.0, 1.0));
    return p + 0.5;
  }

  // (E) ROTATIONAL blend — NO mirror fold, so NO seam can form. Instead of
  // reflecting, we map the angle into one sector by pure rotation (mod, no abs).
  // That gives cyclic/rotational (pinwheel) symmetry: N rotated copies meet
  // edge-to-edge with matching content, so the wedge boundary is continuous —
  // no doubled reflected streak, no stretch line. localSeg adds an inner
  // rotational fold around the focal point. Reads more "spiral" than "mirror".
  vec2 rotMorph(vec2 uv, float aspect, float numPoints, float focusR,
                float globalPhase, float spinPhase, float swirl, float localSeg) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x) - globalPhase;
    float sector = TAU / max(numPoints, 1.0);
    a = mod(a, sector);                 // rotate into sector — NO abs() = no mirror
    p = r * vec2(cos(a), sin(a));
    vec2 c = vec2(focusR, 0.0);
    vec2 q = p - c;
    float rr = length(q);
    float aa = atan(q.y, q.x) + spinPhase + swirl * rr;
    float lseg = TAU / max(localSeg, 1.0);
    aa = mod(aa, lseg);                 // inner rotation, also no mirror
    q = rr * vec2(cos(aa), sin(aa));
    p = c + q;
    p.x /= aspect;
    return p + 0.5;
  }

  // ── DIVIDE-AND-MOVE fold (auto · divide) ───────────────────────────────────
  // A 1-D fold along one axis whose focal point(s) DIVIDE and slide apart as the
  // spread d grows from 0, with no fade. Two modes, identical at d=0 (=abs, one
  // center) so the mode can switch invisibly while consolidated:
  //   bisect  → abs(abs(x) - d)        : 1 center splits into 2 sliding to ±d
  //   trisect → keep center + a mirrored pair: 3 centers (one stays, two leave)
  // We return the folded coordinate; the "kept center" of trisect is just the
  // region nearest x=0, which abs() already leaves in place — so trisect is a
  // SOFTER split that keeps the middle. We blend bisect to trisect with keep
  // (0=bisect, 1=trisect). Both fold so the output = signed distance to the
  // NEAREST center, which is 0 exactly at each center → that's where the mirror
  // tiling places a focal point.
  //   bisect  centers at {-d, +d}      → 2
  //   trisect centers at {-d, 0, +d}   → 3 (keeps the middle)
  // At d=0 every center coincides at 0 and both reduce to abs(x) → identical, so
  // the bisect/trisect choice can flip invisibly while consolidated.
  float fold1D(float x, float d, float keep) {
    float a = abs(x);
    float bisect = abs(a - d);                 // nearest of ±d
    float trisect = min(a, abs(a - d));        // nearest of 0, ±d
    return mix(bisect, trisect, keep);
  }

  // dx/dy = split spread per axis; kx/ky = bisect(0)→trisect(1) per axis; the
  // count is (2 or 3 on x) × (1,2,3 on y). worldRot spins the WHOLE composition
  // before folding; spin/swirl give each cell its own rotation/twist; innerSeg
  // adds a per-cell radial kaleido fold for more diversified patterns.
  vec2 divideMove(vec2 uv, float aspect, float dx, float dy, float kx, float ky,
                  float worldRot, float spinPhase, float swirl, float innerSeg) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    // (3) whole-screen rotation: rotate the entire field before the split fold,
    // so the whole composition turns (strongest in transitions, driven below).
    float wc = cos(worldRot), ws = sin(worldRot);
    p = mat2(wc, -ws, ws, wc) * p;
    float fx = fold1D(p.x, dx, kx);
    float fy = fold1D(p.y, dy, ky);
    vec2 q = vec2(fx, fy);
    // (1) per-cell spin + swirl around each cell origin
    float rr = length(q);
    float aa = atan(q.y, q.x) + spinPhase + swirl * rr;
    // (2) inner kaleido fold per cell for diversified patterns (innerSeg≈1 → off)
    if (innerSeg > 1.5) {
      float seg_a = TAU / innerSeg;
      aa = abs(mod(aa, seg_a) - seg_a * 0.5);
    }
    q = rr * vec2(cos(aa), sin(aa));
    q.x /= aspect;
    return q + 0.5;
  }

  // TRIANGLE (and any N-point) via SOFT-BLENDED focal points — no hard fold, so
  // NO strong seams (unlike an angular mirror fold). nPts points sit evenly on a
  // ring of radius rad; each contributes a spun/translated sample, blended by
  // smooth exp(-dist*sharp) weights → a merged, seamless field. rad=0 ⇒ all
  // points coincide at center (collapses to one) so it merges/blooms naturally.
  // worldRot orbits the whole ring.
  vec2 triPoints(vec2 uv, float aspect, float nPts, float rad, float worldRot,
                 float spinPhase, float swirl, float sharp) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    vec2 acc = vec2(0.0);
    float wsum = 0.0;
    float cs = cos(spinPhase), sn = sin(spinPhase);
    mat2 rot = mat2(cs, -sn, sn, cs);
    for (int k = 0; k < 9; k++) {
      if (float(k) >= nPts) break;
      float ang = worldRot + TAU * float(k) / nPts + PI * 0.5; // +90° → point up
      vec2 c = rad * vec2(cos(ang), sin(ang));
      vec2 lp = p - c;
      float d = length(lp);
      float w = exp(-d * sharp);
      // local spin + radial swirl, then translate back to the point
      float rr = length(lp);
      float aa = atan(lp.y, lp.x) + swirl * rr;
      vec2 sp = rr * vec2(cos(aa), sin(aa));
      sp = rot * sp;
      acc += (c + sp) * w;
      wsum += w;
    }
    if (wsum < 1e-4) return uv;
    vec2 q = acc / wsum;
    q.x /= aspect;
    return q + 0.5;
  }

  // TRIANGLE KALEIDOSCOPE: the classic equilateral-triangle mirror fold (the
  // symmetry group p6m / *632 generated by reflecting across the triangle's
  // three edges). Its mirror lines meet at the triangle VERTICES — so the seams
  // converge at the three corners (each a 6-fold radial star), NOT at the screen
  // center. Standard reflection sequence into the 30-60-90 fundamental domain:
  //   fold |y|, then reflect across the two 60° lines repeatedly.
  // scale sets size; rot orbits the whole figure.
  vec2 triKaleido(vec2 uv, float aspect, float scale, float rot) {
    vec2 p = (uv - 0.5);
    p.x *= aspect;
    float cs = cos(rot), sn = sin(rot);
    p = mat2(cs, -sn, sn, cs) * p;
    p /= scale;
    // hex/triangle reflection: the two mirror lines at ±30° from vertical.
    const vec2 n1 = vec2(-0.5, 0.8660254);   // normal of +60° mirror
    const vec2 n2 = vec2( 0.5, 0.8660254);   // normal of -60° mirror
    p.y = abs(p.y);                          // mirror across the horizontal edge
    p -= 2.0 * min(dot(p, n1), 0.0) * n1;    // fold across +60° mirror
    p -= 2.0 * min(dot(p, n2), 0.0) * n2;    // fold across -60° mirror
    p -= 2.0 * min(dot(p, n1), 0.0) * n1;    // second pass closes the triangle
    p *= scale;
    p.x /= aspect;
    return p + 0.5;
  }

  // a spinning radial-kaleido rosette around center c (each focal point looks
  // like its own little kaleidoscope). spinPhase is SHARED across all centers so
  // the rosettes stay identical = symmetric copies, no fragmentation.
  vec2 rosette(vec2 uv, vec2 c, float segPerPoint, float spinPhase, float swirl) {
    vec2 q = uv - c;
    float r = length(q);
    float a = atan(q.y, q.x) + spinPhase + swirl * r;
    float seg_a = TAU / max(segPerPoint, 1.0);
    a = mod(a, seg_a);
    a = abs(a - seg_a * 0.5);
    return c + r * vec2(cos(a), sin(a));
  }

  // N RADIAL CENTERS orbiting each other: nPts true radial rosettes evenly on a
  // ring of radius rad, the ring rotating by orbit (so the centers circle each
  // other). Each center is the SAME segPerPoint-fold rosette (shared spin) → all
  // identical = coherent/symmetric. Tight blend (high sharp) keeps the centers
  // crisp and well-separated, not smeared. rad=0 ⇒ they coincide (one center).
  vec2 orbitCenters(vec2 uv, float aspect, float nPts, float rad, float orbit,
                    float segPerPoint, float spinPhase, float sharp) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    vec2 acc = vec2(0.0);
    float wsum = 0.0;
    for (int k = 0; k < 9; k++) {
      if (float(k) >= nPts) break;
      float ang = orbit + TAU * float(k) / nPts + PI * 0.5; // +90° → one points up
      vec2 c = rad * vec2(cos(ang), sin(ang));
      float d = length(p - c);
      float w = exp(-d * sharp);                 // tight, well-separated centers
      vec2 folded = rosette(p, c, segPerPoint, spinPhase, 0.0); // true radial fold
      acc += folded * w;
      wsum += w;
    }
    if (wsum < 1e-4) return uv;
    vec2 q = acc / wsum;
    q.x /= aspect;
    return q + 0.5;
  }

  // SYM CELLS — generalizes symSplit to 2/3/4/9 + orbital rings, ALL exact
  // mirror-symmetric copies (each focal point is the SAME spinning rosette), so
  // the frame stays coherent (no per-cell fragmentation). Two layouts:
  //   layout 0 = GRID: abs()-fold into a cell whose origin is at (dx,dy). nx,ny
  //     pick 1/2/3 divisions per axis → 1,2,4,6,9 grids (signed-distance fold,
  //     so every cell is a true mirror copy). dx/dy=0 ⇒ consolidated (count can
  //     change invisibly), matching the smooth-transition rule.
  //   layout 1 = RING: N points evenly on a ring of radius dx (3=triangle, etc.)
  //     via an angular fold; ringRot orbits the whole ring. Perfectly N-fold.
  // spinPhase (shared) spins every rosette together; swirl twists arms.
  vec2 symCells(vec2 uv, float aspect, float lyt, float nAxisX, float nAxisY,
                float dx, float dy, float ringRot, float spinPhase, float swirl,
                float segPerPoint) {
    vec2 p = uv - 0.5;
    p.x *= aspect;
    if (lyt < 0.5) {
      // GRID: per-axis symmetric fold to the nearest of the axis centers.
      // nAxisX=1 → center {0}; =2 → {-dx,+dx} (bisect); =3 → {-dx,0,+dx} (trisect)
      float cx = 0.0, cy = 0.0;
      if (nAxisX >= 1.5) {
        float a = abs(p.x);
        cx = (nAxisX >= 2.5 && a < dx * 0.5) ? 0.0 : sign(p.x) * dx;
      }
      if (nAxisY >= 1.5) {
        float a = abs(p.y);
        cy = (nAxisY >= 2.5 && a < dy * 0.5) ? 0.0 : sign(p.y) * dy;
      }
      vec2 c = vec2(cx, cy);
      vec2 q = rosette(p, c, segPerPoint, spinPhase, swirl);
      q.x /= aspect;
      return q + 0.5;
    } else {
      // RING: fold the plane into one of N angular wedges, each holding a copy of
      // the SAME rosette sitting at radius dx on the wedge centerline → N points
      // evenly on a ring. abs-fold within the wedge keeps mirror symmetry.
      float N = max(nAxisX, 1.0);
      float r = length(p);
      float ang = atan(p.y, p.x) - ringRot;        // orbit the whole ring
      float seg = TAU / N;
      ang = mod(ang, seg);
      ang = abs(ang - seg * 0.5);                   // mirror within wedge
      vec2 pf = r * vec2(cos(ang), sin(ang));
      vec2 c = vec2(dx, 0.0);                        // focal point on wedge centerline
      vec2 q = rosette(pf, c, segPerPoint, spinPhase, swirl);
      q.x /= aspect;
      return q + 0.5;
    }
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
    else if (k == 9) { float w; return symMorph(uv, aspect, pts, orbitRadius, bubbleRate, rotSpeed, driftAmt, seg, w); }
    return uv; // 0 = off
  }

  void main() {
    float aspect = uResolution.x / uResolution.y;
    float m = smoothstep(0.0, 1.0, uMix);

    vec3 col;
    // DIVIDE-AND-MOVE: a single continuous fold (no crossfade, no fade) — the
    // journey eases dx/dy/kx/ky so points divide and slide apart / merge back.
    // uA slots: dx=uA[10], dy=uA[12], kx=uA[5], ky=uA[6], spinPhase=uA[4], swirl=uA[7]
    // uDivide: 1 = v1 (divide-and-move, one global rotation); 2 = symCells
    //   (symmetric multi-rosette: grid/ring, each focal point a shared-phase
    //    spinning rosette). v2 slots: layout=uA[2], nAxisX=uA[5], nAxisY=uA[6],
    //    dx=uA[10], dy=uA[12], ringRot=uA[11], spinPhase=uA[4], swirl=uA[7], seg=uA[1]
    if (uDivide > 0.5) {
      // v1 divide slots: dx=10, dy=12, kx=5, ky=6, worldRot=11, spinPhase=4,
      //   swirl=7, innerSeg=1. uA[3] = triangle/ring point count (0 ⇒ grid).
      // grid and triangle both collapse to one point at center, so switching
      // which fold runs (committed at consolidation) is invisible — no seam.
      // uA[2] = seamStrength (0 = original drawing, NO seams; 1 = full fold).
      vec2 uv;
      if (uDivide > 2.5) {
        // RADIAL-BASE hybrid (auto · radial): the BASE is a clean radial fold
        // (uA[9] segments) — a symmetric home like auto·symmetry — and the bloom
        // (grid/triangle/radial) EMERGES from it via seam. seam 0 = pure clean
        // radial home (bloom hidden); seam 1 = the bloom (base hidden). So the
        // base radial seg can swap invisibly at seam 1, and the bloom structure
        // at seam 0 — both directions clean, always consolidates to radial.
        float seam = smoothstep(0.0, 1.0, uA[2]);
        vec2 baseR = radial(vUv, max(uA[9], 2.0), uA[11], aspect);
        int bt = int(uA[8] + 0.5);
        vec2 folded;
        if (bt == 2)      folded = radial(vUv, max(uA[3], 2.0), uA[11], aspect);
        // bt 1 = N-fold MORPH (the auto·symmetry triangle): N=uA[3] mirror-
        //   symmetric lobes (each mirrored down its own centerline — GUARANTEED
        //   symmetric, no lopsided blobs), a radial focal point at focusR=uA[10]
        //   on each lobe, the whole figure orbiting via globalPhase=uA[11].
        // bt 1 = TRIANGLE KALEIDOSCOPE: equilateral-triangle mirror fold whose
        //   seams meet at the three VERTICES (not the center). scale=uA[10],
        //   rot=uA[11] (worldRot+orbit).
        else if (bt == 1) folded = triKaleido(vUv, aspect, uA[10], uA[11]);
        // GRID bloom: pass worldRot=0 so the fold seams stay axis-aligned
        // (horizontal/vertical), never diagonal. The whole-field swing still
        // lives in the radial base + spin; the grid itself stays square.
        else              folded = divideMove(vUv, aspect, uA[10], uA[12], uA[5], uA[6], 0.0, uA[4], uA[7], uA[1]);
        uv = mix(baseR, folded, seam);
      } else if (uDivide > 1.5) {
        uv = symCells(vUv, aspect, uA[2], uA[5], uA[6], uA[10], uA[12], uA[11], uA[4], uA[7], uA[1]);
      } else {
        // SEAM-EMERGENCE model: the rest state is the ORIGINAL drawing (no
        // seams). seamStrength blends original→folded, so seams EMERGE from the
        // original (0→1), COLLAPSE back into it (1→0). At strength 0 EVERY
        // pattern (grid/triangle/any count) looks identical (the original), so
        // structure swaps there are perfectly invisible — no jagged jumps.
        float seam = smoothstep(0.0, 1.0, uA[2]);
        // whole-field rotation applies to the original too (rotate, then unfold)
        vec2 base = vUv - 0.5; base.x *= aspect;
        float wc = cos(uA[11]), ws = sin(uA[11]);
        base = mat2(wc, -ws, ws, wc) * base;
        vec2 original = base; original.x /= aspect; original += 0.5;
        // bloom type: uA[8] → 0 grid, 1 triangle/ring, 2 RADIAL (true central
        // radial kaleidoscope, N=uA[3] segments). Radial is the home look.
        int bt = int(uA[8] + 0.5);
        vec2 folded;
        if (bt == 2)      folded = radial(vUv, max(uA[3], 2.0), uA[11], aspect);
        else if (bt == 1) folded = triPoints(vUv, aspect, uA[3], uA[10], uA[11], uA[4], uA[7], uA[1]);
        else              folded = divideMove(vUv, aspect, uA[10], uA[12], uA[5], uA[6], uA[11], uA[4], uA[7], uA[1]);
        uv = mix(original, folded, seam);
      }
      col = texture2D(tTrail, uv).rgb;
      col *= uExposure;
      col = col / (col + vec3(1.0));
      float vigD = smoothstep(1.1, 0.25, length(vUv - 0.5));
      col *= mix(0.55, 1.0, vigD);
      col = pow(col, vec3(0.4545));
      col = clamp((col - 0.5) * uContrast + 0.5, 0.0, 1.0);
      gl_FragColor = vec4(col, 1.0);
      return;
    }
    bool bothMorph = int(uA[0] + 0.5) == 9 && int(uB[0] + 0.5) == 9;
    // NOTE: do NOT also gate on uMix here. When a fold-count crossfade begins,
    // uMix resets to 0 for one frame; if that frame fell through to the mapK
    // branch (which doesn't apply the seam feather) the seams would flash bright
    // for a single frame every structure change. Always take the feathered path
    // when both sets are symMorph — at uMix=0 it just shows set A, feathered.
    if (bothMorph) {
      // STRUCTURE crossfade. The continuous params (focusR, rotation, swirl) are
      // shared/lerped so the underlying flow is one continuous field; only the
      // FOLD STRUCTURE (point count N + regional fold) differs between A and B.
      // We render the fold both ways — each with an INTEGER N (never fractional,
      // which would make a sweeping seam) — and dissolve the results. So a
      // count change (8→6) eases in as one symmetry quietly giving way to the
      // other over the same flowing particles, instead of an instant re-tile.
      float focusR = mix(uA[10], uB[10], m);
      float gPhase = mix(uA[4],  uB[4],  m);
      float spin   = mix(uA[7],  uB[7],  m);
      float swirl  = mix(uA[5],  uB[5],  m);
      vec3 cA, cB;
      if (uRotational > 0.5) {
        // (E) rotational: no mirror fold → no seam, no feather needed
        vec2 uvA = rotMorph(vUv, aspect, uA[3], focusR, gPhase, spin, swirl, uA[1]);
        vec2 uvB = rotMorph(vUv, aspect, uB[3], focusR, gPhase, spin, swirl, uB[1]);
        cA = texture2D(tTrail, uvA).rgb;
        cB = texture2D(tTrail, uvB).rgb;
      } else {
        // (A+B) mirror fold with feather: symMorph returns a seam-weight that
        // dims the doubled reflected streak so it fades instead of glowing.
        float wA, wB;
        vec2 uvA = symMorph(vUv, aspect, uA[3], focusR, gPhase, spin, swirl, uA[1], wA);
        vec2 uvB = symMorph(vUv, aspect, uB[3], focusR, gPhase, spin, swirl, uB[1], wB);
        cA = texture2D(tTrail, uvA).rgb * wA;
        cB = texture2D(tTrail, uvB).rgb * wB;
      }
      col = mix(cA, cB, m);
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
  onModeChange: (name: string) => void = () => {}; // active base mode (for the menu)
  onSeamChange: (seam: number, blur: number) => void = () => {}; // seam soften + trail blur

  /** The full list of selectable mode names, in menu order. */
  static modeNames(): string[] {
    return KALEIDO_MODES.map((m) => m.name);
  }

  /** Select a mode by its name (used by the click menu). */
  selectMode(name: string) {
    const i = KALEIDO_MODES.findIndex((m) => m.name === name);
    if (i >= 0) this.setKaleido(i);
  }

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
  private fieldSpeed = 0.99;
  // particle breathing: cycle rate + peak size (px). min stays 1.
  private breatheRate = 0.15;
  private breathePeak = 5;
  // base brightness; music modulates around it
  private intensity = 0.4;
  // seam treatment + trail roundness (live-tunable via keys)
  private seamSoft = 0.6;
  private trailBlur = 0.5;

  // kaleidoscope crossfade: uAVals shown when mix==0, dissolving toward uBVals.
  private uAVals: number[] = new Array(15).fill(0);
  private uBVals: number[] = new Array(15).fill(0);
  private modeA: KaleidoMode = KALEIDO_MODES[0];
  private modeB: KaleidoMode | null = null;
  private mix = 0;
  private mixDur = 1.2;
  // auto-choreography driver
  private autoMode: "cycle" | "music" | "symmetry" | "surface" | "pinwheel" | "divide" | "dividev2" | "radialdiv" | null = null;
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
  private symNextBloom: SymPose = { numPoints: 2, focusR: 0, spin: 0, swirl: 0, circle: 0, localSeg: 6 };
  private symGoalN = 2; // structure we WANT (committed only while consolidated)
  private symGoalSeg = 6;
  private symPhase = 0; // accumulated whole-figure rotation (radians)
  private symSpin = 0; // accumulated regional spin (radians)
  // fold-structure crossfade: dissolve old N/fold → new over symStructMix
  private symStructN = 2; // current (B) fold count
  private symStructSeg = 6; // current (B) regional fold
  private symOldN = 2; // previous (A) fold count being faded out
  private symOldSeg = 6; // previous (A) regional fold
  private symStructMix = 1; // 0→1 crossfade of A(old)→B(new) structure (1 = settled)
  // divide-and-move journey driver (pure continuous, no crossfade)
  private divFrom: DivPose = { dx: 0, dy: 0, kx: 0, ky: 0, spin: 0, swirl: 0, layout: 0, nx: 1, ny: 1, seg: 6 , bloomType: 2, radSeg: 6, seam: 0 };
  private divTo: DivPose = { dx: 0, dy: 0, kx: 0, ky: 0, spin: 0, swirl: 0, layout: 0, nx: 1, ny: 1, seg: 6 , bloomType: 2, radSeg: 6, seam: 0 };
  private divCur: DivPose = { dx: 0, dy: 0, kx: 0, ky: 0, spin: 0, swirl: 0, layout: 0, nx: 1, ny: 1, seg: 6 , bloomType: 2, radSeg: 6, seam: 0 };
  private divT = 1;
  private divHold = 0;
  private divMoveDur = 4;
  private divAtBloom = false;
  private divSpin = 0; // accumulated regional spin (v1 + v2 primary)
  private divSpin2 = 0; // v2: second (often opposite) per-cell spin phase
  private divIsV2 = false;
  private divIsRadial = false; // auto · radial: clean radial base instead of original
  private divBaseSeg = 8; // base-radial fold count (radialdiv home)
  private divWorld = 0; // accumulated whole-screen rotation (v1)
  private divWorldDir = 1; // current whole-screen rotation direction (±1)
  private divHomeSeg = 6; // radial-home fold count (kept stable across blooms)
  private divOrbit = 0; // accumulated orbit for the centers-circling bloom
  // worldRot is eased from→to per leg, landing on a L/R-symmetric angle (radial)
  private divWorldFrom = 0;
  private divWorldTo = 0;

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
  private paletteIndex = Math.max(0, PALETTES.findIndex((p) => p.name === "Magma"));
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
    this.setPalette(this.paletteIndex, true); // honor the default palette (Magma)
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
        uWorldStretch: { value: 1.0 },
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
        uTrailBlur: { value: 0.0 },
        uTexel: { value: new THREE.Vector2(1 / 2, 1 / 2) },
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
        uSeamSoft: { value: 0.6 },
        uRotational: { value: 0 },
        uDivide: { value: 0 },
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
    // widen the (isotropic) world to fill a landscape screen. aspect^0.75 fills
    // most of the width while keeping swirls only mildly elliptical (full aspect
    // would squash them). Never below 1 (don't squeeze a tall window).
    const stretch = Math.max(1, Math.pow(Math.max(1, w / h), 0.75));
    this.pointsMat.uniforms.uWorldStretch.value = stretch;
    this.fadeMat.uniforms.uTexel.value.set(1 / (w * dpr), 1 / (h * dpr));
  };

  private onKey = (e: KeyboardEvent) => {
    // NOTE: mode is chosen ONLY via the menu buttons now — no key cycles it, so
    // it never changes unexpectedly. (The old 'K' cycle was removed.)
    if (e.key === "p" || e.key === "P") {
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
    } else if (e.key === "s") {
      this.setSeamSoft(this.seamSoft + 0.15); // more seam feather/wavy
    } else if (e.key === "S") {
      this.setSeamSoft(this.seamSoft - 0.15);
    } else if (e.key === "b") {
      this.setTrailBlur(this.trailBlur + 0.5); // softer (rounder) trails
    } else if (e.key === "B") {
      this.setTrailBlur(this.trailBlur - 0.5);
    }
  };

  /** Seam feathering / wavy-seam amount (0 = hard mirror lines, ~1 = soft). */
  private setSeamSoft(v: number) {
    this.seamSoft = Math.min(1.5, Math.max(0, v));
    this.displayMat.uniforms.uSeamSoft.value = this.seamSoft;
    this.onSeamChange(this.seamSoft, this.trailBlur);
  }

  /** Isotropic trail blur in texels (0 = crisp streaks, higher = rounder ink). */
  private setTrailBlur(v: number) {
    this.trailBlur = Math.min(4, Math.max(0, v));
    this.fadeMat.uniforms.uTrailBlur.value = this.trailBlur;
    this.onSeamChange(this.seamSoft, this.trailBlur);
  }

  /** Switch kaleidoscope entry by index (wraps). Auto entries start a driver. */
  private setKaleido(index: number) {
    this.kaleidoIndex = ((index % KALEIDO_MODES.length) + KALEIDO_MODES.length) % KALEIDO_MODES.length;
    const m = KALEIDO_MODES[this.kaleidoIndex];
    this.onModeChange(m.name); // tell the menu which base mode is active
    if (m.auto) {
      this.startAuto(m.auto, m.name);
      return;
    }
    this.autoMode = null;
    this.setSurfaceMode(false); // leaving surface → restore legacy field + draw
    this.displayMat.uniforms.uDivide.value = 0; // leaving divide
    this.displayMat.uniforms.uRotational.value = 0;
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
  private startAuto(kind: "cycle" | "music" | "symmetry" | "surface" | "pinwheel" | "divide" | "dividev2" | "radialdiv", label: string) {
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
    // divide family: a single continuous fold, no crossfade. Seed set A.
    // uDivide: 1 = divide (orig base), 2 = split (symCells), 3 = radial (radial base).
    this.displayMat.uniforms.uDivide.value =
      kind === "dividev2" ? 2 : kind === "divide" ? 1 : kind === "radialdiv" ? 3 : 0;
    if (kind === "divide" || kind === "dividev2" || kind === "radialdiv") {
      this.divIsV2 = kind === "dividev2";
      this.divIsRadial = kind === "radialdiv";
      this.displayMat.uniforms.uRotational.value = 0;
      this.modeB = null;
      this.mix = 0;
      this.displayMat.uniforms.uMix.value = 0;
      this.modeA = km(label, { kind: 0 });
      this.fillMode(this.modeA, this.uAVals);
      this.divSpin = 0;
      this.divSpin2 = 0;
      this.divWorld = Math.PI / 2; // start on a L/R-symmetric angle
      this.divWorldFrom = this.divWorld;
      this.divWorldTo = this.divWorld;
      this.divWorldDir = Math.random() < 0.5 ? -1 : 1;
      this.divBaseSeg = [4, 5, 6, 6, 8][Math.floor(Math.random() * 5)];
      this.divFrom = { dx: 0, dy: 0, kx: 0, ky: 0, spin: 0.05, swirl: 0.4, layout: 0, nx: 1, ny: 1, seg: 1 , bloomType: 2, radSeg: 6, seam: 0 };
      this.divCur = { ...this.divFrom };
      this.divTo = this.divFrom;
      this.divAtBloom = false; // first update opens into a bloom
      this.divT = 1;
      this.divHold = 0;
      this.updateDivide(0);
      this.emitKaleidoLabel();
      return;
    }
    // pinwheel = symmetry journey but with the rotational (seamless) fold
    this.displayMat.uniforms.uRotational.value = kind === "pinwheel" ? 1 : 0;
    if (kind === "symmetry" || kind === "pinwheel") {
      // one continuously-morphing kind-9 field driven by updateSymJourney();
      // no A/B crossfade needed (params animate every frame). Seed set A.
      this.symPhase = 0;
      this.symSpin = 0;
      this.modeB = null;
      this.mix = 0;
      this.displayMat.uniforms.uMix.value = 0;
      this.modeA = km(label, { kind: 9, segments: 8 });
      this.fillMode(this.modeA, this.uAVals);
      // start consolidated at center, then open into a freshly-rolled bloom
      this.symFrom = this.centerPose();
      this.symCur = { ...this.symFrom };
      this.symTo = this.symFrom;
      this.symNextBloom = this.randomBloom(); // the bloom the first update opens into
      this.symAtBloom = false; // next arrival is a bloom
      this.symT = 1; // force an immediate retarget on first update
      this.symHold = 0;
      // begin already showing the first bloom's structure (we're consolidated, so
      // committing it now is invisible) — avoids a 1st-frame structure mismatch.
      this.symStructN = this.symOldN = this.symGoalN = this.symNextBloom.numPoints;
      this.symStructSeg = this.symOldSeg = this.symGoalSeg = this.symNextBloom.localSeg;
      this.symStructMix = 1;
      this.updateSymJourney(0); // write initial params
    }
    this.emitKaleidoLabel();
  }

  /** A calm consolidated pose: all points pulled to the center. Point count is
   *  carried through so it can change invisibly while collapsed. */
  private centerPose(numPoints = 2): SymPose {
    return { numPoints, focusR: 0.0, spin: 0.04, swirl: 0.5, circle: 0.0, localSeg: 6 };
  }

  /** Roll a fresh random bloom pose. Picks a distinct ARCHETYPE each time so
   *  blooms feel genuinely different (a tight grid vs a wide slow drift vs a fast
   *  ornate pinwheel) rather than variations on one look — combined with the wide
   *  point-count set it rarely repeats. All smooth: N/localSeg only ever swap at
   *  the consolidated center, the rest are continuously eased. */
  private randomBloom(): SymPose {
    const rnd = Math.random;
    const N = SYM_POINT_COUNTS[Math.floor(rnd() * SYM_POINT_COUNTS.length)];
    const dir = rnd() < 0.5 ? -1 : 1;
    // base ring radius — higher N sits a bit tighter so points don't overlap
    const baseR = N <= 3 ? 0.27 : N <= 6 ? 0.23 : N <= 9 ? 0.20 : 0.17;

    // 5 archetypes, each with its own motion + regional-fold character
    const arche = Math.floor(rnd() * 5);
    let focusR = baseR, spin = 0.08, swirl = 1.5, circle = 0.05, localSeg = 4;
    if (arche === 0) {
      // GRID / still — crisp, low motion, simple regional mirror
      focusR = baseR * (0.9 + rnd() * 0.2);
      spin = 0.02 + rnd() * 0.04; swirl = 0.4 + rnd() * 0.8;
      circle = dir * (0.01 + rnd() * 0.03); localSeg = [2, 4][Math.floor(rnd() * 2)];
    } else if (arche === 1) {
      // PINWHEEL — fast spin + strong circle, energetic
      spin = (0.14 + rnd() * 0.18) * dir; swirl = 1.5 + rnd() * 2.0;
      circle = dir * (0.08 + rnd() * 0.10); localSeg = [3, 4, 6][Math.floor(rnd() * 3)];
    } else if (arche === 2) {
      // ROSETTE — wide, ornate regional fold, gentle
      focusR = baseR * (1.05 + rnd() * 0.25);
      spin = 0.03 + rnd() * 0.06; swirl = 2.0 + rnd() * 2.5;
      circle = dir * (0.02 + rnd() * 0.05); localSeg = [6, 8, 10][Math.floor(rnd() * 3)];
    } else if (arche === 3) {
      // DRIFT — slow, big, hypnotic
      focusR = baseR * (1.0 + rnd() * 0.3);
      spin = (0.02 + rnd() * 0.05) * dir; swirl = 0.8 + rnd() * 1.4;
      circle = dir * (0.03 + rnd() * 0.05); localSeg = [3, 4, 5][Math.floor(rnd() * 3)];
    } else {
      // SWIRL — moderate everything, twisty arms
      spin = (0.06 + rnd() * 0.10) * dir; swirl = 2.4 + rnd() * 2.2;
      circle = dir * (0.04 + rnd() * 0.08); localSeg = [4, 5, 6][Math.floor(rnd() * 3)];
    }
    return { numPoints: N, focusR, spin, swirl, circle, localSeg };
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
          // leaving a bloom → head to center. Roll the NEXT bloom NOW and record
          // its structure as the GOAL; we only COMMIT the structure swap once the
          // points have collapsed to the center (focusR≈0), below.
          this.symNextBloom = this.randomBloom();
          this.symGoalN = this.symNextBloom.numPoints;
          this.symGoalSeg = this.symNextBloom.localSeg;
          this.symTo = this.centerPose(this.symGoalN);
          this.symMoveDur = 3.5 + Math.random() * 2.5;
          this.symHold = 3 + Math.random() * 6; // pause at center (longer sometimes)
          this.symAtBloom = false;
        } else {
          // leaving center → open into the pre-rolled bloom (structure already
          // settled to its N during the consolidated dwell, so opening is clean).
          this.symTo = this.symNextBloom;
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
    // CONTINUOUS params flow smoothly — these are what visibly morphs. Keep them
    // in RAW pose space (no energy boost here): symFrom is a copy of symCur at
    // each leg boundary, so any factor baked in here would COMPOUND every leg and
    // pop focusR by that factor at the boundary. The energy boost is applied only
    // at the uniform write below.
    cur.focusR = L(f.focusR, g.focusR); // open/close (raw)
    cur.swirl = L(f.swirl, g.swirl);   // swirl*radius, no uTime amplification → safe
    cur.circle = L(f.circle, g.circle); // rate fed into accumulated symPhase → safe
    cur.spin = L(f.spin, g.spin);       // rate fed into accumulated symSpin → safe

    // DISCRETE fold structure (N, localSeg). A post-process fold can't lerp these,
    // and crossfading two distinct folds is only invisible while the points are
    // COLLAPSED at the center (all fold counts look like one blob). So we ONLY
    // commit a structure change when focusR is below a small threshold, and fade
    // it FAST so it completes before the points open up again — no ghost double-
    // exposure of two fold counts at a visible radius.
    const consolidated = cur.focusR < 0.06;
    if (consolidated && (this.symGoalN !== this.symStructN || this.symGoalSeg !== this.symStructSeg)) {
      this.symOldN = this.symStructN;
      this.symOldSeg = this.symStructSeg;
      this.symStructN = this.symGoalN;
      this.symStructSeg = this.symGoalSeg;
      this.symStructMix = 0;
    }
    if (this.symStructMix < 1) {
      this.symStructMix = Math.min(1, this.symStructMix + dt / 0.9); // fast (~0.9s) at center
    }

    // advance rotation PHASES from their rates (accumulation makes rate changes
    // jump-free): symPhase = whole-figure circling, symSpin = regional spin.
    this.symPhase += dt * cur.circle * TAU;
    this.symSpin += dt * cur.spin * TAU;

    // write kind-9 slots into BOTH sets. Continuous params are identical (same
    // flowing field); only the fold structure differs — A=old, B=new — and uMix
    // dissolves between them. After the fade A==B so it's a no-op until next N.
    const A = this.uAVals, B = this.uBVals;
    const mix = this.symStructMix * this.symStructMix * (3 - 2 * this.symStructMix);
    // apply the energy boost HERE (not into symCur) so it never compounds across
    // legs. Smoothed autoEnergy → no per-frame jitter in the rendered radius.
    const focusOut = cur.focusR * (1 + this.autoEnergy * 0.22);
    for (const u of [A, B]) {
      u[0] = 9;
      u[10] = focusOut;
      u[4] = this.symPhase;
      u[7] = this.symSpin;
      u[5] = cur.swirl;
    }
    A[3] = this.symOldN; A[1] = this.symOldSeg;
    B[3] = this.symStructN; B[1] = this.symStructSeg;
    this.displayMat.uniforms.uMix.value = mix;
  }

  /** The HOME pose: a gentle central RADIAL kaleidoscope (the most compelling
   *  symmetric look). The journey rests here between blooms — NOT on the bare
   *  original (that's a rare treat, see updateDivide). Low-ish segment count,
   *  partial seam so it reads as a soft radial, not a hard kaleidoscope. */
  private radialHome(): DivPose {
    const r = Math.random;
    const N = [4, 6, 6, 8][Math.floor(r() * 4)];
    return {
      dx: 0, dy: 0, kx: 0, ky: 0,
      spin: (0.03 + r() * 0.05) * (r() < 0.5 ? -1 : 1),
      swirl: 0, layout: 0, nx: 0, ny: 1, seg: 1,
      bloomType: 2, radSeg: N, seam: 0.7,   // radial, rests partly folded
    };
  }

  /** Roll a fresh divide bloom: GRID (per-axis split → 1,2,3,4,6,9), TRIANGLE/
   *  ring (3/5/6 soft points), or RADIAL (higher-N central kaleido). All bloom
   *  to full seam (1.0); the home they return to is a gentle radial. */
  private randomDivBloom(): DivPose {
    const r = Math.random;
    const roll = r();
    if (roll < 0.30) {
      // RADIAL bloom — true central radial symmetry, higher fold count
      const N = [5, 6, 8, 10, 12][Math.floor(r() * 5)];
      return {
        dx: 0, dy: 0, kx: 0, ky: 0,
        spin: (0.04 + r() * 0.10) * (r() < 0.5 ? -1 : 1),
        swirl: 0, layout: 0, nx: 0, ny: 1, seg: 1,
        bloomType: 2, radSeg: N, seam: 1.0,
      };
    }
    if (roll < 0.52) {
      // TRIANGLE / ring bloom (seamless soft points). nx = point count.
      const N = [3, 3, 3, 5, 6][Math.floor(r() * 5)];
      return {
        dx: 0.22 + r() * 0.10, dy: 0, kx: 0, ky: 0,
        spin: (0.05 + r() * 0.16) * (r() < 0.5 ? -1 : 1),
        swirl: 0.3 + r() * 1.6,
        layout: 0, nx: N, ny: 1, seg: 4.5 + r() * 4.0,
        bloomType: 1, radSeg: 6, seam: 1.0,
      };
    }
    // GRID bloom (per-axis split + bisect/trisect)
    const splitX = r() < 0.85;
    const splitY = r() < 0.7;
    const kx = splitX && r() < 0.45 ? 1 : 0;
    const ky = splitY && r() < 0.45 ? 1 : 0;
    const innerSeg = r() < 0.55 ? [2, 3, 4, 6][Math.floor(r() * 4)] : 1;
    return {
      dx: splitX ? 0.18 + r() * 0.16 : 0,
      dy: splitY ? 0.16 + r() * 0.16 : 0,
      kx, ky,
      spin: (0.06 + r() * 0.20) * (r() < 0.5 ? -1 : 1),
      swirl: 0.4 + r() * 2.2,
      layout: 0, nx: 0, ny: 1, seg: innerSeg,
      bloomType: 0, radSeg: 6, seam: 1.0,
    };
  }

  /** Roll a fresh bloom for auto · radial. The point is VARIETY of DESIGN, not a
   *  spinning circle — so blooms favor visually DISTINCT shapes: low-count
   *  radials (triangle/square/pentagon/hexagon, which look very different) and
   *  clean axis-aligned grids/quads. Spin is LOW (occasional, not constant — the
   *  evolution comes from the morphing transitions, not from spinning). All
   *  strictly symmetric; grid seams stay H/V (shader passes worldRot=0). */
  /** Number of L/R-symmetric rotation positions per π for a pose (its mirror
   *  lines sit every π/N; a vertical mirror = L/R-symmetric screen). */
  private divSymN(p: DivPose): number {
    return p.bloomType > 1.5 ? Math.max(2, p.radSeg)   // radial
         : p.bloomType > 0.5 ? 3                        // triangle (p6m)
         : Math.max(2, this.divBaseSeg);                // grid → radial base
  }

  /** Pick the worldRot DESTINATION for a leg: continue turning from the current
   *  angle by ~`turn` radians, then snap to the nearest L/R-symmetric angle for
   *  pose `p` (rot ≡ π/2 mod π/N). Returns an absolute target we ease straight to
   *  — so it ends exactly square, no drift-then-correct. */
  private divWorldTarget(p: DivPose, turn: number): number {
    const step = Math.PI / this.divSymN(p);
    const want = this.divWorld + turn * this.divWorldDir;
    return Math.round((want - Math.PI / 2) / step) * step + Math.PI / 2;
  }

  private randomRadialBloom(): DivPose {
    const r = Math.random;
    // low, often near-zero spin so it doesn't get dizzying; occasionally livelier
    const spin = (r() < 0.5 ? 0.0 : 0.02 + r() * 0.06) * (r() < 0.5 ? -1 : 1);
    const roll = r();
    if (roll < 0.62) {
      // RADIAL design — counts that read as DISTINCT shapes (4 square,
      // 5 pentagon, 6 hexagon, 8 octagon). FLOOR OF 4: a 2- or 3-fold radial
      // reads as a lopsided single/double mirror, not a kaleidoscope. Avoid very
      // high counts too (all look like the same circle).
      const N = [4, 4, 5, 6, 6, 8][Math.floor(r() * 6)];
      return {
        dx: 0, dy: 0, kx: 0, ky: 0,
        spin, swirl: 0, layout: 0, nx: 0, ny: 1, seg: 1,
        bloomType: 2, radSeg: N, seam: 1.0,
      };
    }
    // NOTE: the equilateral TRIANGLE KALEIDOSCOPE (bloomType 1) was dropped from
    // this mode. Its mirror lines meet at the triangle vertices, but the particle
    // cloud is centrally concentrated and the centroid sits at screen center — so
    // at any usable scale only the central horizontal mirror was visible and it
    // degenerated to a "two-way split" (one mirror). The radial + grid blooms are
    // reliably centered and stay >= four-way symmetric.
    // CLEAN GRID / QUAD / SPLIT — both axes split (balanced), seams H/V.
    // kx/ky pick 2 vs 3 divisions per axis → 2×2, 2×3, 3×3 etc.
    const kx = r() < 0.4 ? 1 : 0;
    const ky = r() < 0.4 ? 1 : 0;
    const innerSeg = r() < 0.5 ? [2, 4, 6][Math.floor(r() * 3)] : 1;
    return {
      dx: 0.18 + r() * 0.14,
      dy: 0.18 + r() * 0.14,
      kx, ky,
      spin, swirl: 0.3 + r() * 1.0,
      layout: 0, nx: 0, ny: 1, seg: innerSeg,
      bloomType: 0, radSeg: 6, seam: 1.0,
    };
  }

  /** Roll a fresh SPLIT bloom (v2 = symCells): symmetric multi-rosette. Picks a
   *  GRID (3 as a row, 4=2×2, 6=2×3, 9=3×3) or a RING (3=triangle, 5/6/8 ring),
   *  per user: 3 as triangle (ring), 9 as 3×3 grid. Each focal point is a
   *  shared-phase spinning rosette → exact symmetric copies, coherent. */
  private randomSplitBloom(): DivPose {
    const r = Math.random;
    const seg = [3, 4, 6][Math.floor(r() * 3)];           // rosette fold per point
    const spin = (0.04 + r() * 0.10) * (r() < 0.5 ? -1 : 1);
    const swirl = 0.4 + r() * 1.6;
    const ring = r() < 0.45; // ~45% ring (triangles/orbital), else grid
    if (ring) {
      const N = [3, 3, 5, 6, 8][Math.floor(r() * 5)];     // triangle-weighted
      const dx = 0.20 + r() * 0.10;
      return { dx, dy: 0, kx: 0, ky: 0, spin, swirl, layout: 1, nx: N, ny: 1, seg, bloomType: 0, radSeg: 6, seam: 1 };
    } else {
      // grid: nx,ny ∈ {1,2,3}; bias to give 2×2, 3×3, 2×3
      const nx = [2, 2, 3][Math.floor(r() * 3)];
      const ny = [2, 3, 3][Math.floor(r() * 3)];
      const dx = nx === 1 ? 0 : 0.17 + r() * 0.10;
      const dy = ny === 1 ? 0 : 0.16 + r() * 0.10;
      return { dx, dy, kx: 0, ky: 0, spin, swirl, layout: 0, nx, ny, seg, bloomType: 0, radSeg: 6, seam: 1 };
    }
  }

  /** Consolidated center pose for split mode (everything collapsed). */
  private splitCenterPose(carry: DivPose): DivPose {
    return { dx: 0, dy: 0, kx: 0, ky: 0, spin: 0.04, swirl: 0.4,
             layout: carry.layout, nx: carry.nx, ny: carry.ny, seg: carry.seg, bloomType: 0, radSeg: 6, seam: 1 };
  }

  /** Divide-and-move journey. Eases a DivPose between CENTER (all spreads 0, one
   *  point) and random BLOOMs — points DIVIDE and slide apart on the way out,
   *  MERGE inward on the way back. No fade, no crossfade: bisect/trisect are
   *  identical at d=0 so the per-axis mode can switch invisibly at center. */
  private updateDivide(dt: number) {
    let e: number;
    if (this.audioEl && this.flow && !this.audioEl.paused && !this.audioEl.ended) {
      const s = this.flow.sample(this.audioEl.currentTime);
      e = Math.min(1, s.loudness * 1.2 + s.bass * 0.2);
    } else {
      e = 0.5 + 0.5 * Math.sin(this.clock.elapsedTime * 0.06);
    }
    this.autoEnergy += (e - this.autoEnergy) * (1 - Math.exp(-dt * 0.5));

    const pace = 0.5 + this.autoEnergy * 0.8;
    if (this.divT < 1) {
      this.divT = Math.min(1, this.divT + (dt / this.divMoveDur) * pace);
    } else {
      this.divHold -= dt;
      if (this.divHold <= 0) {
        this.divFrom = { ...this.divCur };
        if (this.divAtBloom) {
          if (this.divIsV2) {
            this.divTo = this.splitCenterPose(this.divCur);
          } else if (this.divIsRadial) {
            // auto · radial: HOME = pure clean radial base (seam 0, bloom hidden).
            // Re-roll the base radial fold count NOW (during the bloom, seam high,
            // base hidden) so the home seg change is invisible; carry otherwise.
            if (Math.random() < 0.5) this.divBaseSeg = [4, 5, 6, 6, 8][Math.floor(Math.random() * 5)];
            this.divTo = { dx: 0, dy: 0, kx: 0, ky: 0, spin: 0.02, swirl: 0.4, layout: 0, nx: 0, ny: 1, seg: 1, bloomType: 2, radSeg: this.divCur.radSeg, seam: 0 };
          } else {
            const home = this.radialHome();
            if (Math.random() < 0.75 && this.divHomeSeg > 0) home.radSeg = this.divHomeSeg;
            this.divHomeSeg = home.radSeg;
            if (Math.random() < 0.11) home.seam = 0.0; // rare bare-original rest
            this.divTo = home;
          }
          this.divMoveDur = 3.5 + Math.random() * 2.5;
          this.divHold = 3 + Math.random() * 4;
          this.divAtBloom = false;
        } else {
          // at home: pick the next bloom (its structure takes effect here, where
          // seam is low and any structure looks alike → invisible swap)
          this.divTo = this.divIsV2 ? this.randomSplitBloom()
            : this.divIsRadial ? this.randomRadialBloom()
            : this.randomDivBloom();
          if (Math.random() < 0.4) this.divWorldDir *= -1;
          this.divMoveDur = 4 + Math.random() * 3;
          this.divHold = 4 + Math.random() * 5;
          this.divAtBloom = true;
        }
        // worldRot leg: decide the destination NOW and ease straight to it, so it
        // lands exactly on a L/R-symmetric angle (radial) — no drift-then-snap.
        // turn ~ a third to a full turn so it visibly rotates while travelling.
        this.divWorldFrom = this.divWorld;
        this.divWorldTo = this.divIsRadial
          ? this.divWorldTarget(this.divTo, 1.0 + Math.random() * 2.6)
          : this.divWorld; // grid/triangle non-radial modes: handled elsewhere
        this.divT = 0;
      }
    }

    const t = this.divT, m = t * t * (3 - 2 * t);
    const L = (x: number, y: number) => x + (y - x) * m;
    const f = this.divFrom, g = this.divTo, cur = this.divCur;
    cur.dx = L(f.dx, g.dx); cur.dy = L(f.dy, g.dy);
    cur.spin = L(f.spin, g.spin); cur.swirl = L(f.swirl, g.swirl);
    if (this.divIsV2) {
      // v2 (split/symCells): snap structure to target (consolidates via dx→0)
      cur.kx = g.kx; cur.ky = g.ky;
      cur.layout = g.layout; cur.nx = g.nx; cur.ny = g.ny; cur.seg = g.seg;
    } else {
      // v1: SEAM dips through 0 at the MIDPOINT of every transition — first half
      // collapses the current fold (fromSeam→0), second half emerges the target
      // (0→toSeam). Structure swaps exactly at the seam=0 crossing (t=0.5), where
      // any fold looks like the seamless original → invisible, no jagged jump.
      if (t < 0.5) {
        const h = t / 0.5, hm = h * h * (3 - 2 * h);
        cur.seam = f.seam * (1 - hm);
      } else {
        const h = (t - 0.5) / 0.5, hm = h * h * (3 - 2 * h);
        cur.seam = g.seam * hm;
        cur.kx = g.kx; cur.ky = g.ky;
        cur.nx = g.nx; cur.seg = g.seg;
        cur.bloomType = g.bloomType; cur.radSeg = g.radSeg;
      }
    }
    this.divSpin += dt * cur.spin * TAU;        // ONE shared spin phase (coherent)
    // WHOLE-SCREEN ROTATION. In radial mode worldRot is an EASED POSE PARAM like
    // everything else: each leg sets divWorldFrom→divWorldTo (target chosen up
    // front to land on a L/R-symmetric angle) and we ease straight there with the
    // SAME m curve — one smooth motion that turns AND arrives square, no separate
    // drift+snap. Non-radial divide/split keep a continuous free spin.
    if (this.divIsRadial) {
      this.divWorld = L(this.divWorldFrom, this.divWorldTo);
    } else {
      const transFlux = (this.divT > 0 && this.divT < 1) ? Math.sin(this.divT * Math.PI) : 0;
      this.divWorld += dt * (0.05 + transFlux * 0.6 + this.autoEnergy * 0.06) * this.divWorldDir;
    }
    // gentle continuous orbit for the "centers circling each other" bloom (bt1)
    this.divOrbit += dt * (0.10 + this.autoEnergy * 0.08) * this.divWorldDir;

    const u = this.uAVals;
    u[0] = 0; // not a symMorph kind; the uDivide branch reads slots directly
    if (this.divIsV2) {
      // symCells slots: layout=uA[2], nx=uA[5], ny=uA[6], dx=uA[10], dy=uA[12],
      // ringRot=uA[11], spinPhase=uA[4], swirl=uA[7], seg=uA[1]
      u[2] = cur.layout; u[5] = cur.nx; u[6] = cur.ny;
      u[10] = cur.dx; u[12] = cur.dy; u[11] = this.divSpin * 0.4; // ring orbits slower
      u[4] = this.divSpin; u[7] = cur.swirl; u[1] = cur.seg;
    } else {
      // v1 slots: dx=10, dy=12, kx=5, ky=6, worldRot=11, spinPhase=4, swirl=7,
      // innerSeg/sharp=1, count=3 (triangle pts OR radial segs by bloomType),
      // seam=2, bloomType=8 (0 grid, 1 triangle, 2 radial)
      u[10] = cur.dx; u[12] = cur.dy; u[5] = cur.kx; u[6] = cur.ky;
      // radial/grid rotation = worldRot ONLY (transition turn + gentle idle +
      // axis-dwell) — no constant spin. BUT the orbiting-centers bloom (bt1)
      // adds a continuous gentle orbit so the 3 radial centers circle each other.
      u[11] = (cur.bloomType > 0.5 && cur.bloomType < 1.5)
        ? this.divWorld + this.divOrbit
        : this.divWorld;
      u[4] = this.divSpin; u[7] = cur.swirl;
      u[1] = cur.seg; u[2] = cur.seam; u[8] = cur.bloomType;
      // uA[3] = count: radial segs (radial), triangle points (triangle), else 0
      u[3] = cur.bloomType > 1.5 ? cur.radSeg : (cur.bloomType > 0.5 ? cur.nx : 0);
      u[9] = this.divBaseSeg; // radial-base fold count (auto · radial)
    }
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

    if (this.autoMode === "symmetry" || this.autoMode === "pinwheel") {
      this.updateSymJourney(dt);
      return;
    }

    if (this.autoMode === "divide" || this.autoMode === "dividev2" || this.autoMode === "radialdiv") {
      this.updateDivide(dt);
      return;
    }
  }

  /** Set how fast the flow field's vector directions change over time. */
  private setFieldSpeed(speed: number) {
    // clamp: 0 = frozen field, up to 0.99 = extreme churn of the underlying flow
    this.fieldSpeed = speed <= 0 ? 0 : Math.min(0.99, Math.max(0.004, speed));
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
    this.pointsMat.uniforms.uColorSpread.value = 0.4;
    this.uAVals[14] = this.modeA.points; // show all centers when idle
    if (this.modeB) this.uBVals[14] = this.modeB.points;
  }

  /** Map sampled music signals onto the engine. LIGHT + COLOR ONLY — the music
   *  must NOT change particle motion/speed/field, because faster motion against
   *  the fixed trail decay reads as varying blur. Motion stays constant; the
   *  beat lives entirely in brightness, contrast, exposure, and color. */
  private applyAudio(dt: number) {
    if (!this.audioEl || !this.flow) return;
    const s = this.flow.sample(this.audioEl.currentTime);
    // brightness pulses with the beat + overall loudness
    this.pointsMat.uniforms.uIntensity.value =
      this.intensity * (0.5 + s.loudness * 0.7 + s.pulse * 1.1);
    // motion uniforms pinned to base every frame so blur is constant w/ music
    this.pointsMat.uniforms.uSizeMax.value = this.breathePeak;
    this.velocityVar.material.uniforms.uCurlStrength.value = 1.0;
    this.positionVar.material.uniforms.uSpeed.value = 1.0;
    this.velocityVar.material.uniforms.uFieldSpeed.value = this.fieldSpeed;
    // color is where the music lives: beat + treble push the hue phase, and
    // per-particle color spread widens with energy so the palette visibly
    // shifts/shimmers on the beat.
    this.pointsMat.uniforms.uColorPhase.value += (s.treble * 0.10 + s.pulse * 0.12 + s.loudness * 0.04) * dt;
    const tgtSpread = 0.4 + s.loudness * 0.5 + s.pulse * 0.3;
    const sp = this.pointsMat.uniforms.uColorSpread;
    sp.value += (tgtSpread - sp.value) * (1 - Math.exp(-dt * 6));
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
    // default to "evolving symmetry" (the headline mode).
    this.setKaleido(KALEIDO_MODES.findIndex((m) => m.name === "evolving symmetry"));
    // push current state so the HUD reflects it immediately
    this.emitKaleidoLabel();
    this.onPaletteChange(PALETTES[this.paletteIndex].name);
    this.onFieldSpeedChange(this.fieldSpeed);
    this.onBreatheChange(this.breatheRate, this.breathePeak);
    this.onSeamChange(this.seamSoft, this.trailBlur);
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
