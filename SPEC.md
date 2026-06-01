# Music Visualizer — Specification

## 1. Concept

A standalone web-based music visualizer that **pre-analyzes an entire audio file
offline**, builds a timeline ("flow map") from the music's structure, then plays
back a GPU-rendered visual in real time, synced to that precomputed timeline.

The visual core is a **GPU particle system advected through a curl-noise flow
field**, rendered with **feedback/trail buffers** for ink-in-water / smoke /
aurora textures, then composited through **kaleidoscopic radial symmetry** and a
**palette** post pass. A **volumetric cloud (Simplex fBm) backdrop** sits behind
the particles and also shapes the flow field. Motion is driven primarily by
**BPM/beat timing**, modulated by loudness, spectral energy, song sections, and
key/mood.

This is a **fresh start**, not an evolution of the existing SpinFreak ride
visualizers. The explicit goals of the new direction are a distinct **look**
(organic, painterly, alive — not retro tunnel/grid) and a fundamentally better
**motion feel** (fluid advection that breathes with the music, not canned/
mechanical movement). Those two priorities drive the technical choices below.

## 2. Goals & Non-Goals

**Goals**
- Read a full track up front; analyze it; derive a structured, time-indexed flow.
- Real-time WebGL playback synced to the audio, tunable live.
- A distinct, organic, painterly look — a clean break from the prior visualizers.
- Fluid, *alive* motion via GPU-particle advection through curl-noise flow fields.
- Configurable kaleidoscope: mirror count, symmetry type, center, rotation.
- User settings for **palette**, **mode**, and **music-interpretation** params.

**Non-Goals (v1)**
- Real-time microphone / live-input reactivity (we deliberately precompute).
- Offline render-to-video export (possible later; engine should not preclude it).
- Mobile-first; target is desktop browser with a capable GPU.

## 3. Architecture

Two phases, cleanly separated by a serializable **FlowMap** artifact.

```
 ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
 │ Audio file  │ ──▶ │ Analysis (offline│ ──▶ │  FlowMap     │
 │ (mp3/wav/…) │     │  in-browser, WASM)│     │  (JSON)      │
 └─────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                              ┌───────────────────────▼───────────────────────┐
                              │ Playback engine                                │
                              │  AudioElement (timekeeper)                     │
                              │   └─ Driver samples FlowMap @ currentTime       │
                              │        └─ Uniforms ──▶ WebGL fragment shader     │
                              └─────────────────────────────────────────────────┘
```

### 3.1 Stack
- **React 19 + Vite + TypeScript** (UI shell + settings; no shared code with SpinFreak).
- **Three.js / @react-three/fiber** for render targets, the particle draw, and post passes.
- **GLSL** throughout: a GPGPU particle simulation (ping-pong FBOs) + draw/post shaders.
- **essentia.js** (WASM) as the analysis workhorse; **Web Audio API** for decoding
  and any supplemental offline `AnalyserNode` passes.

> Implementation note: the particle sim is a classic GPGPU ping-pong — position
> and velocity stored in float textures, advanced each frame by a sim fragment
> shader sampling the curl-noise field, then drawn as `gl.POINTS`. This is
> standard WebGL2 territory (`@react-three/fiber` + custom `WebGLRenderTarget`s).

### 3.2 Why precompute
The timeline is known ahead of time, so motion can be *choreographed* against
beats and section boundaries (lead-ins, anticipation, drops) instead of merely
reacting frame-to-frame. The audio element is the single clock; the visual reads
from the FlowMap at `audio.currentTime`, so A/V stay locked even under frame drops.

## 4. Analysis Phase

Runs once when a track is loaded. Produces the FlowMap. All four signal groups:

| Signal            | What we extract                                              | Drives |
|-------------------|-------------------------------------------------------------|--------|
| **BPM + beats**   | Global tempo, per-beat timestamps, downbeats if available   | Pulse, rotation cadence, warp speed |
| **Energy/loudness** | RMS/loudness envelope + 3-band energy (bass/mid/treble) over time | Brightness, displacement amplitude, flow speed |
| **Sections**      | Structural boundaries (intro/verse/chorus/drop) via self-similarity novelty | Mode/palette transitions, "events" |
| **Key / mood**    | Musical key + estimated valence/energy                      | Default palette suggestion |

**Output cadence:** continuous signals (loudness, band energy) sampled to a fixed
grid (e.g. ~30–60 Hz, or analysis-hop-aligned) and stored as typed arrays; discrete
events (beats, downbeats, section starts) stored as timestamp lists.

### 4.1 FlowMap schema (draft)
```ts
interface FlowMap {
  version: 1;
  durationSec: number;
  sampleRateHz: number;          // grid rate for continuous tracks
  bpm: number;
  beats: number[];               // seconds
  downbeats?: number[];          // seconds
  sections: { startSec: number; label?: string; }[];
  key?: { tonic: string; scale: "major" | "minor"; };
  mood?: { valence: number; energy: number; }; // 0..1
  tracks: {
    loudness: Float32Array;      // overall, normalized 0..1
    bass: Float32Array;          // band energy 0..1
    mid: Float32Array;
    treble: Float32Array;
  };
}
```

## 5. Rendering Phase

A multi-pass GPU pipeline. The particle simulation and feedback trails create the
*motion feel*; the post passes create the *look*. All passes read shared uniforms
updated each frame from the FlowMap + a decaying beat-pulse impulse.

### 5.1 Pipeline (per frame)
```
1. Flow field      curl(noise) field, parameterized by FlowMap signals
                   (bass bends the field, beats kick its scale/rotation).
2. Particle sim    ping-pong FBOs: velocity ← flow field; position += velocity.
                   Respawn/emit policy reacts to beats & sections.
3. Particle draw   gl.POINTS into an HDR target, additive blend, palette-tinted.
4. Feedback/trails blend prev frame * decay + new draw → trail target.
                   Decay & advection of the trail buffer = the "ink/smoke" look.
5. Cloud backdrop  domain-warped Simplex fBm rendered behind particles.
6. Kaleidoscope    radial symmetry applied to the composited image (see §6.1).
7. Palette + post  palette mapping, bloom/glow, vignette, grain.
```

### 5.2 Core building blocks
1. **Vector flow field (curl noise)** — divergence-free curl-of-noise field; the
   engine's motion source. Field scale, rotation, and warp are music-driven.
2. **GPU particles** — large count (target ~250k–1M depending on GPU), advected by
   the field. Emission, lifetime, and speed react to beats/energy.
3. **Feedback trails** — ping-pong trail buffer with per-frame decay; the dominant
   contributor to the painterly, alive texture.
4. **Clouds (domain-warped Simplex fBm)** — backdrop *and* a modulation input to
   the flow field, so the particles trace cloud structure.
5. **Kaleidoscope / radial symmetry** — post pass over the composite (see §6.1).

### 5.3 Modes (v1)
Modes are presets over the same pipeline (emission style, trail decay, field
character, blend) — not separate shaders:
- **Flow** — long-lived particles, low trail decay → flowing ribbons/streamlines.
- **Clouds** — dense slow particles + strong cloud backdrop → soft volumetric drift.
- **Sharp** — short trails, ridged/high-frequency field, hard symmetry seams →
  crisp, faceted, high-contrast.
- *(extensible: new modes are just new parameter presets.)*

Mode is a user setting; transitions can also be auto-triggered at section
boundaries (configurable).

## 6. Settings

Three setting groups, persisted (localStorage), live-adjustable during playback.

- `segments` — number of mirrored wedges (e.g. 1–24; 1 = off).
- `symmetry` — symmetry family (see below).
- `centerX`, `centerY` — symmetry origin (default screen center; offset for
  "off-axis" looks).
- `rotationSpeed` — base spin, scaled by BPM.
- `twist` — radial rotation gradient (spiral effect).

**Symmetry families (planned):**
- `radial` — evenly-spaced mirrored/rotated wedges around a center (the classic).
- `grid` — tiled/repeated reflection across X/Y (wallpaper-group style).
- `polar` — repetition in polar space (angle *and* radius bands).
- `multi-point` — **multiple symmetry origins**, each its own mirror center.
- `moving mirrors` — symmetry centers that **drift/animate** over time (music-driven),
  so the mirror layout itself is in motion.

These are a clear progression: start with `radial`, then `grid`/`polar`, then the
multi-point and moving-mirror variants (the most distinctive, do last).

### 6.2 Palette
- Named palettes (curated gradients/ramps), e.g. "Aurora", "Ember", "Mono".
- `auto` option: pick palette from detected key/mood.
- Mapping of palette ramp to signals (e.g. loudness → ramp position).

### 6.3 Music interpretation
- `beatSensitivity` — strength/decay of the per-beat pulse.
- `motionDriver` — which signal leads motion (loudness | bass | mid | treble | beats).
- `intensity` — global amplitude of displacement/brightness response.
- `sectionReactivity` — how strongly section changes trigger mode/palette shifts.
- `smoothing` — temporal smoothing of continuous signals.

## 7. UI (v1)
- Drop-zone / file picker to load a track.
- Analysis progress indicator (first-load only, cached by file hash).
- Transport (play/pause/seek) on the audio element.
- Settings panel for the three groups in §6.
- Fullscreen toggle.

## 8. Aesthetic reference
**Refik Anadol** (e.g. *Unsupervised*, MoMA — https://www.moma.org/magazine/articles/821).
Flowing "data pigment," machine-hallucination color fields, soft morphing fluid
masses. This is the north-star look: organic, painterly, alive — the particle-flow +
feedback-trail engine is chosen precisely to get into this territory. Intended uses
the engine should suit: an ambient **screensaver** and a **concert-video** backdrop.

## 9. Open Questions / Decided
- **Section detection** — user likes it; keep. Start with a simple energy/novelty
  heuristic, refine toward essentia.js segmentation.
- **Caching** — confirmed wanted: cache analysis by file hash (IndexedDB).
- **Kaleidoscope breadth** — confirmed direction: radial + grid + polar + multi-point
  + moving-mirror families (see §6.1). Build radial first.
- **Video export — DEFERRED.** Not now. Goal right now is just to see it work; other
  uses (screensaver, concert visuals) may come later but don't gate the design.

## 10. Milestones (suggested)
1. **Scaffold** — Vite/React/Three project, render-target plumbing, audio load+play.
2. **Particle core** — GPGPU ping-pong sim + curl-noise flow field, drawn as points.
   *(This is the make-or-break "does the motion feel alive" milestone — do it early.)*
3. **Feedback trails** — trail buffer + decay; tune for the painterly look.
4. **Analysis** — essentia.js integration → FlowMap (BPM/beats first, then bands).
5. **Driver** — sample FlowMap at `currentTime`; music drives field + emission + pulse.
6. **Kaleidoscope** — radial symmetry post pass + settings.
7. **Modes + palettes** — Flow/Clouds/Sharp presets, palette mapping, post (bloom/grain).
8. **Sections/key/mood** — auto transitions + auto palette; interpretation settings panel.
9. **Polish** — analysis caching (IndexedDB), performance/particle-count tuning, presets.
