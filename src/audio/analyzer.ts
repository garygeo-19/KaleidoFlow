import { guess } from "web-audio-beat-detector";
import type { FlowMap } from "./flowmap";

// Continuous-signal grid rate. 60 Hz ≈ one sample per render frame.
const SAMPLE_RATE_HZ = 60;

export type ProgressFn = (stage: string) => void;

/**
 * Offline analysis of a whole track → FlowMap.
 *  - tempo/beats via web-audio-beat-detector (great on steady 4/4 like techno)
 *  - loudness from raw RMS, bass/mid/treble via biquad-filtered offline renders
 */
export async function analyzeTrack(url: string, onProgress?: ProgressFn): Promise<FlowMap> {
  onProgress?.("fetching");
  const resp = await fetch(url);
  const arrayBuffer = await resp.arrayBuffer();

  onProgress?.("decoding");
  const AC: typeof AudioContext =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AC();
  const buffer = await ctx.decodeAudioData(arrayBuffer);
  await ctx.close();

  onProgress?.("detecting tempo");
  let bpm = 128;
  let offset = 0;
  try {
    const g = await guess(buffer);
    bpm = g.bpm;
    offset = g.offset;
  } catch (e) {
    console.warn("[analyzer] tempo detection failed, defaulting to 128 BPM", e);
  }

  onProgress?.("analyzing energy");
  const loudness = computeRms(buffer, SAMPLE_RATE_HZ);
  const bass = await computeBandRms(buffer, SAMPLE_RATE_HZ, "lowpass", 120, 0.7);
  const mid = await computeBandRms(buffer, SAMPLE_RATE_HZ, "bandpass", 1200, 0.8);
  const treble = await computeBandRms(buffer, SAMPLE_RATE_HZ, "highpass", 5000, 0.7);

  for (const env of [loudness, bass, mid, treble]) {
    normalize(env);
    smooth(env, 0.4);
  }

  const durationSec = buffer.duration;
  const beats: number[] = [];
  if (bpm > 0) {
    const beatDur = 60 / bpm;
    for (let t = offset; t < durationSec; t += beatDur) beats.push(t);
  }

  onProgress?.("done");
  return {
    version: 1,
    durationSec,
    sampleRateHz: SAMPLE_RATE_HZ,
    bpm,
    beatOffsetSec: offset,
    beats,
    loudness,
    bass,
    mid,
    treble,
  };
}

/** Per-hop RMS of a (down-mixed) buffer. */
function computeRms(buffer: AudioBuffer, hz: number): Float32Array {
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
  const n = ch0.length;
  const hop = Math.max(1, Math.floor(buffer.sampleRate / hz));
  const frames = Math.ceil(n / hop);
  const out = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    const end = Math.min(start + hop, n);
    let sum = 0;
    for (let i = start; i < end; i++) {
      const s = ch1 ? (ch0[i] + ch1[i]) * 0.5 : ch0[i];
      sum += s * s;
    }
    out[f] = Math.sqrt(sum / Math.max(1, end - start));
  }
  return out;
}

/** Render the buffer through a biquad filter offline, then take per-hop RMS. */
async function computeBandRms(
  buffer: AudioBuffer,
  hz: number,
  type: BiquadFilterType,
  freq: number,
  q: number,
): Promise<Float32Array> {
  const length = Math.ceil(buffer.duration * buffer.sampleRate);
  const offline = new OfflineAudioContext(1, length, buffer.sampleRate);
  const src = offline.createBufferSource();
  src.buffer = buffer;
  const filter = offline.createBiquadFilter();
  filter.type = type;
  filter.frequency.value = freq;
  filter.Q.value = q;
  src.connect(filter).connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return computeRms(rendered, hz);
}

/** Scale to 0..1 against a robust (99th-percentile) max so transients don't crush the signal. */
function normalize(a: Float32Array) {
  if (a.length === 0) return;
  const sorted = Float32Array.from(a).sort();
  const p = sorted[Math.floor(sorted.length * 0.99)] || 1e-6;
  const inv = 1 / Math.max(p, 1e-6);
  for (let i = 0; i < a.length; i++) a[i] = Math.min(1, a[i] * inv);
}

/** Light causal low-pass to tame frame-to-frame jitter. */
function smooth(a: Float32Array, alpha: number) {
  for (let i = 1; i < a.length; i++) a[i] = a[i] * alpha + a[i - 1] * (1 - alpha);
}
