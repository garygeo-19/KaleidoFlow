// The precomputed, time-indexed analysis of a track. Continuous signals are
// sampled to a fixed grid (sampleRateHz); beats are a list of timestamps.
export interface FlowMap {
  version: 1;
  durationSec: number;
  sampleRateHz: number;
  bpm: number;
  beatOffsetSec: number;
  beats: number[]; // seconds
  loudness: Float32Array; // overall RMS, normalized 0..1
  bass: Float32Array; // band RMS, normalized 0..1
  mid: Float32Array;
  treble: Float32Array;
}

// What the visualizer consumes each frame.
export interface MusicSignals {
  loudness: number;
  bass: number;
  mid: number;
  treble: number;
  pulse: number; // beat impulse, spikes to 1 on a beat then decays
}

/**
 * Wraps a FlowMap and samples it at an arbitrary playback time. Keeps a small
 * cursor so the per-beat impulse is cheap to evaluate each frame.
 */
export class FlowPlayer {
  private beatCursor = 0;

  constructor(public readonly flow: FlowMap) {}

  reset() {
    this.beatCursor = 0;
  }

  private sampleEnv(arr: Float32Array, t: number): number {
    if (arr.length === 0) return 0;
    const x = t * this.flow.sampleRateHz;
    const i = Math.floor(x);
    if (i < 0) return arr[0];
    if (i >= arr.length - 1) return arr[arr.length - 1];
    const f = x - i;
    return arr[i] * (1 - f) + arr[i + 1] * f;
  }

  /** Sample all signals at time `t` (seconds). `pulseDecay` in 1/sec. */
  sample(t: number, pulseDecay = 7): MusicSignals {
    const beats = this.flow.beats;
    // handle seeking backwards
    if (this.beatCursor > 0 && beats[this.beatCursor - 1] > t) this.beatCursor = 0;
    while (this.beatCursor < beats.length && beats[this.beatCursor] <= t) this.beatCursor++;
    const lastBeat = this.beatCursor > 0 ? beats[this.beatCursor - 1] : -1;
    const pulse = lastBeat >= 0 ? Math.exp(-(t - lastBeat) * pulseDecay) : 0;

    return {
      loudness: this.sampleEnv(this.flow.loudness, t),
      bass: this.sampleEnv(this.flow.bass, t),
      mid: this.sampleEnv(this.flow.mid, t),
      treble: this.sampleEnv(this.flow.treble, t),
      pulse,
    };
  }
}
