import { useEffect, useRef, useState } from "react";
import { Visualizer } from "./visualizer/Visualizer";
import { analyzeTrack } from "./audio/analyzer";
import { FlowPlayer } from "./audio/flowmap";

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const vizRef = useRef<Visualizer | null>(null);

  const [kaleido, setKaleido] = useState("off");
  const [palette, setPalette] = useState("");
  const [evolve, setEvolve] = useState(0);
  const [breathe, setBreathe] = useState({ rate: 0, peak: 0 });
  const [mode, setMode] = useState("");
  const [seam, setSeam] = useState({ soft: 0, blur: 0 });

  const [status, setStatus] = useState("idle");
  const [bpm, setBpm] = useState(0);
  const [track, setTrack] = useState("");
  const [playing, setPlaying] = useState(false);

  const modeNames = Visualizer.modeNames();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const viz = new Visualizer(canvas);
    vizRef.current = viz;
    viz.onKaleidoChange = setKaleido;
    viz.onPaletteChange = setPalette;
    viz.onFieldSpeedChange = setEvolve;
    viz.onBreatheChange = (rate, peak) => setBreathe({ rate, peak });
    viz.onModeChange = setMode;
    viz.onSeamChange = (soft, blur) => setSeam({ soft, blur });
    viz.start();
    return () => {
      viz.dispose();
      vizRef.current = null;
    };
  }, []);

  async function loadAndPlay(url: string, name: string) {
    const audio = audioRef.current;
    const viz = vizRef.current;
    if (!audio || !viz) return;
    setTrack(name);
    // Start playback NOW, synchronously inside the click gesture. The browser's
    // autoplay policy rejects play() if it's called after the long analyze
    // await (the user-gesture activation expires) — so kick it off first and
    // attach the analysis when it's ready.
    audio.src = url;
    audio.currentTime = 0;
    const playPromise = audio.play().catch(() => {});
    setStatus("analyzing");
    try {
      const flow = await analyzeTrack(url, (stage) => setStatus(stage));
      setBpm(Math.round(flow.bpm));
      viz.attachAudio(audio, new FlowPlayer(flow));
      await playPromise;
      if (audio.paused) await audio.play().catch(() => {});
      setStatus(audio.paused ? "ready (press play)" : "playing");
    } catch (e) {
      console.error(e);
      setStatus("ready (press play)");
    }
  }

  function togglePlay() {
    const audio = audioRef.current;
    if (!audio || !audio.src) return;
    if (audio.paused) audio.play();
    else audio.pause();
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) loadAndPlay(URL.createObjectURL(file), file.name);
  }

  function pickMode(name: string) {
    vizRef.current?.selectMode(name);
  }

  return (
    <>
      <canvas ref={canvasRef} />
      <audio
        ref={audioRef}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />

      <div className="panel">
        <button onClick={() => loadAndPlay("/techno-01.mp3", "techno-01")}>
          ▶ Techno sample
        </button>
        <button onClick={togglePlay} disabled={!track}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <label className="file">
          Load file…
          <input type="file" accept="audio/*" onChange={onFile} hidden />
        </label>
      </div>

      <div className="modemenu">
        <div className="modegroup">
          <div className="modegroup-label">mode</div>
          {modeNames.map((name) => (
            <button
              key={name}
              className={"modebtn" + (name === mode ? " active" : "")}
              onClick={() => pickMode(name)}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <div className="hud">
        {track ? (
          <>
            <b>{track}</b> · {bpm ? `${bpm} BPM` : "—"} · {status} &nbsp;|&nbsp;{" "}
          </>
        ) : (
          <>music-visualizer · no track loaded &nbsp;|&nbsp; </>
        )}
        <b>K</b> kaleido: {kaleido} &nbsp;|&nbsp;{" "}
        <b>P</b> {palette} &nbsp;|&nbsp;{" "}
        <b>[ ]</b> evolve: {evolve === 0 ? "frozen" : evolve.toFixed(3)} &nbsp;|&nbsp;{" "}
        <b>, .</b> breathe: {breathe.rate.toFixed(2)} &nbsp;|&nbsp;{" "}
        <b>- =</b> size: {breathe.peak.toFixed(0)}px
        <br />
        <b>s S</b> seam-soft: {seam.soft.toFixed(2)} &nbsp;|&nbsp;{" "}
        <b>b B</b> trail-blur: {seam.blur.toFixed(1)}
      </div>
    </>
  );
}
