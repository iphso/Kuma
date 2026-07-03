/**
 * <kuma-player> — WebGPU model player as a self-contained custom element.
 *
 * Attributes:
 *   src         path to a .iph package; triggers load when changed
 *   debug       (boolean) show debug panel on mount
 *   vanilla     (boolean) hide debug/status UI for a plain video-player surface
 *   autoplay    (boolean) start playing immediately after load
 *   low-memory  (boolean) memory-constrained posture: pool depth 1, one frame in flight
 *   pool-depth  (number) explicit buffer-pool ring depth override
 *
 * CSS custom properties (set on the element or a parent):
 *   --kp-accent    progress bar / accent colour  (default: #ffffff)
 *   --kp-radius    card border radius            (default: 12px)
 *
 * Usage in any HTML/Astro/etc:
 *   import 'kuma-bart';  // registers <kuma-player>
 *   <kuma-player src="/artifacts/model.iph"></kuma-player>
 */

import { KumaModel, BUFFER_POOL_DEPTH } from "../index.js";
import type { VerifyReport, ProfileReport } from "../index.js";
import { OnnxModel } from "../onnx/model.js";
import { peekIphFormat } from "../iph/loader.js";
import { GpuFrameRenderer } from "./gpuRender.js";

// ── Icons ──────────────────────────────────────────────────────────────────────

const ICON_PLAY = `<svg xmlns="http://www.w3.org/2000/svg" width="10" height="12" viewBox="0 0 10 12" fill="currentColor" aria-hidden="true"><path d="M0 0v12l10-6z"/></svg>`;
const ICON_PAUSE = `<svg xmlns="http://www.w3.org/2000/svg" width="11" height="12" viewBox="0 0 11 12" fill="currentColor" aria-hidden="true"><rect width="4" height="12" rx="1"/><rect x="7" width="4" height="12" rx="1"/></svg>`;
const ICON_GEAR = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/><path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.469l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/></svg>`;

// ── Styles ─────────────────────────────────────────────────────────────────────

const STYLES = `
:host {
  display: block;
  --kp-accent: #ffffff;
  --kp-radius: 12px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; }

.kp {
  background: #111113;
  border-radius: var(--kp-radius);
  overflow: hidden;
}

/* ── Viewport ─────────────────────────────────────────────────────────────── */
.kp-viewport {
  position: relative;
  background: repeating-conic-gradient(#1d1f24 0% 25%, #16181c 0% 50%) 50% / 20px 20px;
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
}

canvas {
  display: block;
  max-width: 100%;
  height: auto;
}

.kp-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: rgba(255,255,255,0.2);
  letter-spacing: 0.05em;
  pointer-events: none;
}

.kp-status {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 20px 12px 7px;
  font-size: 10.5px;
  font-family: ui-monospace, 'Cascadia Code', 'SF Mono', monospace;
  color: rgba(255,255,255,0.35);
  background: linear-gradient(to top, rgba(0,0,0,0.5), transparent);
  pointer-events: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Control bar ──────────────────────────────────────────────────────────── */
.kp-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 10px;
  height: 46px;
  background: #0d0d10;
  border-top: 1px solid rgba(255,255,255,0.045);
}

.kp-play {
  flex-shrink: 0;
  width: 30px;
  height: 30px;
  border: none;
  background: rgba(255,255,255,0.08);
  border-radius: 50%;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 120ms ease;
  padding: 0;
}
.kp-play:hover:not(:disabled) { background: rgba(255,255,255,0.15); }
.kp-play:active:not(:disabled) { background: rgba(255,255,255,0.22); }
.kp-play:disabled { opacity: 0.25; cursor: default; }

/* ── Progress bar ─────────────────────────────────────────────────────────── */
.kp-progress {
  flex: 1;
  position: relative;
  height: 20px;
  display: flex;
  align-items: center;
  cursor: pointer;
}

.kp-progress::before {
  content: '';
  position: absolute;
  left: 0; right: 0;
  height: 3px;
  background: rgba(255,255,255,0.1);
  border-radius: 99px;
  transition: height 150ms ease;
  pointer-events: none;
}

.kp-progress:hover::before { height: 5px; }

.kp-progress-fill {
  position: absolute;
  top: 50%;
  left: 0;
  transform: translateY(-50%);
  height: 3px;
  background: var(--kp-accent);
  border-radius: 99px;
  width: 0%;
  pointer-events: none;
  transition: height 150ms ease;
}

.kp-progress:hover .kp-progress-fill { height: 5px; }

.kp-scrubber {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: pointer;
  z-index: 1;
}

.kp-time {
  flex-shrink: 0;
  min-width: 3.2ch;
  text-align: right;
  font-family: ui-monospace, 'Cascadia Code', 'SF Mono', monospace;
  font-size: 11.5px;
  font-variant-numeric: tabular-nums;
  color: rgba(255,255,255,0.38);
  white-space: nowrap;
}

.kp-debug-btn {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border: none;
  background: transparent;
  color: rgba(255,255,255,0.28);
  cursor: pointer;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 120ms ease, background 120ms ease;
  padding: 0;
}
.kp-debug-btn:hover {
  color: rgba(255,255,255,0.6);
  background: rgba(255,255,255,0.07);
}
.kp-debug-btn.active {
  color: rgba(255,255,255,0.85);
  background: rgba(255,255,255,0.1);
}

/* ── Debug panel ──────────────────────────────────────────────────────────── */
.kp-debug {
  border-top: 1px solid rgba(255,255,255,0.055);
  padding: 12px 14px;
  background: #090909;
}
.kp-debug.hidden { display: none; }

:host([vanilla]) .kp-status,
:host([vanilla]) .kp-debug-btn,
:host([vanilla]) .kp-debug {
  display: none;
}

.kp-debug-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.kp-debug-row button {
  font: inherit;
  font-size: 12px;
  padding: 5px 12px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.6);
  border-radius: 6px;
  cursor: pointer;
  transition: background 120ms, color 120ms;
}
.kp-debug-row button:hover:not(:disabled) {
  background: rgba(255,255,255,0.09);
  color: rgba(255,255,255,0.9);
}
.kp-debug-row button:disabled { opacity: 0.3; cursor: default; }

details {
  border-top: 1px solid rgba(255,255,255,0.055);
  padding-top: 8px;
}
details summary {
  cursor: pointer;
  font-size: 11.5px;
  color: rgba(255,255,255,0.32);
  padding: 2px 0;
  list-style: none;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 5px;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: '▸';
  display: inline-block;
  font-size: 9px;
  transition: transform 150ms;
}
details[open] summary::before { transform: rotate(90deg); }
details summary:hover { color: rgba(255,255,255,0.6); }

.kp-log {
  margin-top: 8px;
  padding: 10px 12px;
  background: #050507;
  color: rgba(72, 230, 120, 0.8);
  font-family: ui-monospace, 'Cascadia Code', 'SF Mono', monospace;
  font-size: 11px;
  line-height: 1.65;
  border-radius: 6px;
  max-height: 260px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
  scrollbar-width: thin;
  scrollbar-color: rgba(255,255,255,0.1) transparent;
}
`;

// ── Template ───────────────────────────────────────────────────────────────────

const TEMPLATE = `
<div class="kp">
  <div class="kp-viewport">
    <canvas class="kp-canvas" style="display:none"></canvas>
    <div class="kp-placeholder">No model loaded</div>
    <div class="kp-status"></div>
  </div>
  <div class="kp-controls">
    <button class="kp-play" disabled aria-label="Play">${ICON_PLAY}</button>
    <div class="kp-progress" role="presentation">
      <div class="kp-progress-fill"></div>
      <input class="kp-scrubber" type="range" min="0" max="1" step="0.001" value="0"
             disabled aria-label="Scrub position"/>
    </div>
    <span class="kp-time">0.000</span>
    <button class="kp-debug-btn" aria-label="Toggle debug panel" title="Debug">${ICON_GEAR}</button>
  </div>
  <div class="kp-debug hidden">
    <div class="kp-debug-row">
      <button class="kp-verify" disabled>Verify</button>
      <button class="kp-profile" disabled>Profile</button>
    </div>
    <details>
      <summary>Log</summary>
      <pre class="kp-log">(idle)</pre>
    </details>
  </div>
</div>
`;

// ── Constants ──────────────────────────────────────────────────────────────────

// Cold-start fallback only: used for the window before a model has loaded. Once one has,
// the player reads model.maxInFlight (its real, possibly lowered pool depth) instead -- so
// a `low-memory` player serializes to a single in-flight frame. See `maxInFlight` below.
const DEFAULT_MAX_IN_FLIGHT = BUFFER_POOL_DEPTH;
const DEFAULT_PLAY_SWEEP_MS = 6000;
// Fallback pacing for models with no fps in their .iph metadata -- a reasonable
// scrubbing/preview cadence, not a target to hit when the real fps is known.
const DEFAULT_PLAY_FRAME_INTERVAL_MS = 1000 / 60;
const STALL_THRESHOLD_MS = 50;
const PROFILE_WARMUP_RUNS = 5;

// ── KumaPlayer ─────────────────────────────────────────────────────────────────

export class KumaPlayer extends HTMLElement {
  static readonly observedAttributes = ["src", "debug", "vanilla", "autoplay", "low-memory", "pool-depth"];

  private readonly shadow: ShadowRoot;

  // DOM refs (bound in bindRefs after shadow is populated)
  private canvas!: HTMLCanvasElement;
  private placeholderEl!: HTMLDivElement;
  private statusEl!: HTMLDivElement;
  private playBtn!: HTMLButtonElement;
  private scrubber!: HTMLInputElement;
  private progressFill!: HTMLDivElement;
  private timeEl!: HTMLSpanElement;
  private debugBtn!: HTMLButtonElement;
  private debugPanel!: HTMLDivElement;
  private verifyBtn!: HTMLButtonElement;
  private profileBtn!: HTMLButtonElement;
  private logEl!: HTMLPreElement;

  // Playback state
  private model: KumaModel | undefined;
  private onnxModel: OnnxModel | undefined;
  private gpuRenderer: GpuFrameRenderer | undefined;
  private timeInputName: string | undefined;
  private inFlight = 0;
  private pendingT: number | undefined;
  private playing = false;
  private loadPending = false;  // true when play was clicked before model loaded
  private loadInProgress = false;
  private hasConnectedBefore = false;
  // Bumped at the start of every load() -- lets a superseded in-flight load() or
  // submitFrame() callback (from a model the user has since switched away from) detect
  // it's stale and bail out instead of touching the newer model's state.
  private loadGeneration = 0;
  private playStartMs = 0;
  private capacityHits = 0;
  private requestedFrames = 0;
  private lastTickMs = -Infinity;
  private frameSequence = 0;
  private latestRenderedSequence = -1;
  private playSweepMs = DEFAULT_PLAY_SWEEP_MS;
  // How often playTick actually submits a new inference -- driven by the model's own
  // authored fps (from .iph playback metadata) so playback doesn't run inference more
  // often than the source video actually had distinct frames. A model can compute a
  // frame in a fraction of a millisecond, but sampling it at, say, 240Hz for a 24fps
  // source doesn't make the video smoother -- it just burns GPU on in-between times
  // nothing was ever encoded at, and reads as motion-smoothed/sped-up rather than
  // "faithful playback at real duration."
  private playFrameIntervalMs = DEFAULT_PLAY_FRAME_INTERVAL_MS;
  private lastPlayFrameMs = -Infinity;
  private debugVisible = false;

  constructor() {
    super();
    this.shadow = this.attachShadow({ mode: "open" });
    this.shadow.innerHTML = `<style>${STYLES}</style>${TEMPLATE}`;
    this.bindRefs();
    this.attachListeners();
  }

  private bindRefs(): void {
    const q = <T extends Element>(sel: string) => this.shadow.querySelector<T>(sel)!;
    this.canvas = q<HTMLCanvasElement>(".kp-canvas");
    this.placeholderEl = q<HTMLDivElement>(".kp-placeholder");
    this.statusEl = q<HTMLDivElement>(".kp-status");
    this.playBtn = q<HTMLButtonElement>(".kp-play");
    this.scrubber = q<HTMLInputElement>(".kp-scrubber");
    this.progressFill = q<HTMLDivElement>(".kp-progress-fill");
    this.timeEl = q<HTMLSpanElement>(".kp-time");
    this.debugBtn = q<HTMLButtonElement>(".kp-debug-btn");
    this.debugPanel = q<HTMLDivElement>(".kp-debug");
    this.verifyBtn = q<HTMLButtonElement>(".kp-verify");
    this.profileBtn = q<HTMLButtonElement>(".kp-profile");
    this.logEl = q<HTMLPreElement>(".kp-log");
  }

  private attachListeners(): void {
    this.playBtn.addEventListener("click", () => this.togglePlay());
    this.scrubber.addEventListener("input", () => {
      const t = Number(this.scrubber.value);
      this.updateT(t);
      this.submitFrame(t);
    });
    this.debugBtn.addEventListener("click", () => this.setDebugVisible(!this.debugVisible));
    this.verifyBtn.addEventListener("click", () => { void this.runVerify(); });
    this.profileBtn.addEventListener("click", () => { void this.runProfile(); });
  }

  connectedCallback(): void {
    if (this.hasAttribute("debug")) this.setDebugVisible(true);
    this.syncMode();
    const src = this.getAttribute("src");
    if (src) this.maybeLoad(src);
    this.hasConnectedBefore = true;
  }

  attributeChangedCallback(name: string, oldVal: string | null, newVal: string | null): void {
    if (!this.isConnected) return;
    if (name === "src" && newVal !== null && newVal !== oldVal) {
      if (this.hasConnectedBefore) {
        // A `src` change after the element has already connected once can only come from
        // explicit imperative code (e.g. a model picker) -- necessarily a deliberate
        // selection, not static markup -- so load immediately. The "wait for a play
        // click" lazy gate (see maybeLoad) applies only to a `src` already present when
        // the element first connects (custom-element upgrade always runs
        // attributeChangedCallback for pre-existing attributes *before*
        // connectedCallback, so hasConnectedBefore is still false at that point --
        // connectedCallback's own maybeLoad(src) call handles that case instead).
        void this.load(newVal).catch((err: unknown) => this.logError("Load error", err));
      }
    }
    if (name === "debug") {
      this.setDebugVisible(newVal !== null);
    }
    if (name === "vanilla") {
      this.syncMode();
    }
  }

  /** Load immediately if autoplay is set; otherwise arm the play button so the
   * first click loads-then-plays (avoids uploading all models' weights to VRAM
   * simultaneously on pages with multiple players declared upfront in markup). Only
   * used for a `src` already present at initial connect -- see attributeChangedCallback. */
  private maybeLoad(src: string): void {
    if (this.hasAttribute("autoplay")) {
      void this.load(src).catch((err: unknown) => this.logError("Load error", err));
    } else {
      this.placeholderEl.textContent = "Click ▶ to load";
      this.playBtn.disabled = false;
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  async load(src: string): Promise<void> {
    const myGeneration = ++this.loadGeneration;
    this.loadInProgress = true;
    this.stopPlayback();
    // Eagerly release the previous model's GPU/session resources instead of just
    // dropping the reference -- WebGPU buffers/pipelines and ORT sessions otherwise
    // linger until GC gets around to the whole object graph, which is exactly the kind
    // of per-frame-scale leak this player's memory work has been about avoiding. Fire-
    // and-forget: nothing below depends on either completing, and this model is already
    // unreachable from any future render.
    this.model?.destroy();
    if (this.onnxModel) void this.onnxModel.destroy().catch((err: unknown) => console.error("[kuma-player] session release error", err));
    this.model = undefined;
    this.onnxModel = undefined;
    this.gpuRenderer = undefined;
    // Abandon any frames still in flight for the previous model -- submitFrame's own
    // completion callbacks check loadGeneration and will no-op now that it's stale, so
    // this counter is safe to reset out from under them rather than waiting them out.
    this.inFlight = 0;
    this.pendingT = undefined;
    this.setControlsEnabled(false);
    this.resetCanvas();
    this.placeholderEl.textContent = "Loading…";
    this.placeholderEl.style.display = "";
    this.statusEl.textContent = "";
    this.logEl.textContent = `loading ${src}…`;

    // Fetch once so we can peek the format before dispatching to the right model class.
    const res = await fetch(src);
    if (!res.ok) throw new Error(`Failed to fetch "${src}": ${res.status} ${res.statusText}`);
    const bytes = await res.arrayBuffer();
    if (myGeneration !== this.loadGeneration) return; // superseded by a newer load() while fetching
    const format = peekIphFormat(bytes);
    const isOnnx = format === "onnx" || format === "onnx-branching";

    let inputs: ReadonlyArray<{ name: string; shape?: number[] }>;
    let outputs: ReadonlyArray<{ name: string; shape?: number[] }>;
    let durationSeconds: number | undefined;

    if (isOnnx) {
      const onnxModel = await OnnxModel.load(bytes, { executionProviders: ["webgpu"] });
      if (myGeneration !== this.loadGeneration) {
        // Superseded while loading -- this model never got wired into `this`, so nothing
        // else can reach it; release it directly instead of leaking its sessions.
        void onnxModel.destroy().catch((err: unknown) => console.error("[kuma-player] session release error", err));
        return;
      }
      this.onnxModel = onnxModel;
      inputs = onnxModel.inputs;
      outputs = onnxModel.outputs;
      durationSeconds = onnxModel.playback?.duration_seconds;
      this.log(`format: ${format}`);
      this.log(`inputs: ${JSON.stringify(inputs)}`);
      this.log(`outputs: ${JSON.stringify(outputs)}`);
      this.log("compiling shaders…");
      const t0onnx = performance.now();
      await onnxModel.warmAll((done, total) => {
        this.log(`  segment ${done}/${total} compiled`);
      });
      if (myGeneration !== this.loadGeneration) {
        void onnxModel.destroy().catch((err: unknown) => console.error("[kuma-player] session release error", err));
        return;
      }
      this.log(`shader compilation done in ${(performance.now() - t0onnx).toFixed(0)}ms.`);
    } else {
      // `low-memory` attribute → memory-constrained posture (pool depth 1 = one frame in
      // flight, the dominant peak-memory lever; small buffers still pooled). For Chromebooks
      // / integrated GPUs where VRAM is shared system RAM. `pool-depth="N"` overrides the
      // depth explicitly (and thus how many frames the player pipelines, since maxInFlight
      // follows pool depth) -- raise it on devices with memory headroom for smoother
      // playback; each extra depth is ~one more in-flight frame's worth of activations.
      const lowMemory = this.hasAttribute("low-memory");
      const depthAttr = Number(this.getAttribute("pool-depth"));
      const bufferPoolDepth = Number.isInteger(depthAttr) && depthAttr > 0 ? depthAttr : undefined;
      const model = await KumaModel.load(bytes, { lowMemory, bufferPoolDepth });
      if (myGeneration !== this.loadGeneration) {
        // Superseded while loading -- never wired into `this`, so safe (and necessary,
        // to avoid leaking its device/buffers) to destroy directly.
        model.destroy();
        return;
      }
      this.model = model;
      if (bufferPoolDepth !== undefined) this.log(`pool depth ${bufferPoolDepth}: up to ${bufferPoolDepth} frame(s) in flight`);
      else if (lowMemory) this.log("low-memory mode: pool depth 1, 1 frame in flight (small buffers pooled)");
      this.gpuRenderer = new GpuFrameRenderer(model.gpuDevice, this.canvas);
      inputs = model.inputs;
      outputs = model.outputs;
      durationSeconds = model.playback?.duration_seconds;
      this.log(`inputs: ${JSON.stringify(inputs)}`);
      this.log(`outputs: ${JSON.stringify(outputs)}`);

      this.log("warming up…");
      const t0 = performance.now();
      await model.warmUp();
      if (myGeneration !== this.loadGeneration) {
        model.destroy();
        return;
      }
      this.log(`warm-up done in ${(performance.now() - t0).toFixed(0)}ms.`);
    }

    const playback = isOnnx ? this.onnxModel!.playback : this.model!.playback;
    this.playFrameIntervalMs = playback?.fps ? 1000 / playback.fps : DEFAULT_PLAY_FRAME_INTERVAL_MS;

    if (durationSeconds !== undefined) {
      this.playSweepMs = durationSeconds * 1000;
      this.log(
        `playback: ${durationSeconds}s/sweep` +
          (playback?.fps !== undefined ? ` @ ${playback.fps}fps` : "") +
          ` (from .iph metadata)`,
      );
    } else {
      this.playSweepMs = DEFAULT_PLAY_SWEEP_MS;
      this.log(`playback: no duration_seconds — using ${(DEFAULT_PLAY_SWEEP_MS / 1000).toFixed(1)}s default.`);
    }

    // shape is often absent for ONNX manifests; default to [1] (scalar time input).
    const inputSpec = inputs[0]!;
    const n = (inputSpec.shape ?? [1]).reduce((a, b) => a * b, 1);
    const isScalar = n === 1;
    this.timeInputName = isScalar ? inputSpec.name : undefined;

    if (!isOnnx) this.verifyBtn.disabled = false;
    if (isScalar) {
      this.scrubber.disabled = false;
      this.playBtn.disabled = false;
      if (!isOnnx) this.profileBtn.disabled = false;
      this.log(`scrubbing "${inputSpec.name}" over t ∈ [0, 1].`);
      this.submitFrame(Number(this.scrubber.value));
    } else {
      this.log(`"${inputSpec.name}" has ${n} elements — scrubber disabled.`);
    }

    this.placeholderEl.style.display = "none";
    this.canvas.style.display = "";

    this.loadInProgress = false;
    if (isScalar && (this.hasAttribute("autoplay") || this.loadPending)) {
      this.loadPending = false;
      this.startPlayback();
    }
  }

  // ── Internal ────────────────────────────────────────────────────────────────

  /** Replace the canvas element so the next load always starts with a fresh context.
   * A canvas can only have one context type (webgpu vs 2d) for its lifetime, so we
   * must swap it out before switching between KumaModel (WebGPU) and OnnxModel (2D). */
  private resetCanvas(): void {
    const fresh = document.createElement("canvas");
    fresh.className = "kp-canvas";
    fresh.style.display = "none";
    this.canvas.replaceWith(fresh);
    this.canvas = fresh;
  }

  /** Render a CHW/NCHW Float32Array output to the canvas via the 2D API. */
  private renderCpuFrame(data: Float32Array, shape: readonly number[]): boolean {
    const dims = shape.length === 4 ? shape.slice(1) : [...shape];
    if (dims.length !== 3) return false;
    const [channels, height, width] = dims as [number, number, number];
    if (channels !== 1 && channels !== 3) return false;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return false;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const imageData = ctx.createImageData(width, height);
    const plane = width * height;
    const px = imageData.data;
    for (let i = 0; i < plane; i++) {
      const r = Math.round(Math.max(0, Math.min(1, data[i]!)) * 255);
      px[i * 4]     = r;
      px[i * 4 + 1] = channels >= 2 ? Math.round(Math.max(0, Math.min(1, data[plane + i]!)) * 255) : r;
      px[i * 4 + 2] = channels >= 3 ? Math.round(Math.max(0, Math.min(1, data[2 * plane + i]!)) * 255) : r;
      px[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return true;
  }

  private log(line: string): void {
    this.logEl.textContent += `\n${line}`;
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  private logError(prefix: string, err: unknown): void {
    this.loadInProgress = false;
    this.loadPending = false;
    const msg = err instanceof Error ? err.message : String(err);
    this.log(`${prefix}: ${msg}`);
    console.error("[kuma-player]", err);
    this.shadow.querySelector("details")!.open = true;
  }

  private setControlsEnabled(enabled: boolean): void {
    this.playBtn.disabled = !enabled;
    this.scrubber.disabled = !enabled;
    this.verifyBtn.disabled = !enabled;
    this.profileBtn.disabled = !enabled;
  }

  private updateT(t: number): void {
    this.scrubber.value = String(t);
    this.progressFill.style.width = `${t * 100}%`;
    this.timeEl.textContent = t.toFixed(3);
  }

  private formatFps(elapsedMs: number): string {
    if (elapsedMs <= 0) return "∞fps";
    return `${(1000 / elapsedMs).toFixed(1)}fps`;
  }

  private setDebugVisible(show: boolean): void {
    if (show && this.hasAttribute("vanilla")) show = false;
    this.debugVisible = show;
    this.debugPanel.classList.toggle("hidden", !show);
    this.debugBtn.classList.toggle("active", show);
  }

  private syncMode(): void {
    if (this.hasAttribute("vanilla")) {
      this.setDebugVisible(false);
    } else if (this.hasAttribute("debug")) {
      this.setDebugVisible(true);
    }
  }

  private togglePlay(): void {
    if (this.playing) {
      this.stopPlayback();
    } else if (!this.model && !this.loadInProgress) {
      const src = this.getAttribute("src");
      if (src) {
        this.loadPending = true;
        void this.load(src).catch((err: unknown) => this.logError("Load error", err));
      }
    } else if (this.model) {
      this.startPlayback();
    }
  }

  private startPlayback(): void {
    this.playing = true;
    this.playBtn.innerHTML = ICON_PAUSE;
    this.playBtn.setAttribute("aria-label", "Pause");
    // Start from current scrubber position rather than always resetting to 0.
    this.playStartMs = performance.now() - Number(this.scrubber.value) * this.playSweepMs;
    this.capacityHits = 0;
    this.requestedFrames = 0;
    this.lastPlayFrameMs = -Infinity;
    requestAnimationFrame((ts) => this.playTick(ts));
  }

  /** In-flight frame cap: the loaded model's real pool depth, or the cold-start default
   * before one has loaded. A `low-memory` model reports 1 here. */
  private get maxInFlight(): number {
    return this.model?.maxInFlight ?? DEFAULT_MAX_IN_FLIGHT;
  }

  private stopPlayback(): void {
    if (this.playing && this.requestedFrames > 0) {
      this.log(
        `Play: ${this.capacityHits}/${this.requestedFrames} frames ` +
          `(${((100 * this.capacityHits) / this.requestedFrames).toFixed(1)}%) hit MAX_IN_FLIGHT=${this.maxInFlight} cap.`,
      );
    }
    this.playing = false;
    this.playBtn.innerHTML = ICON_PLAY;
    this.playBtn.setAttribute("aria-label", "Play");
  }

  /** Submit a frame render at time `t` — latest-wins under the in-flight cap. */
  private submitFrame(t: number): void {
    const name = this.timeInputName;
    if (!name) return;

    // Captured now so this call's own completion callbacks can tell, once they finally
    // resolve, whether the user has since switched to a different model (load() bumps
    // this) -- if so, this.model/this.onnxModel/this.inFlight/etc. already belong to
    // that newer model, and this frame's stale result must not touch any of it.
    const generation = this.loadGeneration;
    this.requestedFrames++;
    // ONNX InferenceSession.run() is not concurrent-safe on the same session —
    // limit to 1 in-flight for ONNX, use the loaded KumaModel's real pool depth otherwise.
    const maxInFlight = this.onnxModel ? 1 : this.maxInFlight;
    if (this.inFlight >= maxInFlight) {
      this.capacityHits++;
      this.pendingT = t;
      return;
    }
    this.pendingT = undefined;
    this.inFlight++;
    const seq = this.frameSequence++;
    const input = new Float32Array([t]);
    const t0 = performance.now();

    const drain = () => {
      // The previous model's inFlight/pendingT counters no longer apply once a newer
      // load() has reset them for its own model -- decrementing or pulling a pending
      // frame here would corrupt the new model's bookkeeping instead of this stale one's.
      if (generation !== this.loadGeneration) return;
      this.inFlight--;
      if (this.pendingT !== undefined) {
        const next = this.pendingT;
        this.pendingT = undefined;
        this.submitFrame(next);
      }
    };

    if (this.onnxModel) {
      // ONNX path: inference → 2D canvas rendering.
      const om = this.onnxModel;
      const outSpec = om.outputs[0];
      void om
        .run({ [name]: input })
        .then((outputs) => {
          if (generation !== this.loadGeneration) return; // model switched away from mid-run; discard
          if (seq > this.latestRenderedSequence) {
            this.latestRenderedSequence = seq;
            const outName = outSpec?.name ?? Object.keys(outputs)[0]!;
            const result = outputs[outName];
            // shape comes from the real ort tensor dims, not the manifest
            if (result) this.renderCpuFrame(result.data, result.shape);
            const elapsed = performance.now() - t0;
            this.statusEl.textContent =
              `t=${t.toFixed(3)}  ${this.formatFps(elapsed)}  "${outName}"  ${JSON.stringify(result?.shape ?? "?")}`;
          }
        })
        .catch((err: unknown) => {
          // A stale generation's session may have been release()d out from under this
          // call by a newer load() -- that's an expected rejection, not a real error.
          if (generation !== this.loadGeneration) return;
          this.logError(`Error at t=${t.toFixed(3)}`, err);
          this.stopPlayback();
        })
        .finally(drain);
      return;
    }

    // KumaModel (WebGPU) path.
    const m = this.model;
    if (!m) { drain(); return; }
    void m
      .runToGpu({ [name]: input })
      .then(async (outputs) => {
        const current = generation === this.loadGeneration;
        if (current && seq > this.latestRenderedSequence) {
          this.latestRenderedSequence = seq;
          for (const out of outputs) this.gpuRenderer?.render(out.buffer, out.shape);
          const elapsed = performance.now() - t0;
          if (elapsed > STALL_THRESHOLD_MS) {
            console.warn(`[kuma-player] runToGpu spike: ${elapsed.toFixed(1)}ms at t=${t.toFixed(3)}`);
          }
          if (outputs[0]) {
            this.statusEl.textContent =
              `t=${t.toFixed(3)}  ${this.formatFps(elapsed)} enqueue  ` +
              `"${outputs[0].name}"  ${JSON.stringify(outputs[0].shape)}`;
          }
          // Only worth waiting for real completion (for the stall-spike measurement
          // above) when this is still the live model -- a stale model's device may
          // already be destroy()d by a newer load(), which would reject this wait.
          await m.gpuDevice.queue.onSubmittedWorkDone();
        }
        // Output buffers are large (≥ frame size), bypass the pool, and are never
        // destroyed by the scheduler -- caller owns them. Destroy explicitly now that
        // both the compute pass that wrote them and the render pass that read them
        // are confirmed done (or, if stale, right away -- destroy() on a buffer whose
        // device is already gone is a safe no-op).
        for (const out of outputs) out.buffer.destroy();
      })
      .catch((err: unknown) => {
        // A stale generation's device may have been destroy()d out from under this call
        // by a newer load() -- that's an expected rejection, not a real error.
        if (generation !== this.loadGeneration) return;
        this.logError(`Error at t=${t.toFixed(3)}`, err);
        this.stopPlayback();
      })
      .finally(drain);
  }

  private playTick(nowMs: number): void {
    if (!this.playing) return;
    if (Number.isFinite(this.lastTickMs) && nowMs - this.lastTickMs > STALL_THRESHOLD_MS) {
      console.warn(`[kuma-player] main-thread stall: ${(nowMs - this.lastTickMs).toFixed(1)}ms between rAF callbacks`);
    }
    this.lastTickMs = nowMs;
    if (nowMs - this.lastPlayFrameMs >= this.playFrameIntervalMs) {
      this.lastPlayFrameMs = nowMs;
      const t = ((nowMs - this.playStartMs) % this.playSweepMs) / this.playSweepMs;
      this.updateT(t);
      this.submitFrame(t);
    }
    requestAnimationFrame((ts) => this.playTick(ts));
  }

  private async runVerify(): Promise<void> {
    if (!this.model) return;
    this.verifyBtn.disabled = true;
    this.log("running verify()…");
    const t0 = performance.now();
    const report = await this.model.verify();
    this.log(`verify done in ${(performance.now() - t0).toFixed(0)}ms:\n${this.formatVerifyReport(report)}`);
    console.log("[kuma-player] verify report", report);
    this.verifyBtn.disabled = false;
  }

  private async runProfile(): Promise<void> {
    const m = this.model;
    const name = this.timeInputName;
    if (!m || !name) return;
    this.profileBtn.disabled = true;
    this.log(`running profile() ×${PROFILE_WARMUP_RUNS}…`);
    const input = new Float32Array([Number(this.scrubber.value)]);
    let report: (ProfileReport & { realMilliseconds: number; baselineMilliseconds: number }) | undefined;
    const realMs: number[] = [];
    const baseMs: number[] = [];
    for (let i = 0; i < PROFILE_WARMUP_RUNS; i++) {
      report = await m.profile({ [name]: input });
      realMs.push(report.realMilliseconds);
      baseMs.push(report.baselineMilliseconds);
    }
    this.log(`real ms across ${PROFILE_WARMUP_RUNS} runs: ${realMs.map((v) => v.toFixed(1)).join(", ")}`);
    this.log(`baseline ms: ${baseMs.map((v) => v.toFixed(1)).join(", ")}`);
    this.log(this.formatProfileReport(report!));
    console.log("[kuma-player] profile report (last run)", report);
    this.profileBtn.disabled = false;
  }

  private formatVerifyReport(report: VerifyReport): string {
    const lines = [`${report.ok ? "PASS" : "FAIL"} (${report.branches.length} branch(es))`];
    for (const b of report.branches) {
      lines.push(`  branch ${b.branch}: ${b.nodesChecked} checked, ${b.mismatches.length} mismatches, ${b.nodesMissing.length} missing`);
      for (const m of b.mismatches) {
        for (const d of m.diffs) {
          lines.push(`    ${m.node}[${m.part}] ${d.field}: golden=${d.golden} actual=${d.actual}`);
        }
      }
    }
    return lines.join("\n");
  }

  private formatProfileReport(
    report: ProfileReport & { realMilliseconds: number; baselineMilliseconds: number },
  ): string {
    const lines = [
      `real=${report.realMilliseconds.toFixed(1)}ms  baseline=${report.baselineMilliseconds.toFixed(1)}ms  per-op total=${(report.totalMicroseconds / 1000).toFixed(2)}ms (${report.perNode.length} nodes)`,
      `top ops by % of per-op total:`,
    ];
    for (const op of report.byTarget.slice(0, 12)) {
      const pct = (100 * op.totalMicroseconds) / report.totalMicroseconds;
      lines.push(`  ${op.target.padEnd(30)} ${pct.toFixed(1).padStart(5)}%  n=${op.count}  avg=${op.avgMicroseconds.toFixed(1)}µs`);
    }
    return lines.join("\n");
  }
}

if (typeof customElements !== "undefined" && !customElements.get("kuma-player")) {
  customElements.define("kuma-player", KumaPlayer);
}
