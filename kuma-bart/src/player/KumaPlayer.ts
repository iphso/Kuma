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
 *   import 'kuma-bart/player';  // registers <kuma-player>
 *   <kuma-player src="/artifacts/model.iph"></kuma-player>
 */

import { BUFFER_POOL_DEPTH } from "../engine/context.js";
import type { VerifyReport } from "../engine/verify.js";
import type { ProfileReport } from "../engine/profile.js";
import { peekIphFormat } from "../iph/loader.js";
import { KumaBackend, OnnxBackend, type PlayerBackend } from "./backends.js";
import { LatestWinsScheduler, type FrameTicket } from "./frameScheduler.js";
import { ICON_PAUSE, ICON_PLAY, STYLES, TEMPLATE } from "./styles.js";

// Cold-start fallback only: used before a model has loaded. Once one has, the player
// uses the backend's real cap (a `low-memory` model reports 1).
const DEFAULT_MAX_IN_FLIGHT = BUFFER_POOL_DEPTH;
const DEFAULT_PLAY_SWEEP_MS = 6000;
// Fallback pacing for models with no fps in their .iph metadata -- a reasonable
// scrubbing/preview cadence, not a target to hit when the real fps is known.
const DEFAULT_PLAY_FRAME_INTERVAL_MS = 1000 / 60;
const STALL_THRESHOLD_MS = 50;
const PROFILE_WARMUP_RUNS = 5;

type ProfileResult = ProfileReport & { realMilliseconds: number; baselineMilliseconds: number };

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
  private backend: PlayerBackend | undefined;
  private timeInputName: string | undefined;
  private readonly frames = new LatestWinsScheduler(
    () => this.backend?.maxInFlight ?? DEFAULT_MAX_IN_FLIGHT,
    (t, ticket) => this.runFrame(t, ticket),
  );
  private playing = false;
  private loadPending = false; // true when play was clicked before model loaded
  private loadInProgress = false;
  private hasConnectedBefore = false;
  // Bumped at the start of every load() so a superseded in-flight load() can detect it's
  // stale and bail out instead of touching the newer model's state.
  private loadGeneration = 0;
  private playStartMs = 0;
  private lastTickMs = -Infinity;
  private playSweepMs = DEFAULT_PLAY_SWEEP_MS;
  // How often playTick submits a new inference: the model's own authored fps (from .iph
  // playback metadata), so playback doesn't sample in-between times the source never
  // had -- that reads as motion-smoothed/sped-up rather than faithful playback.
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
    if (name === "src" && newVal !== null && newVal !== oldVal && this.hasConnectedBefore) {
      // A `src` change after the first connect can only come from imperative code (e.g. a
      // model picker) -- a deliberate selection, so load immediately. A `src` present at
      // first connect goes through maybeLoad's lazy gate instead (custom-element upgrade
      // runs attributeChangedCallback for existing attributes before connectedCallback,
      // while hasConnectedBefore is still false).
      void this.load(newVal).catch((err: unknown) => this.logError("Load error", err));
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
    const stale = () => myGeneration !== this.loadGeneration;
    this.loadInProgress = true;
    this.stopPlayback();
    // Release the previous model's GPU device / sessions now rather than whenever GC
    // reaches them, and abandon its in-flight frames (their tickets go stale).
    this.backend?.destroy();
    this.backend = undefined;
    this.frames.reset();
    this.setControlsEnabled(false);
    this.resetCanvas();
    this.placeholderEl.textContent = "Loading…";
    this.placeholderEl.style.display = "";
    this.statusEl.textContent = "";
    this.logEl.textContent = `loading ${src}…`;

    // Fetch once so we can peek the format before dispatching to the right backend.
    const res = await fetch(src);
    if (!res.ok) throw new Error(`Failed to fetch "${src}": ${res.status} ${res.statusText}`);
    const bytes = await res.arrayBuffer();
    if (stale()) return;
    const format = peekIphFormat(bytes) ?? "kuma";

    let backend: PlayerBackend;
    if (format === "onnx" || format === "onnx-branching") {
      backend = await OnnxBackend.load(bytes, format, this.canvas);
    } else {
      // `low-memory` → pool depth 1 (one frame in flight; small buffers still pooled), for
      // Chromebooks / integrated GPUs where VRAM is shared system RAM. `pool-depth="N"`
      // sets the depth (and so the in-flight cap) explicitly.
      const lowMemory = this.hasAttribute("low-memory");
      const depthAttr = Number(this.getAttribute("pool-depth"));
      const bufferPoolDepth = Number.isInteger(depthAttr) && depthAttr > 0 ? depthAttr : undefined;
      backend = await KumaBackend.load(bytes, this.canvas, { lowMemory, bufferPoolDepth });
      if (bufferPoolDepth !== undefined) this.log(`pool depth ${bufferPoolDepth}: up to ${bufferPoolDepth} frame(s) in flight`);
      else if (lowMemory) this.log("low-memory mode: pool depth 1, 1 frame in flight (small buffers pooled)");
    }
    if (stale()) {
      // Never wired into `this`, so nothing else can reach it.
      backend.destroy();
      return;
    }
    this.backend = backend;
    this.log(`format: ${format}`);
    this.log(`inputs: ${JSON.stringify(backend.inputs)}`);
    this.log(`outputs: ${JSON.stringify(backend.outputs)}`);
    await backend.warmUp((line) => this.log(line));
    if (stale()) {
      backend.destroy();
      return;
    }

    const playback = backend.playback;
    this.playFrameIntervalMs = playback?.fps ? 1000 / playback.fps : DEFAULT_PLAY_FRAME_INTERVAL_MS;
    const durationSeconds = playback?.duration_seconds;
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
    const inputSpec = backend.inputs[0]!;
    const n = (inputSpec.shape ?? [1]).reduce((a, b) => a * b, 1);
    const isScalar = n === 1;
    this.timeInputName = isScalar ? inputSpec.name : undefined;

    const canDebug = backend.kumaModel !== undefined;
    this.verifyBtn.disabled = !canDebug;
    if (isScalar) {
      this.scrubber.disabled = false;
      this.playBtn.disabled = false;
      this.profileBtn.disabled = !canDebug;
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
   * A canvas can only have one context type (webgpu vs 2d) for its lifetime, so it must
   * be swapped out before switching between the WebGPU and ONNX (2D) backends. */
  private resetCanvas(): void {
    const fresh = document.createElement("canvas");
    fresh.className = "kp-canvas";
    fresh.style.display = "none";
    this.canvas.replaceWith(fresh);
    this.canvas = fresh;
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
    } else if (!this.backend && !this.loadInProgress) {
      const src = this.getAttribute("src");
      if (src) {
        this.loadPending = true;
        void this.load(src).catch((err: unknown) => this.logError("Load error", err));
      }
    } else if (this.backend) {
      this.startPlayback();
    }
  }

  private startPlayback(): void {
    this.playing = true;
    this.playBtn.innerHTML = ICON_PAUSE;
    this.playBtn.setAttribute("aria-label", "Pause");
    // Start from current scrubber position rather than always resetting to 0.
    this.playStartMs = performance.now() - Number(this.scrubber.value) * this.playSweepMs;
    this.frames.resetStats();
    this.lastPlayFrameMs = -Infinity;
    requestAnimationFrame((ts) => this.playTick(ts));
  }

  private stopPlayback(): void {
    const { requested, capacityHits } = this.frames;
    if (this.playing && requested > 0) {
      const cap = this.backend?.maxInFlight ?? DEFAULT_MAX_IN_FLIGHT;
      this.log(
        `Play: ${capacityHits}/${requested} frames (${((100 * capacityHits) / requested).toFixed(1)}%) ` +
          `hit the ${cap}-frame in-flight cap.`,
      );
    }
    this.playing = false;
    this.playBtn.innerHTML = ICON_PLAY;
    this.playBtn.setAttribute("aria-label", "Play");
  }

  /** Request a frame at time `t` — latest-wins under the in-flight cap. */
  private submitFrame(t: number): void {
    if (this.timeInputName) this.frames.submit(t);
  }

  private async runFrame(t: number, ticket: FrameTicket): Promise<void> {
    const backend = this.backend;
    const name = this.timeInputName;
    if (!backend || !name) return;
    try {
      const status = await backend.renderFrame({ [name]: new Float32Array([t]) }, ticket);
      if (status !== undefined && ticket.isCurrent()) this.statusEl.textContent = `t=${t.toFixed(3)}  ${status}`;
    } catch (err) {
      // A stale frame's device/session may have been released by a newer load() --
      // an expected rejection, not a real error.
      if (!ticket.isCurrent()) return;
      this.logError(`Error at t=${t.toFixed(3)}`, err);
      this.stopPlayback();
    }
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
    const model = this.backend?.kumaModel;
    if (!model) return;
    this.verifyBtn.disabled = true;
    this.log("running verify()…");
    const t0 = performance.now();
    const report = await model.verify();
    this.log(`verify done in ${(performance.now() - t0).toFixed(0)}ms:\n${this.formatVerifyReport(report)}`);
    console.log("[kuma-player] verify report", report);
    this.verifyBtn.disabled = false;
  }

  private async runProfile(): Promise<void> {
    const model = this.backend?.kumaModel;
    const name = this.timeInputName;
    if (!model || !name) return;
    this.profileBtn.disabled = true;
    this.log(`running profile() ×${PROFILE_WARMUP_RUNS}…`);
    const input = new Float32Array([Number(this.scrubber.value)]);
    let report: ProfileResult | undefined;
    const realMs: number[] = [];
    const baseMs: number[] = [];
    for (let i = 0; i < PROFILE_WARMUP_RUNS; i++) {
      report = await model.profile({ [name]: input });
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

  private formatProfileReport(report: ProfileResult): string {
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
