import { KumaModel, type KumaMemoryOptions } from "../model.js";
import { OnnxModel } from "../onnx/model.js";
import { releaseOutputs } from "../engine/scheduler.js";
import type { PlaybackMeta } from "../types/manifest.js";
import { GpuFrameRenderer } from "./gpuRender.js";
import type { FrameTicket } from "./frameScheduler.js";

export interface IOInfo {
  name: string;
  shape?: number[];
}

/** What <kuma-player> needs from a loaded package, whichever runtime executes it. */
export interface PlayerBackend {
  readonly format: string;
  readonly inputs: readonly IOInfo[];
  readonly outputs: readonly IOInfo[];
  readonly playback: PlaybackMeta | undefined;
  /** How many frames may be in flight at once. */
  readonly maxInFlight: number;
  /** Set when verify()/profile() are available (WebGPU/WGSL packages only). */
  readonly kumaModel?: KumaModel;
  /** Pays shader compilation up front, during the load wait. */
  warmUp(log: (line: string) => void): Promise<void>;
  /** Runs one frame and renders it if `ticket.claimRender()` allows. Resolves to a
   * status line for a rendered frame, undefined otherwise. */
  renderFrame(feeds: Record<string, Float32Array>, ticket: FrameTicket): Promise<string | undefined>;
  /** Releases the backend's GPU device / sessions. Safe to call more than once. */
  destroy(): void;
}

const STALL_THRESHOLD_MS = 50;

function formatFps(elapsedMs: number): string {
  return elapsedMs <= 0 ? "∞fps" : `${(1000 / elapsedMs).toFixed(1)}fps`;
}

/** WGSL `.iph` packages: runToGpu + a GPU-direct render pass, no CPU readback. */
export class KumaBackend implements PlayerBackend {
  readonly format = "kuma";
  private readonly renderer: GpuFrameRenderer;

  constructor(readonly kumaModel: KumaModel, canvas: HTMLCanvasElement) {
    this.renderer = new GpuFrameRenderer(kumaModel.gpuDevice, canvas);
  }

  static async load(bytes: ArrayBuffer, canvas: HTMLCanvasElement, memory: KumaMemoryOptions): Promise<KumaBackend> {
    return new KumaBackend(await KumaModel.load(bytes, memory), canvas);
  }

  get inputs(): readonly IOInfo[] {
    return this.kumaModel.inputs;
  }
  get outputs(): readonly IOInfo[] {
    return this.kumaModel.outputs;
  }
  get playback(): PlaybackMeta | undefined {
    return this.kumaModel.playback;
  }
  get maxInFlight(): number {
    return this.kumaModel.maxInFlight;
  }

  async warmUp(log: (line: string) => void): Promise<void> {
    log("warming up…");
    const t0 = performance.now();
    await this.kumaModel.warmUp();
    log(`warm-up done in ${(performance.now() - t0).toFixed(0)}ms.`);
  }

  async renderFrame(feeds: Record<string, Float32Array>, ticket: FrameTicket): Promise<string | undefined> {
    const t0 = performance.now();
    const outputs = await this.kumaModel.runToGpu(feeds);
    try {
      if (!ticket.claimRender()) return undefined;
      for (const out of outputs) this.renderer.render(out.buffer, out.shape);
      const elapsed = performance.now() - t0;
      if (elapsed > STALL_THRESHOLD_MS) {
        console.warn(`[kuma-player] runToGpu spike: ${elapsed.toFixed(1)}ms`);
      }
      // Wait for real completion so the in-flight cap tracks GPU work, not just
      // submission; skipped for a stale frame, whose device may already be destroyed.
      await this.kumaModel.gpuDevice.queue.onSubmittedWorkDone();
      const first = outputs[0];
      return first ? `${formatFps(elapsed)} enqueue  "${first.name}"  ${JSON.stringify(first.shape)}` : undefined;
    } finally {
      // Fresh (non-pooled) output buffers are this caller's to free; destroy() on a
      // device that is already gone is a safe no-op.
      releaseOutputs(outputs);
    }
  }

  destroy(): void {
    this.kumaModel.destroy();
  }
}

/** ONNX `.iph` packages, run via onnxruntime-web, rendered through the 2D canvas API.
 * InferenceSession.run() is not concurrent-safe, hence one frame in flight. */
export class OnnxBackend implements PlayerBackend {
  readonly maxInFlight = 1;

  constructor(
    private readonly model: OnnxModel,
    readonly format: string,
    private readonly canvas: HTMLCanvasElement,
  ) {}

  static async load(bytes: ArrayBuffer, format: string, canvas: HTMLCanvasElement): Promise<OnnxBackend> {
    return new OnnxBackend(await OnnxModel.load(bytes, { executionProviders: ["webgpu"] }), format, canvas);
  }

  get inputs(): readonly IOInfo[] {
    return this.model.inputs;
  }
  get outputs(): readonly IOInfo[] {
    return this.model.outputs;
  }
  get playback(): PlaybackMeta | undefined {
    return this.model.playback;
  }

  async warmUp(log: (line: string) => void): Promise<void> {
    log("compiling shaders…");
    const t0 = performance.now();
    await this.model.warmAll((done, total) => log(`  segment ${done}/${total} compiled`));
    log(`shader compilation done in ${(performance.now() - t0).toFixed(0)}ms.`);
  }

  async renderFrame(feeds: Record<string, Float32Array>, ticket: FrameTicket): Promise<string | undefined> {
    const t0 = performance.now();
    const outputs = await this.model.run(feeds);
    if (!ticket.claimRender()) return undefined;
    const outName = this.model.outputs[0]?.name ?? Object.keys(outputs)[0]!;
    const result = outputs[outName];
    // shape comes from the real ort tensor dims, not the manifest
    if (result) this.renderCpuFrame(result.data, result.shape);
    return `${formatFps(performance.now() - t0)}  "${outName}"  ${JSON.stringify(result?.shape ?? "?")}`;
  }

  destroy(): void {
    void this.model.destroy().catch((err: unknown) => console.error("[kuma-player] session release error", err));
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
    const byte = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255);
    for (let i = 0; i < plane; i++) {
      const r = byte(data[i]!);
      px[i * 4] = r;
      px[i * 4 + 1] = channels >= 2 ? byte(data[plane + i]!) : r;
      px[i * 4 + 2] = channels >= 3 ? byte(data[2 * plane + i]!) : r;
      px[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return true;
  }
}
