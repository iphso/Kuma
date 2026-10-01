import { loadIphPackage } from "./iph/loader.js";
import { requestComputeDevice, uploadBufferSlice, uploadFloat32 } from "webgpu-compute-kit";
import { runGraph, releaseOutputs, type RunGraphOutput, type RunGraphParams } from "./engine/scheduler.js";
import { verifyAgainstGolden, warmUp, type VerifyReport } from "./engine/verify.js";
import { profileGraph, type ProfileReport } from "./engine/profile.js";
import {
  createBufferPoolState,
  DEFAULT_BUFFER_POOL_DEPTH,
  DEFAULT_POOL_SIZE_THRESHOLD,
  type BufferPoolState,
  type ResolvedTensor,
} from "./engine/context.js";
import type { SnippetFn } from "./engine/snippets.js";
import { KumaManifestError } from "./errors.js";
import type { IOSpec, KumaManifest, PlaybackMeta } from "./types/manifest.js";
import type { GoldenData } from "./types/golden.js";

/** Tuning knobs for how aggressively the runtime trades throughput for a smaller memory
 * footprint. All optional.
 * Aimed at shared-memory devices (Chromebooks, integrated GPUs) where VRAM *is* system
 * RAM and the headroom for pooled buffers + multiple in-flight frames simply isn't there.
 *
 * The compute cost of these 3M-param models is tiny; the binding constraint is memory, so
 * giving up the GC-pause-avoiding buffer pool and the multi-frame pipelining (neither of
 * which buys anything on a device that can't hit 60fps anyway) is close to free here. */
export interface KumaMemoryOptions {
  /** Memory-constrained posture (Chromebooks, integrated GPUs): pool depth 1 -- one frame
   * in flight, no pipelining -- which is the dominant peak-memory lever, since each extra
   * in-flight frame is a whole extra copy of the live activation set. Leaves the pool size
   * threshold at its default so small intermediates are still POOLED (reused across frames)
   * rather than freed-and-reallocated every frame: zeroing the threshold as well forces an
   * allocation + mid-graph-submit storm that makes playback visibly chunky, for a
   * steady-state saving of ~the pool's own footprint. A caller that truly needs the
   * absolute floor can still set `poolSizeThreshold: 0` explicitly. Individual fields
   * below always win if also set.
   *
   * Omit this entirely and `load()` guesses for you via webgpu-compute-kit's
   * isLikelyLowMemoryDevice (fallback/software adapter, or a low navigator.deviceMemory
   * reading) -- best-effort, since WebGPU has no portable free-VRAM query. Pass an
   * explicit `true`/`false` to skip the guess. */
  lowMemory?: boolean;
  /** Buffer-pool ring depth. Default 3 (DEFAULT_BUFFER_POOL_DEPTH), or 1 when lowMemory;
   * 1 disables rotation. Also caps how many frames KumaPlayer keeps in flight, since each
   * in-flight frame needs its own live intermediates -- the dominant peak-memory term
   * during playback. */
  bufferPoolDepth?: number;
  /** Bytes at or above which a buffer skips the persistent pool and is freed eagerly.
   * Default 1 MB. Lower it to shrink steady-state footprint toward just the weights, at
   * the cost of more allocation churn and more mid-graph GPU submits. 0 = free everything
   * eagerly. */
  poolSizeThreshold?: number;
}

function resolveMemoryOptions(
  options: KumaMemoryOptions,
  isLikelyLowMemory: boolean,
): {
  depth: number;
  sizeThreshold: number;
} {
  // Explicit lowMemory (true or false) always wins; omitted defers to the device-tier
  // guess from requestComputeDevice() (webgpu-compute-kit's isLikelyLowMemoryDevice).
  const lowMemory = options.lowMemory ?? isLikelyLowMemory;
  const depth = options.bufferPoolDepth ?? (lowMemory ? 1 : DEFAULT_BUFFER_POOL_DEPTH);
  // lowMemory drops depth (the big lever) but keeps the default pool threshold -- see the
  // KumaMemoryOptions.lowMemory doc for why zeroing it hurt smoothness for little gain.
  const sizeThreshold = options.poolSizeThreshold ?? DEFAULT_POOL_SIZE_THRESHOLD;
  return { depth: Math.max(1, depth), sizeThreshold: Math.max(0, sizeThreshold) };
}

/** A loaded `.iph` package, ready to run via WebGPU. */
export class KumaModel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly manifest: KumaManifest,
    private readonly kernels: ReadonlyMap<string, string>,
    private readonly snippets: ReadonlyMap<string, string>,
    private readonly weightBuffers: ReadonlyMap<string, ResolvedTensor>,
    private readonly pipelineCache: Map<string, GPUComputePipeline>,
    private readonly snippetCache: Map<string, SnippetFn>,
    private readonly golden: GoldenData | undefined,
    private readonly constantCache: Map<string, GPUBuffer>,
    private readonly bufferPool: BufferPoolState,
  ) {}

  static async load(
    source: ArrayBuffer | Response | string,
    memory: KumaMemoryOptions = {},
  ): Promise<KumaModel> {
    const { manifest, weights, kernels, snippets, golden } = await loadIphPackage(source);
    const { device, isLikelyLowMemory } = await requestComputeDevice();

    const weightBuffers = new Map<string, ResolvedTensor>();
    for (const w of manifest.weights) {
      const buffer = uploadBufferSlice(device, weights, w.byte_offset, w.byte_length);
      weightBuffers.set(w.name, { buffer, shape: w.shape });
    }

    const { depth, sizeThreshold } = resolveMemoryOptions(memory, isLikelyLowMemory);
    return new KumaModel(
      device,
      manifest,
      kernels,
      snippets,
      weightBuffers,
      new Map(),
      new Map(),
      golden,
      new Map(),
      createBufferPoolState({ depth, sizeThreshold }),
    );
  }

  /** Read-only access to the GPU device this model's buffers/pipelines live on -- for
   * a caller (e.g. the demo) that wants to render or otherwise consume a `runRaw()`/
   * `runToGpu()` output's raw `buffer` entirely on-GPU, since WebGPU resources are
   * device-scoped (you can't bind a buffer from this device into a pipeline created on
   * another), or that wants to call `gpuDevice.queue.onSubmittedWorkDone()` itself for
   * backpressure after a `runToGpu()` call. */
  get gpuDevice(): GPUDevice {
    return this.device;
  }

  /** How many run()/runRaw()/runToGpu() calls a caller should keep in flight at once.
   * Equals this model's buffer-pool depth: submitting more than this doesn't get more
   * frames genuinely in flight (runGraph's acquireGenerationSlot blocks at the top of the
   * (depth+1)-th concurrent call until an earlier one's GPU work is confirmed done) and,
   * more importantly here, each extra in-flight frame needs its own set of live
   * intermediates -- the main thing a low-memory model is trying to avoid. KumaPlayer
   * reads this instead of the BUFFER_POOL_DEPTH module constant so a model loaded with
   * `{ lowMemory: true }` (depth 1) actually serializes to one frame at a time. */
  get maxInFlight(): number {
    return this.bufferPool.depth;
  }

  get inputs(): readonly IOSpec[] {
    return this.manifest.inputs;
  }

  get outputs(): readonly IOSpec[] {
    return this.manifest.outputs;
  }

  /** Optional playback metadata (fps/duration_seconds) the Python exporter embeds when
   * the caller provided it (see kuma.manifest.build_playback_meta) -- undefined for any
   * .iph that didn't. A caller scrubbing/playing a normalized [0,1] time input should
   * use duration_seconds instead of assuming one fixed speed for every model: different
   * exports are authored at different speeds. */
  get playback(): PlaybackMeta | undefined {
    return this.manifest.playback;
  }

  /** Immediately releases this model's GPU device and everything allocated from it --
   * weight buffers, pooled/scratch buffers, pipelines -- in one call. Call this when
   * done with a model (e.g. right before loading a different one into the same player)
   * instead of just dropping the reference: each `load()` requests its own fresh
   * GPUDevice (see requestComputeDevice), so an abandoned model's full buffer set
   * otherwise lingers until JS GC gets around to collecting the whole device graph,
   * not the instant its owner stops using it. Safe to call more than once (device
   * destroy() is idempotent); every other method on this instance becomes unusable
   * afterward since its underlying device is gone. */
  destroy(): void {
    this.device.destroy();
  }

  private buildInputs(inputs: Record<string, Float32Array>): {
    inputBuffers: Map<string, ResolvedTensor>;
    rawInputs: Map<string, Float32Array>;
  } {
    const inputBuffers = new Map<string, ResolvedTensor>();
    const rawInputs = new Map<string, Float32Array>();
    for (const spec of this.manifest.inputs) {
      const data = inputs[spec.name];
      if (!data) {
        throw new KumaManifestError(
          `Missing input "${spec.name}" — expected one of: ${this.manifest.inputs.map((s) => s.name).join(", ")}`,
        );
      }
      const expected = spec.shape ? spec.shape.reduce((a, b) => a * b, 1) : data.length;
      if (data.length !== expected) {
        throw new KumaManifestError(
          `Input "${spec.name}" expected ${expected} elements (shape ${JSON.stringify(spec.shape)}), got ${data.length}.`,
        );
      }
      rawInputs.set(spec.name, data);
      const buffer = uploadFloat32(this.device, data);
      inputBuffers.set(spec.name, { buffer, shape: spec.shape ?? [data.length] });
    }
    return { inputBuffers, rawInputs };
  }

  /** Uploads `inputs`, runs `fn` against them, then destroys the uploaded buffers: this
   * model owns its input uploads (runGraph never destroys caller-provided inputs).
   * Destroying right after submit is safe -- WebGPU defers the actual free until the
   * submitted work using them completes. */
  private async withInputs<T>(
    inputs: Record<string, Float32Array>,
    fn: (params: RunGraphParams) => Promise<T>,
  ): Promise<T> {
    const { inputBuffers, rawInputs } = this.buildInputs(inputs);
    try {
      return await fn({
        device: this.device,
        manifest: this.manifest,
        kernels: this.kernels,
        pipelineCache: this.pipelineCache,
        weightBuffers: this.weightBuffers,
        inputBuffers,
        rawInputs,
        snippets: this.snippets,
        snippetCache: this.snippetCache,
        constantCache: this.constantCache,
        bufferPool: this.bufferPool,
      });
    } finally {
      for (const t of inputBuffers.values()) t.buffer.destroy();
    }
  }

  /** Like `run`, but returns the full RunGraphOutput[] (shape + the GPU buffer the
   * data was read back from, not just the data) -- for a caller that wants to render
   * or otherwise consume a result on-GPU. The caller owns the result: call each
   * output's `release()` (or `releaseOutputs`) once done with its `buffer`, or outputs
   * above the pool size threshold leak. */
  async runRaw(inputs: Record<string, Float32Array>): Promise<RunGraphOutput[]> {
    return this.withInputs(inputs, (params) => runGraph(params));
  }

  /** Like `runRaw`, but skips the GPU->CPU readback for the model's own outputs
   * entirely -- `data` on the result is an empty Float32Array(0); use `buffer` (e.g.
   * to render directly, which the GPU orders after this call's own submission with no
   * CPU-side wait needed), then `release()` it. The readback measured 60-70ms of a
   * ~65-75ms frame for a 720x1280x3 model, far more than encoding every dispatch
   * (~2ms), so an interactive path that only needs pixels on screen should use this.
   * For real GPU compute time, see profile()'s realMilliseconds. */
  async runToGpu(inputs: Record<string, Float32Array>): Promise<RunGraphOutput[]> {
    return this.withInputs(inputs, (params) => runGraph({ ...params, skipOutputReadback: true }));
  }

  async run(inputs: Record<string, Float32Array>): Promise<Record<string, Float32Array>> {
    const outputs = await this.runRaw(inputs);
    releaseOutputs(outputs);
    const result: Record<string, Float32Array> = {};
    for (const out of outputs) {
      result[out.name] = out.data;
    }
    return result;
  }

  /** Runs the same graph a real `run(inputs)` call would, but with GPU timestamp
   * queries around every dispatched node instead of one shared compute pass -- see
   * engine/profile.ts for what that does and doesn't distort (short version: its
   * totalMicroseconds runs 3-4x *higher* than real frame time, a hard consequence of
   * WebGPU's timestampWrites being pass-scoped -- treat byTarget's percentages as the
   * signal). `realMilliseconds` is measured the way the interactive path runs (skip the
   * output readback, wait for GPU completion via onSubmittedWorkDone) so there's one
   * trustworthy number in the same report. `baselineMilliseconds` measures
   * onSubmittedWorkDone() with *nothing newly submitted*, isolating that call's own fixed
   * latency in this browser, which dominates once real GPU work is only a few ms. Throws
   * if this device/browser doesn't support the 'timestamp-query' WebGPU feature. */
  async profile(
    inputs: Record<string, Float32Array>,
  ): Promise<ProfileReport & { realMilliseconds: number; baselineMilliseconds: number }> {
    const report = await this.withInputs(inputs, (params) => profileGraph(params));

    const baselineT0 = performance.now();
    await this.device.queue.onSubmittedWorkDone();
    const baselineMilliseconds = performance.now() - baselineT0;

    const t0 = performance.now();
    const outputs = await this.runToGpu(inputs);
    await this.device.queue.onSubmittedWorkDone();
    const realMilliseconds = performance.now() - t0;
    releaseOutputs(outputs);

    return { ...report, realMilliseconds, baselineMilliseconds };
  }

  /** Runs every branch directly against golden.json's own recorded reference input
   * (bypassing the time-based router entirely) and compares the actual computed values
   * against what a real eager-PyTorch run produced -- see engine/verify.ts. Throws if
   * this package wasn't exported with golden.json (i.e. the exporter had no example
   * inputs to capture it with). */
  async verify(): Promise<VerifyReport> {
    if (!this.golden) {
      throw new KumaManifestError(
        "This .iph package has no golden.json -- it was exported without example inputs to capture golden values with.",
      );
    }
    return verifyAgainstGolden(
      {
        device: this.device,
        kernels: this.kernels,
        pipelineCache: this.pipelineCache,
        weightBuffers: this.weightBuffers,
        snippets: this.snippets,
        snippetCache: this.snippetCache,
        constantCache: this.constantCache,
        bufferPool: this.bufferPool,
      },
      this.manifest,
      this.golden,
    );
  }

  /** Runs every branch once, directly against golden.json's own recorded reference
   * input (bypassing the router, same mechanism as verify()) -- purely to populate
   * pipelineCache/constantCache for every kernel/shape every branch could possibly
   * need, before the caller starts interacting with the model. Without this, the
   * *first* time a given branch (or a kernel-routing decision within it) actually
   * gets exercised during interactive use -- e.g. scrubbing across a segment boundary
   * into a branch never yet rendered -- pays for real shader compilation right then, a
   * genuine stutter landing at exactly the worst moment. Intended to be called once,
   * right after load(), before enabling any interactive controls. A no-op (not a
   * throw, unlike verify()) when this package has no golden.json -- nothing to warm up
   * ahead of time that a first real run() call wouldn't do anyway, since there's only
   * the one graph. */
  async warmUp(): Promise<void> {
    if (!this.golden) return;
    await warmUp(
      {
        device: this.device,
        kernels: this.kernels,
        pipelineCache: this.pipelineCache,
        weightBuffers: this.weightBuffers,
        snippets: this.snippets,
        snippetCache: this.snippetCache,
        constantCache: this.constantCache,
        bufferPool: this.bufferPool,
      },
      this.manifest,
      this.golden,
    );
  }
}
