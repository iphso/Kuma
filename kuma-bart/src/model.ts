import { loadIphPackage } from "./iph/loader.js";
import { readBuffers, requestComputeDevice, uploadBufferSlice, uploadFloat32 } from "webgpu-compute-kit";
import { recordGraph, releaseOutputs, type RunGraphOutput } from "./engine/scheduler.js";
import { verifyAgainstGolden, warmUp, type VerifyContext, type VerifyReport } from "./engine/verify.js";
import { profileGraph, type ProfileReport } from "./engine/profile.js";
import {
  createBufferPoolState,
  DEFAULT_BUFFER_POOL_DEPTH,
  DEFAULT_POOL_SIZE_THRESHOLD,
  type BufferPoolState,
  type ResolvedTensor,
} from "./engine/context.js";
import { TransientSet } from "./engine/transients.js";
import { numElements } from "./engine/shape.js";
import type { SnippetFn } from "./engine/snippets.js";
import { KumaManifestError, KumaShapeError } from "./errors.js";
import type { IOSpec, KumaManifest, PlaybackMeta } from "./types/manifest.js";
import type { GoldenData } from "./types/golden.js";
import type { GpuTensor } from "./types/tensor.js";

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
   * explicit `true`/`false` to skip the guess. With a caller-provided `device` there is
   * no adapter to guess from, so this defaults to false. */
  lowMemory?: boolean;
  /** Recommended number of run()/runRaw()/runToGpu()/encode() calls to keep in flight
   * (`maxInFlight`). Default 3 (DEFAULT_BUFFER_POOL_DEPTH), or 1 when lowMemory. Each
   * in-flight call holds its own scratch set of live intermediates -- the dominant
   * peak-memory term during playback -- so KumaPlayer caps its in-flight frames at this.
   * Also the ring depth of the pool verify()/warmUp()/profile() run on. */
  bufferPoolDepth?: number;
  /** Bytes at or above which a buffer skips verify()/warmUp()/profile()'s persistent ring
   * pool and is freed eagerly. Default 1 MB. Does not affect run()/encode(), whose scratch
   * sets recycle every intermediate by liveness instead. */
  poolSizeThreshold?: number;
}

export interface KumaLoadOptions extends KumaMemoryOptions {
  /** Run on this device instead of requesting a new one (shared-device mode): the model's
   * weights, pipelines and scratch live on it, so a caller's own buffers on the same
   * device can be passed straight to `encode()` and its outputs consumed by the caller's
   * kernels. The caller keeps ownership: `destroy()` frees the model's resources but never
   * destroys the device. `lowMemory` defaults to false (no adapter to guess from). */
  device?: GPUDevice;
}

export interface KumaEncodeOptions {
  /** CPU-side values of the inputs that `js_snippet` nodes read for branch routing
   * (switch models only, e.g. the normalized time of a multi-segment export). Branch
   * selection happens on the CPU while recording, before any GPU work, so a GPU buffer
   * alone can't drive it. Required iff the package has js_snippet nodes; ignored otherwise. */
  rawInputs?: ReadonlyMap<string, Float32Array>;
  /** Write outputs into these caller-owned tensors (by manifest output name; any subset)
   * instead of handing back scratch: one `copyBufferToBuffer` per output at the end of
   * the encode, honouring `byteOffset` (needs COPY_DST usage). The returned `outputs`
   * then reference the caller's tensors and stay valid after `release()`. */
  outputs?: ReadonlyMap<string, GpuTensor>;
}

export interface EncodedRun {
  /** The manifest's declared outputs, in manifest output order. Unless redirected via
   * `KumaEncodeOptions.outputs`, each is scratch from this encode (or a weight/input a
   * passthrough aliased) and is valid only until `release()` -- consume it in commands
   * recorded before then (in the same encoder, or one submitted after it). */
  outputs: GpuTensor[];
  /** The same outputs keyed by manifest output name. */
  outputsByName: ReadonlyMap<string, GpuTensor>;
  /** Hands this encode's scratch back to the model for reuse. Call it once the encoder
   * has been submitted (or discarded) and every consumer of the scratch `outputs` has
   * been recorded; nothing waits for GPU completion -- buffer reuse by later submissions
   * is queue-ordered. Idempotent. Never submits. */
  release(): void;
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

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

function fmtShape(shape: readonly number[] | undefined): string {
  return shape ? `[${shape.join(", ")}]` : "unknown";
}

/** A loaded `.iph` package, ready to run via WebGPU. */
export class KumaModel {
  /** Scratch sets not currently held by an unreleased encode (see TransientSet). */
  private readonly idleSets: TransientSet[] = [];
  private readonly busySets = new Set<TransientSet>();
  /** Declared (or placeholder-derived) shape per input name; undefined when the
   * package declares none (e.g. an ONNX-style export's scalar `norm_t`). */
  private readonly inputShapes: ReadonlyMap<string, number[] | undefined>;
  private readonly hasSnippets: boolean;

  private constructor(
    private readonly device: GPUDevice,
    private readonly ownsDevice: boolean,
    private readonly manifest: KumaManifest,
    private readonly kernels: ReadonlyMap<string, string>,
    private readonly snippets: ReadonlyMap<string, string>,
    private readonly weightBuffers: ReadonlyMap<string, ResolvedTensor>,
    private readonly pipelineCache: Map<string, GPUComputePipeline>,
    private readonly snippetCache: Map<string, SnippetFn>,
    private readonly golden: GoldenData | undefined,
    private readonly constantCache: Map<string, GPUBuffer>,
    private readonly bufferPool: BufferPoolState,
  ) {
    const shapes = new Map<string, number[] | undefined>();
    for (const spec of manifest.inputs) {
      const placeholder = manifest.graph.nodes.find((n) => n.op === "placeholder" && n.name === spec.name);
      shapes.set(spec.name, spec.shape ?? placeholder?.meta.shape);
    }
    this.inputShapes = shapes;
    this.hasSnippets = manifest.graph.nodes.some((n) => n.op === "js_snippet");
  }

  static async load(
    source: ArrayBuffer | Response | string,
    options: KumaLoadOptions = {},
  ): Promise<KumaModel> {
    const { manifest, weights, kernels, snippets, golden } = await loadIphPackage(source);
    let device = options.device;
    let isLikelyLowMemory = false;
    if (!device) ({ device, isLikelyLowMemory } = await requestComputeDevice());

    const weightBuffers = new Map<string, ResolvedTensor>();
    for (const w of manifest.weights) {
      const buffer = uploadBufferSlice(device, weights, w.byte_offset, w.byte_length);
      weightBuffers.set(w.name, { buffer, shape: w.shape });
    }

    const { depth, sizeThreshold } = resolveMemoryOptions(options, isLikelyLowMemory);
    return new KumaModel(
      device,
      options.device === undefined,
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

  /** The GPU device this model's buffers/pipelines live on -- the one passed to
   * `load({ device })`, or the model's own otherwise. WebGPU resources are
   * device-scoped: every `GpuTensor` given to `encode()` must be a buffer of this device,
   * and a `runRaw()`/`runToGpu()`/`encode()` output can only be consumed by pipelines
   * created on it. Also where to call `queue.onSubmittedWorkDone()` for backpressure. */
  get gpuDevice(): GPUDevice {
    return this.device;
  }

  /** How many run()/runRaw()/runToGpu()/encode() calls a caller should keep in flight at
   * once (`bufferPoolDepth`, default 3, or 1 when lowMemory). Not a hard limit: each
   * unreleased call holds its own scratch set of live intermediates, allocated on demand,
   * so exceeding this costs memory rather than blocking. KumaPlayer reads this so a model
   * loaded with `{ lowMemory: true }` actually serializes to one frame at a time. */
  get maxInFlight(): number {
    return this.bufferPool.depth;
  }

  /** Bytes currently held by run()/encode() scratch sets, idle or in use -- roughly
   * (peak live activation bytes) x (most calls ever unreleased at once). */
  get scratchBytes(): number {
    let bytes = 0;
    for (const set of this.idleSets) bytes += set.bytes;
    for (const set of this.busySets) bytes += set.bytes;
    return bytes;
  }

  /** Encodes recorded but not yet released. */
  get activeEncodes(): number {
    return this.busySets.size;
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

  /** Releases everything this model allocated. For a model that requested its own
   * device (no `device` in load options) that is one `GPUDevice.destroy()`, which frees
   * weights, scratch, pipelines and all in one go -- call it instead of just dropping the
   * reference, or the whole buffer set lingers until JS GC collects the device graph. For
   * a shared device the device is left alone and the model's weight, constant, pool and
   * idle scratch buffers are destroyed individually (release every encode first; a set
   * still held by an unreleased encode is not touched). Safe to call more than once;
   * every other method on this instance is unusable afterward. */
  destroy(): void {
    if (this.ownsDevice) {
      this.device.destroy();
      return;
    }
    for (const t of this.weightBuffers.values()) {
      t.buffer.destroy();
      t.imag?.destroy();
    }
    for (const buffer of this.constantCache.values()) buffer.destroy();
    this.constantCache.clear();
    for (const slots of this.bufferPool.pools.values()) for (const buffer of slots) buffer?.destroy();
    this.bufferPool.pools.clear();
    for (const set of this.idleSets) set.destroy();
    this.idleSets.length = 0;
    this.pipelineCache.clear();
  }

  /** Checks a caller's tensor against what `role` "`name`" needs: f32, the expected
   * shape (when the package declares one), an aligned offset, enough bytes, and the
   * usage the way it will be used requires. */
  private checkTensor(
    role: "Input" | "Output",
    name: string,
    tensor: GpuTensor,
    expectedShape: readonly number[] | undefined,
    usage: { flag: number; name: string },
  ): { shape: number[]; byteOffset: number; byteLength: number } {
    const where = `${role} "${name}"`;
    if (tensor.dtype !== "f32") {
      throw new KumaShapeError(`${where}: dtype "${String(tensor.dtype)}" is not supported -- kuma-bart tensors are f32.`);
    }
    if (expectedShape && !sameShape(tensor.shape, expectedShape)) {
      throw new KumaShapeError(`${where} expected shape ${fmtShape(expectedShape)}, got ${fmtShape(tensor.shape)}.`);
    }
    const byteOffset = tensor.byteOffset ?? 0;
    if (!Number.isInteger(byteOffset) || byteOffset < 0 || byteOffset % 4 !== 0) {
      throw new KumaShapeError(`${where}: byteOffset ${byteOffset} must be a non-negative multiple of 4.`);
    }
    const byteLength = numElements(tensor.shape) * 4;
    if (tensor.buffer.size < byteOffset + byteLength) {
      throw new KumaShapeError(
        `${where}: buffer is ${tensor.buffer.size} bytes, but shape ${fmtShape(tensor.shape)} at byteOffset ${byteOffset} needs ${byteOffset + byteLength}.`,
      );
    }
    if ((tensor.buffer.usage & usage.flag) === 0) {
      throw new KumaShapeError(`${where}: buffer was created without ${usage.name} usage, which binding it this way needs.`);
    }
    return { shape: [...tensor.shape], byteOffset, byteLength };
  }

  /** Records this model's whole graph into `encoder` -- one compute dispatch per node --
   * reading `inputs` (GPU-resident, see GpuTensor for the layout) and returns the
   * GPU-resident outputs, synchronously, without submitting or waiting. The caller
   * submits the encoder whenever it likes, typically after recording its own kernels
   * that consume the outputs and further `encode()` calls (any number of them: each
   * takes its own scratch set, so nothing is shared between encodes in flight). Then
   * `release()` each run once submitted.
   *
   * Inputs must match the manifest's names and declared shapes (KumaShapeError on a
   * mismatch, KumaManifestError when one is missing), be f32 and live on `gpuDevice`
   * (not checkable here -- see GpuTensor). Errors leave nothing recorded except an
   * input copy that may already have been encoded. */
  encode(inputs: ReadonlyMap<string, GpuTensor>, encoder: GPUCommandEncoder, opts: KumaEncodeOptions = {}): EncodedRun {
    // Everything that can be checked without touching scratch, first.
    const checked: Array<{ spec: IOSpec; tensor: GpuTensor; shape: number[]; byteOffset: number; byteLength: number }> = [];
    for (const spec of this.manifest.inputs) {
      const tensor = inputs.get(spec.name);
      if (!tensor) {
        throw new KumaManifestError(
          `Missing input "${spec.name}" — expected one of: ${this.manifest.inputs.map((s) => s.name).join(", ")}`,
        );
      }
      const offset = tensor.byteOffset ?? 0;
      const usage = offset === 0 ? { flag: GPUBufferUsage.STORAGE, name: "STORAGE" } : { flag: GPUBufferUsage.COPY_SRC, name: "COPY_SRC" };
      checked.push({ spec, tensor, ...this.checkTensor("Input", spec.name, tensor, this.inputShapes.get(spec.name), usage) });
    }
    if (this.hasSnippets && !opts.rawInputs) {
      throw new KumaManifestError(
        "This package routes between branches with js_snippet nodes, which read input values on the CPU: pass the routing inputs' values as encode(..., { rawInputs }).",
      );
    }

    const set = this.idleSets.pop() ?? new TransientSet(this.device);
    this.busySets.add(set);
    const giveBack = (): void => {
      set.reset();
      this.busySets.delete(set);
      this.idleSets.push(set);
    };

    let outputs: GpuTensor[];
    try {
      const inputBuffers = new Map<string, ResolvedTensor>();
      for (const { spec, tensor, shape, byteOffset, byteLength } of checked) {
        let buffer = tensor.buffer;
        if (byteOffset !== 0) {
          // Kernels bind whole buffers, so an offset input is staged by one GPU copy.
          buffer = set.acquire(`input:${spec.name}`, byteLength);
          encoder.copyBufferToBuffer(tensor.buffer, byteOffset, buffer, 0, byteLength);
        }
        inputBuffers.set(spec.name, { buffer, shape });
      }

      const recorded = recordGraph(
        {
          device: this.device,
          manifest: this.manifest,
          kernels: this.kernels,
          pipelineCache: this.pipelineCache,
          weightBuffers: this.weightBuffers,
          inputBuffers,
          rawInputs: opts.rawInputs,
          snippets: this.snippets,
          snippetCache: this.snippetCache,
          constantCache: this.constantCache,
        },
        { encoder, transients: set },
      );

      outputs = this.manifest.outputs.map((spec, i) => {
        const tensor = recorded.outputTensors[i]!;
        const byteLength = numElements(tensor.shape) * 4;
        const target = opts.outputs?.get(spec.name);
        if (!target) return { buffer: tensor.buffer, shape: [...tensor.shape], dtype: "f32", byteOffset: 0 };
        const dst = this.checkTensor("Output", spec.name, target, tensor.shape, { flag: GPUBufferUsage.COPY_DST, name: "COPY_DST" });
        encoder.copyBufferToBuffer(tensor.buffer, 0, target.buffer, dst.byteOffset, byteLength);
        return target;
      });
    } catch (err) {
      giveBack();
      throw err;
    }

    const outputsByName = new Map<string, GpuTensor>();
    this.manifest.outputs.forEach((spec, i) => outputsByName.set(spec.name, outputs[i]!));
    let released = false;
    return {
      outputs,
      outputsByName,
      release: () => {
        if (released) return;
        released = true;
        giveBack();
      },
    };
  }

  /** Uploads CPU inputs (validated against the manifest), encodes and submits the graph
   * in one command buffer, optionally reads the outputs back, and hands them out as
   * RunGraphOutputs whose `release()` (all of them) returns the scratch to the model
   * and frees the input uploads. */
  private async runEncoded(inputs: Record<string, Float32Array>, readback: boolean): Promise<RunGraphOutput[]> {
    const gpuInputs = new Map<string, GpuTensor>();
    const rawInputs = new Map<string, Float32Array>();
    const uploaded: GPUBuffer[] = [];
    const destroyUploads = (): void => {
      for (const buffer of uploaded) buffer.destroy();
      uploaded.length = 0;
    };
    for (const spec of this.manifest.inputs) {
      const data = inputs[spec.name];
      if (!data) {
        destroyUploads();
        throw new KumaManifestError(
          `Missing input "${spec.name}" — expected one of: ${this.manifest.inputs.map((s) => s.name).join(", ")}`,
        );
      }
      const shape = this.inputShapes.get(spec.name) ?? [data.length];
      const expected = numElements(shape);
      if (data.length !== expected) {
        destroyUploads();
        throw new KumaManifestError(
          `Input "${spec.name}" expected ${expected} elements (shape ${JSON.stringify(shape)}), got ${data.length}.`,
        );
      }
      rawInputs.set(spec.name, data);
      const buffer = uploadFloat32(this.device, data);
      uploaded.push(buffer);
      gpuInputs.set(spec.name, { buffer, shape, dtype: "f32" });
    }

    const encoder = this.device.createCommandEncoder();
    let run: EncodedRun;
    try {
      run = this.encode(gpuInputs, encoder, { rawInputs });
    } catch (err) {
      destroyUploads();
      throw err;
    }
    this.device.queue.submit([encoder.finish()]);

    const datas = readback
      ? await readBuffers(
          this.device,
          run.outputs.map((o) => ({ buffer: o.buffer, byteLength: numElements(o.shape) * 4 })),
        )
      : [];

    // The uploads are freed with the scratch: an output may alias one (a passthrough-only
    // graph), and destroy() while submitted work still references a buffer is fine.
    let remaining = run.outputs.length;
    const releaseAll = (): void => {
      run.release();
      destroyUploads();
    };
    if (remaining === 0) releaseAll();
    return run.outputs.map((o, i) => {
      let done = false;
      return {
        name: this.manifest.outputs[i]!.name,
        shape: [...o.shape],
        data: datas[i] ?? new Float32Array(0),
        buffer: o.buffer,
        release: () => {
          if (done) return;
          done = true;
          if (--remaining === 0) releaseAll();
        },
      };
    });
  }

  /** Like `run`, but returns the full RunGraphOutput[] (shape + the GPU buffer the
   * data was read back from, not just the data) -- for a caller that wants to render
   * or otherwise consume a result on-GPU. The caller owns the result: call each
   * output's `release()` (or `releaseOutputs`); once every output of the call is
   * released its scratch goes back to the model for the next call, so `buffer` is
   * valid until then. */
  async runRaw(inputs: Record<string, Float32Array>): Promise<RunGraphOutput[]> {
    return this.runEncoded(inputs, true);
  }

  /** Like `runRaw`, but skips the GPU->CPU readback for the model's own outputs
   * entirely -- `data` on the result is an empty Float32Array(0); use `buffer` (e.g.
   * to render directly, which the GPU orders after this call's own submission with no
   * CPU-side wait needed), then `release()` it. The readback measured 60-70ms of a
   * ~65-75ms frame for a 720x1280x3 model, far more than encoding every dispatch
   * (~2ms), so an interactive path that only needs pixels on screen should use this.
   * For real GPU compute time, see profile()'s realMilliseconds. */
  async runToGpu(inputs: Record<string, Float32Array>): Promise<RunGraphOutput[]> {
    return this.runEncoded(inputs, false);
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

  /** The ring-pool run parameters verify()/warmUp()/profile() share (see engine/verify.ts). */
  private ringParams(): VerifyContext {
    return {
      device: this.device,
      kernels: this.kernels,
      pipelineCache: this.pipelineCache,
      weightBuffers: this.weightBuffers,
      snippets: this.snippets,
      snippetCache: this.snippetCache,
      constantCache: this.constantCache,
      bufferPool: this.bufferPool,
    };
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
    const inputBuffers = new Map<string, ResolvedTensor>();
    const rawInputs = new Map<string, Float32Array>();
    for (const spec of this.manifest.inputs) {
      const data = inputs[spec.name];
      if (!data) {
        throw new KumaManifestError(
          `Missing input "${spec.name}" — expected one of: ${this.manifest.inputs.map((s) => s.name).join(", ")}`,
        );
      }
      rawInputs.set(spec.name, data);
      inputBuffers.set(spec.name, { buffer: uploadFloat32(this.device, data), shape: this.inputShapes.get(spec.name) ?? [data.length] });
    }
    let report: ProfileReport;
    try {
      report = await profileGraph({ ...this.ringParams(), manifest: this.manifest, inputBuffers, rawInputs });
    } finally {
      for (const t of inputBuffers.values()) t.buffer.destroy();
    }

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
    return verifyAgainstGolden(this.ringParams(), this.manifest, this.golden);
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
    await warmUp(this.ringParams(), this.manifest, this.golden);
  }
}
