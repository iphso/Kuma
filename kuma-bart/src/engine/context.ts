import { isNodeRef, type ArgValue, type GraphNode } from "../types/manifest.js";
import { createStorageBuffer, createUniformBuffer, uploadFloat32 } from "../gpu/buffers.js";
import { packParams, packTypedParams, type TypedParamField } from "../gpu/params.js";
import { KumaShapeError, KumaManifestError } from "../errors.js";
import { numElements } from "./shape.js";
import { computeDispatchGrid, KERNEL_WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIMENSION } from "./dispatch.js";
import {
  BufferPool,
  PipelineCache,
  DEFAULT_BUFFER_POOL_DEPTH,
  DEFAULT_POOL_SIZE_THRESHOLD,
  createBufferPoolState,
  acquireGenerationSlot,
  releaseGenerationSlot,
  type BufferPoolState,
  type BufferPoolOptions,
} from "webgpu-compute-kit";

// The buffer-pool ring/generation-pipelining mechanism (BufferPoolState,
// createBufferPoolState, acquireGenerationSlot, releaseGenerationSlot) and the WGSL
// pipeline cache/dispatch mechanics (getPipeline/bindAndDispatch below) used to be
// defined here directly. Neither one actually depends on anything graph-specific -- they
// were extracted into `webgpu-compute-kit` so a non-ML WebGPU project doesn't have to
// rediscover the same GC-pause and GPU-completion-latency fixes. This file now wraps that
// package's generic pieces with kuma's own graph-specific keying (node name + call index)
// and cache-sharing conventions (a single Map<string, GPUComputePipeline> threaded through
// model.ts/scheduler.ts/profile.ts/verify.ts).

export { type BufferPoolState, type BufferPoolOptions, createBufferPoolState, acquireGenerationSlot, releaseGenerationSlot };

/** Re-exported from webgpu-compute-kit's DEFAULT_BUFFER_POOL_DEPTH -- see that package's
 * pool.ts for the full rationale (GC-pause fix, ring depth vs. VRAM trade-off). */
export const BUFFER_POOL_DEPTH = DEFAULT_BUFFER_POOL_DEPTH;
export { DEFAULT_BUFFER_POOL_DEPTH, DEFAULT_POOL_SIZE_THRESHOLD };

export interface ResolvedTensor {
  buffer: GPUBuffer;
  shape: number[];
  /** Set when this tensor logically represents a complex64 value — `buffer` holds the
   * real part, `imag` the imaginary part, paired up purely on the JS side by
   * aten.complex.default (see ops/complex.ts). No kernel ever sees an interleaved
   * complex buffer; real/imag are always two separate same-shape real GPUBuffers. */
  imag?: GPUBuffer;
}

/**
 * Per-node execution context handed to an op handler. Wraps the GPU mechanics
 * (pipeline caching, bind groups, dispatch, pooled buffer allocation) so handlers only
 * deal with tensors and shapes. Every kernel binds its buffers in the exact order
 * declared in its WGSL source (storage buffers first, uniform Params last) —
 * `dispatchKernel` relies on callers passing buffers in that same order.
 */
export class OpContext {
  // Counts this node's own uniform()/uniformTyped() calls so each one gets a distinct
  // cache slot below -- some ops (fft.ts's matmulBasis, called repeatedly from
  // complexIfftAlongAxis) build more than one params buffer per node dispatch, all
  // through this same ctx instance, and node name alone would collapse them into one
  // shared (and not necessarily identical) cached buffer.
  private uniformCallIndex = 0;
  // Same rationale as uniformCallIndex, for createBuffer below -- a node can call it
  // more than once (e.g. fft.ts's multi-step intermediates), and each call needs its
  // own pool ring, not a shared one.
  private createBufferCallIndex = 0;

  private readonly pool: BufferPool;
  private readonly pipelines: PipelineCache;

  constructor(
    public readonly device: GPUDevice,
    private readonly kernels: ReadonlyMap<string, string>,
    pipelineCache: Map<string, GPUComputePipeline>,
    private readonly pass: GPUComputePassEncoder,
    private readonly resolved: Map<string, ResolvedTensor>,
    public readonly node: GraphNode,
    private readonly constantCache: Map<string, GPUBuffer>,
    private readonly bufferPool: BufferPoolState,
    // Every fresh (non-pooled) buffer createBuffer hands out is recorded here so the
    // caller (scheduler/profile) can free it after the run. Without this, an op's
    // *intra-op* intermediates -- buffers it allocates but never setOutput()s into
    // `resolved` (e.g. floor_divide's `divided`, fft's multi-step temporaries) -- are
    // invisible to both the early-free pass and the end-of-run cleanup (both scan only
    // `resolved`), so they leak. Harmless in the default pool regime (those buffers come
    // from the pool and are reused), but at sizeThreshold 0 (low-memory) every createBuffer
    // bypasses the pool and allocates fresh, turning that blind spot into a per-frame leak.
    private readonly transientBuffers: Set<GPUBuffer>,
  ) {
    this.pool = new BufferPool(device, bufferPool);
    // `pipelineCache` stays a plain Map (threaded through model.ts/scheduler.ts/
    // profile.ts/verify.ts unchanged) -- PipelineCache wraps it rather than owning its
    // own, so every OpContext across every node shares the one cache a model was loaded
    // with, exactly as before this file started delegating to webgpu-compute-kit.
    this.pipelines = new PipelineCache(device, pipelineCache);
  }

  resolve(arg: ArgValue): ResolvedTensor {
    if (!isNodeRef(arg)) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") expected a tensor reference, got ${JSON.stringify(arg)}.`,
      );
    }
    const tensor = this.resolved.get(arg.node_ref);
    if (!tensor) {
      throw new KumaShapeError(
        `Node "${this.node.name}" references "${arg.node_ref}" before it was computed — manifest graph is not topologically ordered.`,
      );
    }
    return tensor;
  }

  /** Returns this call's slot in a `bufferPool.depth`-deep ring buffer, keyed by this
   * node + which createBuffer call this is within it -- sized once (lazily, on first
   * use) to `shape`'s byte length, safe to reuse forever after since the same node
   * always produces the same shape on every call (Kuma is a static-shape compiler).
   * Reuse-timing safety across calls is the scheduler's job (acquireGenerationSlot,
   * called once per runGraph/profileGraph call before any createBuffer call can
   * happen) -- see webgpu-compute-kit's BufferPool for the ring-rotation mechanics this
   * delegates to; the node-name/call-index key is kuma's own graph-specific scheme for
   * choosing what "this resource" means.
   *
   * Buffers >= `bufferPool.sizeThreshold` bypass the pool and are allocated fresh -- the
   * caller (scheduler.ts's cleanup pass) will destroy them after submit, letting the
   * driver reclaim VRAM without going through JS GC. The call-index still increments
   * so subsequent small-buffer calls within the same node get distinct pool keys. */
  createBuffer(shape: readonly number[]): GPUBuffer {
    const byteLength = numElements(shape) * 4;
    const callIndex = this.createBufferCallIndex++;
    const key = `${this.node.name}::${callIndex}`;
    const buffer = this.pool.acquire(key, byteLength);
    if (byteLength >= this.bufferPool.sizeThreshold) {
      this.transientBuffers.add(buffer);
    }
    return buffer;
  }

  /** A zero-filled buffer of `byteLength` bytes — for bias-less ops (e.g. conv2d/linear with no bias arg). */
  zeros(byteLength: number): GPUBuffer {
    return createStorageBuffer(this.device, byteLength);
  }

  /** Like `zeros`, but cached by `key` — for a zero buffer whose size is itself a fixed
   * (shape-derived) constant reused across calls (e.g. fft.ts's zero bias for its DFT
   * matmuls), so it's allocated once instead of once per inference call. Never written
   * to by any kernel (every binding that uses one is read-only), so sharing one instance
   * across every call site that asks for the same size is safe. */
  getOrCreateZeroBuffer(key: string, byteLength: number): GPUBuffer {
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = this.zeros(byteLength);
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Upload a JS-computed constant (e.g. a DFT basis matrix — see engine/dft.ts) as a
   * storage buffer, ready to bind as a kernel input like any other tensor. */
  uploadConstant(data: Float32Array): GPUBuffer {
    return uploadFloat32(this.device, data);
  }

  /** Like `uploadConstant`, but only computes + uploads once per `key`, ever — reused
   * across every node and every subsequent inference call on this model. For data
   * that's a pure function of a fixed shape, never of the actual input (e.g. a DFT
   * basis matrix keyed by transform length): recomputing and re-uploading it on every
   * single frame is pure waste, not extra correctness. `key` is caller-namespaced (e.g.
   * "complexIfft:cos:180") since this cache is shared across every op, not per-kernel. */
  getOrUploadConstant(key: string, compute: () => Float32Array): GPUBuffer {
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = this.uploadConstant(compute());
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Pack u32 fields (WGSL struct order) into a ready-to-bind uniform buffer. Cached
   * by value: most fields come from graph structure, but ONNX exports can have concrete
   * batch-dependent shapes that differ between playback and golden verification for the
   * same node. The values still are not data-dependent, so caching identical param
   * payloads remains useful; the payload itself must be part of the key. */
  uniform(fields: readonly number[]): GPUBuffer {
    const key = `params:${this.node.name}:${this.uniformCallIndex++}:${fields.join(",")}`;
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = createUniformBuffer(this.device, packParams(fields));
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Like `uniform`, for Params structs that mix u32 and f32 fields (e.g. clamp,
   * group_norm/layer_norm's eps). Same value-keyed caching rationale as `uniform`. */
  uniformTyped(fields: readonly TypedParamField[]): GPUBuffer {
    const key = `params:${this.node.name}:${this.uniformCallIndex++}:${JSON.stringify(fields)}`;
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = createUniformBuffer(this.device, packTypedParams(fields));
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Register this node's computed result so later nodes can resolve it by name. */
  setOutput(buffer: GPUBuffer, shape: readonly number[], imag?: GPUBuffer): void {
    this.resolved.set(this.node.name, { buffer, shape: [...shape], imag });
  }

  /** Register an already-built ResolvedTensor as this node's result (e.g. a free
   * passthrough that reuses an existing buffer, or a complex pairing). */
  setOutputTensor(tensor: ResolvedTensor): void {
    this.resolved.set(this.node.name, tensor);
  }

  /** For multi-output nodes (e.g. aten.chunk.default): register one of several results
   * under a synthetic per-index key, later picked up by a getitem(thisNode, i) node. */
  setIndexedOutput(index: number, tensor: ResolvedTensor): void {
    this.resolved.set(`${this.node.name}::${index}`, tensor);
  }

  /** Resolve a multi-output node's i'th result, registered via `setIndexedOutput`. */
  resolveIndexed(nodeName: string, index: number): ResolvedTensor {
    const tensor = this.resolved.get(`${nodeName}::${index}`);
    if (!tensor) {
      throw new KumaShapeError(
        `Node "${this.node.name}" (getitem) references "${nodeName}"[${index}], which was never registered as a multi-output result.`,
      );
    }
    return tensor;
  }

  /** Record one compute dispatch for `kernelName` against the shared command encoder's
   * compute pass. `buffers` must be in the exact @binding(0..n) order of the kernel's
   * WGSL source. `dispatchElements` is the total number of `out[]` elements the kernel
   * computes — every kernel using this entry point uses `@workgroup_size(64)` and folds
   * `gid.y` into its linear index (see engine/dispatch.ts), so dispatches needing more
   * than 65535 workgroups spread across a 2D grid instead of failing. Kernels with a
   * genuinely 2D notion of work (e.g. a tiled matmul, where workgroup_id.(x,y) means
   * "which output tile" rather than folding into one linear index) should use
   * `dispatchKernel2D` instead — its grid shape carries real meaning to the shader. */
  dispatchKernel(kernelName: string, buffers: GPUBuffer[], dispatchElements: number): void {
    const totalWorkgroups = Math.max(1, Math.ceil(dispatchElements / KERNEL_WORKGROUP_SIZE));
    const grid = computeDispatchGrid(totalWorkgroups);
    if (!grid) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") needs ${totalWorkgroups} workgroups, exceeding ` +
          `even a ${MAX_WORKGROUPS_PER_DIMENSION}x${MAX_WORKGROUPS_PER_DIMENSION} 2D dispatch grid's capacity.`,
      );
    }
    this.pipelines.dispatch(this.pass, this.getPipeline(kernelName), buffers, grid.x, grid.y, 1);
  }

  /** Like `dispatchKernel`, but for a kernel whose `@builtin(workgroup_id)` is read
   * directly as grid coordinates (2D or 3D) rather than folded into a linear element
   * index -- `gridX`/`gridY`/`gridZ` are workgroup *counts* along each axis (e.g.
   * ceil(N/TILE), ceil(M/TILE) for a tiled matmul; or ceil(W/TILE), ceil(H/TILE),
   * batch*channels for a tiled conv), not element counts. `gridZ` defaults to 1 for
   * kernels that only need a 2D grid. */
  dispatchKernelGrid(kernelName: string, buffers: GPUBuffer[], gridX: number, gridY: number, gridZ: number = 1): void {
    this.pipelines.dispatch(this.pass, this.getPipeline(kernelName), buffers, gridX, gridY, gridZ);
  }

  private getPipeline(kernelName: string): GPUComputePipeline {
    const source = this.kernels.get(kernelName);
    if (!source) {
      throw new KumaManifestError(`.iph package is missing kernel source "kernels/${kernelName}".`);
    }
    return this.pipelines.getPipeline(kernelName, source);
  }
}

export type OpHandler = (ctx: OpContext) => void;
