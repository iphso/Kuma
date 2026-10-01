import { isNodeRef, type ArgValue, type GraphNode } from "../types/manifest.js";
import { packParams, packTypedParams, type TypedParamField } from "../gpu/params.js";
import { KumaShapeError, KumaManifestError } from "../errors.js";
import { numElements } from "./shape.js";
import {
  BufferPool,
  PipelineCache,
  DEFAULT_BUFFER_POOL_DEPTH,
  DEFAULT_POOL_SIZE_THRESHOLD,
  MAX_WORKGROUPS_PER_DIMENSION,
  createBufferPoolState,
  acquireGenerationSlot,
  releaseGenerationSlot,
  createStorageBuffer,
  createUniformBuffer,
  uploadFloat32,
  type BufferPoolState,
  type BufferPoolOptions,
} from "webgpu-compute-kit";

// The pool ring / generation pipelining and the pipeline cache are webgpu-compute-kit's;
// this file adds kuma's graph-specific keying (node name + call index) on top.
export { type BufferPoolState, type BufferPoolOptions, createBufferPoolState, acquireGenerationSlot, releaseGenerationSlot };

/** webgpu-compute-kit's DEFAULT_BUFFER_POOL_DEPTH, under kuma-bart's public name. */
export const BUFFER_POOL_DEPTH = DEFAULT_BUFFER_POOL_DEPTH;
export { DEFAULT_BUFFER_POOL_DEPTH, DEFAULT_POOL_SIZE_THRESHOLD };

/** Every kuma kernel dispatched through `dispatchKernel` uses `@workgroup_size(64)` and
 * folds its linear index as `gid.x + gid.y * (num_workgroups.x * 64)` (see kuma's
 * kernels/__init__.py). A kuma kernel-authoring convention, not a WebGPU constraint. */
export const KERNEL_WORKGROUP_SIZE = 64;

export interface ResolvedTensor {
  buffer: GPUBuffer;
  shape: number[];
  /** Set when this tensor logically represents a complex64 value — `buffer` holds the
   * real part, `imag` the imaginary part, paired up purely on the JS side by
   * aten.complex.default (see ops/complex.ts). No kernel ever sees an interleaved
   * complex buffer; real/imag are always two separate same-shape real GPUBuffers. */
  imag?: GPUBuffer;
}

// The kit wrappers are stateless views over a model's own pipeline Map / pool state, so
// they are memoized per underlying object: built once per model, not once per node.
const pipelineCaches = new WeakMap<Map<string, GPUComputePipeline>, PipelineCache>();
const bufferPools = new WeakMap<BufferPoolState, BufferPool>();

function pipelineCacheFor(device: GPUDevice, cache: Map<string, GPUComputePipeline>): PipelineCache {
  let wrapper = pipelineCaches.get(cache);
  if (!wrapper) {
    wrapper = new PipelineCache(device, cache);
    pipelineCaches.set(cache, wrapper);
  }
  return wrapper;
}

function bufferPoolFor(device: GPUDevice, state: BufferPoolState): BufferPool {
  let wrapper = bufferPools.get(state);
  if (!wrapper) {
    wrapper = new BufferPool(device, state);
    bufferPools.set(state, wrapper);
  }
  return wrapper;
}

/** Everything one graph-encoding call shares across all of its nodes. */
export class EncodeEnv {
  readonly pipelines: PipelineCache;
  readonly pool: BufferPool;
  /** Resolved results by node name (plus "<node>::<i>" for multi-output nodes). */
  readonly resolved: Map<string, ResolvedTensor>;

  constructor(
    readonly device: GPUDevice,
    readonly kernels: ReadonlyMap<string, string>,
    pipelineCache: Map<string, GPUComputePipeline>,
    readonly bufferPool: BufferPoolState,
    readonly constantCache: Map<string, GPUBuffer>,
    /** Receives every fresh (non-pooled) buffer `createBuffer` hands out, including
     * op-internal intermediates that never reach `resolved` -- the caller owns freeing
     * them. */
    readonly ownedBuffers: Set<GPUBuffer>,
    seeded: ReadonlyMap<string, ResolvedTensor> = new Map(),
  ) {
    this.pipelines = pipelineCacheFor(device, pipelineCache);
    this.pool = bufferPoolFor(device, bufferPool);
    this.resolved = new Map(seeded);
  }
}

/**
 * Per-node execution context handed to an op handler. Wraps the GPU mechanics
 * (pipeline caching, bind groups, dispatch, pooled buffer allocation) so handlers only
 * deal with tensors and shapes. Every kernel binds its buffers in the exact order
 * declared in its WGSL source (storage buffers first, uniform Params last) —
 * `dispatchKernel` relies on callers passing buffers in that same order.
 */
export class OpContext {
  // A node can build several params buffers and several output buffers (e.g. fft.ts's
  // multi-step intermediates); each call gets its own cache slot / pool ring.
  private uniformCallIndex = 0;
  private createBufferCallIndex = 0;

  constructor(
    private readonly env: EncodeEnv,
    private readonly pass: GPUComputePassEncoder,
    public readonly node: GraphNode,
  ) {}

  get device(): GPUDevice {
    return this.env.device;
  }

  /** Whether the loaded package ships `kernels/<kernelName>` -- for handlers that prefer
   * a newer kernel but must still run packages exported before it existed. */
  hasKernel(kernelName: string): boolean {
    return this.env.kernels.has(kernelName);
  }

  resolve(arg: ArgValue): ResolvedTensor {
    if (!isNodeRef(arg)) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") expected a tensor reference, got ${JSON.stringify(arg)}.`,
      );
    }
    const tensor = this.env.resolved.get(arg.node_ref);
    if (!tensor) {
      throw new KumaShapeError(
        `Node "${this.node.name}" references "${arg.node_ref}" before it was computed — manifest graph is not topologically ordered.`,
      );
    }
    return tensor;
  }

  /** A `shape`-sized output buffer from this node's pool ring, keyed by node name + which
   * createBuffer call this is within the node. Safe to reuse across calls because kuma is
   * a static-shape compiler (same key, same size); reuse timing is guarded by the
   * scheduler's acquireGenerationSlot. Buffers >= the pool's sizeThreshold bypass the
   * pool, are allocated fresh, and are recorded in `ownedBuffers` for the caller to
   * free. */
  createBuffer(shape: readonly number[]): GPUBuffer {
    const byteLength = numElements(shape) * 4;
    const key = `${this.node.name}::${this.createBufferCallIndex++}`;
    const buffer = this.env.pool.acquire(key, byteLength);
    if (byteLength >= this.env.bufferPool.sizeThreshold) {
      this.env.ownedBuffers.add(buffer);
    }
    return buffer;
  }

  /** A zero-filled read-only buffer of `byteLength` bytes, cached by `key` for the model's
   * lifetime (e.g. the zero bias of a bias-less conv/linear). No kernel ever writes to a
   * binding that receives one, so every caller asking for the same key can share it. */
  getOrCreateZeroBuffer(key: string, byteLength: number): GPUBuffer {
    let buffer = this.env.constantCache.get(key);
    if (!buffer) {
      buffer = createStorageBuffer(this.env.device, byteLength);
      this.env.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Computes and uploads `compute()` once per `key`, for the model's lifetime. For data
   * that is a pure function of fixed shapes/literals, never of input data (e.g. a DFT
   * basis keyed by transform length). `key` is caller-namespaced and must encode every
   * input `compute` depends on (including the element count), since this cache is shared
   * across every op and every node. */
  getOrUploadConstant(key: string, compute: () => Float32Array): GPUBuffer {
    let buffer = this.env.constantCache.get(key);
    if (!buffer) {
      buffer = uploadFloat32(this.env.device, compute());
      this.env.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Pack u32 fields (WGSL struct order) into a ready-to-bind uniform buffer, cached by
   * node + call index + value. The payload is part of the key because ONNX-style exports
   * can give the same node different concrete batch shapes (playback vs. verification). */
  uniform(fields: readonly number[]): GPUBuffer {
    const key = `params:${this.node.name}:${this.uniformCallIndex++}:${fields.join(",")}`;
    return this.cachedUniform(key, () => packParams(fields));
  }

  /** Like `uniform`, for Params structs that mix u32 and f32 fields (e.g. clamp,
   * group_norm/layer_norm's eps). */
  uniformTyped(fields: readonly TypedParamField[]): GPUBuffer {
    const key = `params:${this.node.name}:${this.uniformCallIndex++}:${JSON.stringify(fields)}`;
    return this.cachedUniform(key, () => packTypedParams(fields));
  }

  private cachedUniform(key: string, pack: () => ArrayBuffer): GPUBuffer {
    let buffer = this.env.constantCache.get(key);
    if (!buffer) {
      buffer = createUniformBuffer(this.env.device, pack());
      this.env.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Register this node's computed result so later nodes can resolve it by name. */
  setOutput(buffer: GPUBuffer, shape: readonly number[], imag?: GPUBuffer): void {
    this.env.resolved.set(this.node.name, { buffer, shape: [...shape], imag });
  }

  /** Register an already-built ResolvedTensor as this node's result (e.g. a free
   * passthrough that reuses an existing buffer, or a complex pairing). */
  setOutputTensor(tensor: ResolvedTensor): void {
    this.env.resolved.set(this.node.name, tensor);
  }

  /** For multi-output nodes (e.g. aten.chunk.default): register one of several results
   * under a synthetic per-index key, later picked up by a getitem(thisNode, i) node. */
  setIndexedOutput(index: number, tensor: ResolvedTensor): void {
    this.env.resolved.set(`${this.node.name}::${index}`, tensor);
  }

  /** Resolve a multi-output node's i'th result, registered via `setIndexedOutput`. */
  resolveIndexed(nodeName: string, index: number): ResolvedTensor {
    const tensor = this.env.resolved.get(`${nodeName}::${index}`);
    if (!tensor) {
      throw new KumaShapeError(
        `Node "${this.node.name}" (getitem) references "${nodeName}"[${index}], which was never registered as a multi-output result.`,
      );
    }
    return tensor;
  }

  /** Record one dispatch of a one-invocation-per-element kernel. `buffers` must be in the
   * kernel's @binding(0..n) order; `dispatchElements` is the number of `out[]` elements.
   * Spreads across a 2D grid past 65535 workgroups (see KERNEL_WORKGROUP_SIZE). Kernels
   * that read `workgroup_id` as real grid coordinates (tiled matmul/conv) use
   * `dispatchKernelGrid` instead. */
  dispatchKernel(kernelName: string, buffers: GPUBuffer[], dispatchElements: number): void {
    const maxElements = KERNEL_WORKGROUP_SIZE * MAX_WORKGROUPS_PER_DIMENSION * MAX_WORKGROUPS_PER_DIMENSION;
    if (dispatchElements > maxElements) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") dispatches ${dispatchElements} elements, exceeding ` +
          `a ${MAX_WORKGROUPS_PER_DIMENSION}x${MAX_WORKGROUPS_PER_DIMENSION} 2D dispatch grid's capacity.`,
      );
    }
    this.env.pipelines.dispatchLinear(this.pass, this.getPipeline(kernelName), buffers, dispatchElements, KERNEL_WORKGROUP_SIZE);
  }

  /** Like `dispatchKernel`, but `gridX`/`gridY`/`gridZ` are workgroup *counts* the kernel
   * reads directly via `workgroup_id` (e.g. ceil(N/TILE), ceil(M/TILE) for a tiled matmul;
   * batch*channels as Z for a tiled conv), not element counts. */
  dispatchKernelGrid(kernelName: string, buffers: GPUBuffer[], gridX: number, gridY: number, gridZ: number = 1): void {
    this.env.pipelines.dispatch(this.pass, this.getPipeline(kernelName), buffers, gridX, gridY, gridZ);
  }

  private getPipeline(kernelName: string): GPUComputePipeline {
    const source = this.env.kernels.get(kernelName);
    if (!source) {
      throw new KumaManifestError(`.iph package is missing kernel source "kernels/${kernelName}".`);
    }
    return this.env.pipelines.getPipeline(kernelName, source);
  }
}

export type OpHandler = (ctx: OpContext) => void;
