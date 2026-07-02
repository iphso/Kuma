import { isNodeRef, type ArgValue, type GraphNode } from "../types/manifest.js";
import { createStorageBuffer, createUniformBuffer, uploadFloat32 } from "../gpu/buffers.js";
import { packParams, packTypedParams, type TypedParamField } from "../gpu/params.js";
import { KumaShapeError, KumaManifestError } from "../errors.js";
import { numElements } from "./shape.js";
import { computeDispatchGrid, KERNEL_WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIMENSION } from "./dispatch.js";

export interface ResolvedTensor {
  buffer: GPUBuffer;
  shape: number[];
  /** Set when this tensor logically represents a complex64 value — `buffer` holds the
   * real part, `imag` the imaginary part, paired up purely on the JS side by
   * aten.complex.default (see ops/complex.ts). No kernel ever sees an interleaved
   * complex buffer; real/imag are always two separate same-shape real GPUBuffers. */
  imag?: GPUBuffer;
}

/** How many copies of each (node, call-within-node) output buffer to keep alive and
 * rotate through, instead of allocating-then-immediately-destroying a fresh one every
 * single call -- this is the fix for real, measured GC pauses (confirmed via Chrome
 * DevTools Performance: "GC surge right before the big lag", correlated with dropped
 * frames on the Frames track) caused by ~270+ fresh GPUBuffer/bind-group/uniform-buffer
 * allocations every frame, at up to 60fps. Must be large enough that, by the time a
 * slot's generation comes back around, the GPU has had a real chance to actually finish
 * with its previous occupant -- see acquireGenerationSlot/releaseGenerationSlot. Not
 * tied to any one model's branch count or shape; this is a fixed rotation depth applied
 * uniformly per (node, call-index), however many of those a given manifest has.
 *
 * Only applies to buffers below POOL_SIZE_THRESHOLD -- large spatial activations bypass
 * the pool entirely and are allocated fresh each call, letting the WebGPU driver recycle
 * their VRAM at the OS/driver level (not JS GC). Measured: the bunny model has 506 MB of
 * unique intermediate buffers, 61 of which exceed 1 MB; pooling all of them at depth 3
 * reserved ~1.5 GB of VRAM on load and OOM'd on any device without dedicated VRAM.
 * Pooling only the small ones (<1 MB) reduces steady-state to ~30 MB while preserving
 * the GC-pause fix for the ~210 small/medium buffers it was actually needed for.
 *
 * Depth 3: full pipelining for the common case (discrete/desktop GPUs), where the extra
 * in-flight slot noticeably smooths playback over depth 2 -- dropping to 2 traded away
 * that smoothness for a VRAM saving most devices don't need. Memory-constrained devices
 * (Chromebooks, integrated-GPU laptops) still want fewer in-flight frames; rather than
 * lowering the shared default, model.ts's device-tier heuristic (see
 * gpu/device.ts's isLikelyLowMemoryDevice) auto-selects a lower depth for those devices
 * specifically, so the common case doesn't pay for the constrained case's trade-off.
 *
 * This is only the *default*; a per-model override travels on BufferPoolState.depth (see
 * createBufferPoolState / KumaModel.load's memory options). Depth 1 disables rotation
 * entirely -- every call reuses the one slot, which acquireGenerationSlot serializes
 * against the GPU's completion signal, collapsing peak pool VRAM to a single generation's
 * worth at the cost of throughput. That's the right trade on a Chromebook. */
export const DEFAULT_BUFFER_POOL_DEPTH = 3;

/** Back-compat re-export of the default depth. KumaPlayer derives a *cold-start* in-flight
 * cap from this for the window before a model has actually loaded; once one has, it reads
 * the model's real (possibly overridden) depth instead. */
export const BUFFER_POOL_DEPTH = DEFAULT_BUFFER_POOL_DEPTH;

/** Buffers at least this big bypass the pool and are allocated fresh each call, then freed
 * eagerly (by scheduler.ts's liveness pass) the moment their last consumer is encoded --
 * so peak VRAM tracks the *concurrently-live* large-activation set, not the whole graph's
 * sum. This is only the *default*; a per-model override travels on
 * BufferPoolState.sizeThreshold. Lowering it pushes more buffers onto that
 * allocate-fresh/free-early path, shrinking steady-state pool footprint toward just the
 * weights, at the cost of more allocation churn and more mid-graph submits. A threshold of
 * 0 frees *every* intermediate at its last use -- the absolute-minimum-memory setting. */
export const DEFAULT_POOL_SIZE_THRESHOLD = 1024 * 1024; // 1 MB

/** Persists across every call sharing it (one instance lives on KumaModel, threaded
 * through exactly like pipelineCache/constantCache) -- `pools` is keyed by
 * `${node.name}::${call-index-within-that-node}` (a node can call ctx.createBuffer more
 * than once, e.g. fft.ts's multi-step intermediates), each entry a fixed-size ring of
 * BUFFER_POOL_DEPTH slots. Slots are null until the generation that first needs them,
 * then lazily allocated and reused forever after -- this avoids allocating all depth
 * copies upfront on first use (which previously tripled the pool footprint during
 * warmUp). `callGeneration`/`confirmedGeneration` guard reuse safety -- see
 * acquireGenerationSlot. */
export interface BufferPoolState {
  pools: Map<string, (GPUBuffer | null)[]>;
  callGeneration: number;
  confirmedGeneration: number;
  /** Ring depth for this model's pool. Defaults to DEFAULT_BUFFER_POOL_DEPTH; a
   * memory-constrained caller can lower it to 1 (no rotation). Lives here rather than as
   * a module const so two models loaded in one page can pick different trade-offs. */
  depth: number;
  /** Per-model copy of the pool/early-free byte threshold. Defaults to
   * DEFAULT_POOL_SIZE_THRESHOLD; lowering it (toward 0) shrinks steady-state footprint. */
  sizeThreshold: number;
}

export interface BufferPoolOptions {
  /** Ring depth (>=1). Omit for DEFAULT_BUFFER_POOL_DEPTH. */
  depth?: number;
  /** Pool/early-free byte threshold (>=0). Omit for DEFAULT_POOL_SIZE_THRESHOLD. */
  sizeThreshold?: number;
}

export function createBufferPoolState(options: BufferPoolOptions = {}): BufferPoolState {
  return {
    pools: new Map(),
    callGeneration: 0,
    confirmedGeneration: -1,
    depth: Math.max(1, options.depth ?? DEFAULT_BUFFER_POOL_DEPTH),
    sizeThreshold: Math.max(0, options.sizeThreshold ?? DEFAULT_POOL_SIZE_THRESHOLD),
  };
}

/** Call once, before dispatching anything, from every entry point that shares this
 * pool (runGraph, profileGraph). WebGPU has no per-resource fence -- the only
 * completion signal at all is onSubmittedWorkDone(), measured this session at a real
 * ~30ms latency *independent of how much work or data is involved*. This must never
 * let a call reuse a pooled slot before its previous occupant's GPU work has actually
 * finished, without paying that ~30ms on literally every call (which would undo the
 * whole point of pooling). Usually resolves immediately: confirmedGeneration is kept
 * caught up asynchronously by releaseGenerationSlot below, so by the time a slot's
 * generation comes back around (BUFFER_POOL_DEPTH calls later) it has very likely
 * already been confirmed without this needing to await anything itself. Only actually
 * awaits when the caller is submitting faster than the GPU + its completion-
 * notification latency can keep up with -- the same amortization idea as this
 * session's in-flight pipelining, just now owning a real correctness invariant
 * (no premature buffer reuse) instead of a soft memory-safety heuristic. */
export async function acquireGenerationSlot(device: GPUDevice, pool: BufferPoolState): Promise<void> {
  const reuseGeneration = pool.callGeneration - pool.depth;
  if (reuseGeneration >= 0 && pool.confirmedGeneration < reuseGeneration) {
    await device.queue.onSubmittedWorkDone();
    pool.confirmedGeneration = pool.callGeneration - 1;
  }
}

/** Call once, right after submit, from the same caller as acquireGenerationSlot.
 * Advances callGeneration for whichever call comes next, and -- not awaited here, on
 * purpose -- kicks off tracking for when *this* call's own GPU work actually
 * completes, so a future acquireGenerationSlot call BUFFER_POOL_DEPTH generations from
 * now finds confirmedGeneration already caught up rather than needing to wait. */
export function releaseGenerationSlot(device: GPUDevice, pool: BufferPoolState): void {
  const myGeneration = pool.callGeneration;
  pool.callGeneration = myGeneration + 1;
  void device.queue.onSubmittedWorkDone().then(() => {
    pool.confirmedGeneration = Math.max(pool.confirmedGeneration, myGeneration);
  });
}

/**
 * Per-node execution context handed to an op handler. Wraps the GPU mechanics
 * (pipeline caching, bind groups, dispatch) so handlers only deal with tensors and
 * shapes. Every kernel binds its buffers in the exact order declared in its WGSL
 * source (storage buffers first, uniform Params last) — `dispatchKernel` relies on
 * callers passing buffers in that same order.
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

  constructor(
    public readonly device: GPUDevice,
    private readonly kernels: ReadonlyMap<string, string>,
    private readonly pipelineCache: Map<string, GPUComputePipeline>,
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
  ) {}

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
   * happen) -- by the time this runs, it's already safe to hand back whichever slot
   * `bufferPool.callGeneration % bufferPool.depth` points at.
   *
   * Slots are allocated lazily -- only the slot for the current generation is created
   * on first encounter, not all `depth` at once. This reduces peak memory during warmUp
   * (which previously allocated all depth copies on a node's first run) by up to
   * `depth`×. After `depth` distinct generations have touched a node the ring is fully
   * warm -- same steady state as before. (A low-memory model runs at depth 1, so there's
   * no ring at all: one slot, reused, with acquireGenerationSlot serializing against the
   * GPU between calls.)
   *
   * Buffers >= `bufferPool.sizeThreshold` bypass the pool and are allocated fresh -- the
   * caller (scheduler.ts's cleanup pass) will destroy them after submit, letting the
   * driver reclaim VRAM without going through JS GC. The call-index still increments
   * so subsequent small-buffer calls within the same node get distinct pool keys. */
  createBuffer(shape: readonly number[]): GPUBuffer {
    const byteLength = numElements(shape) * 4;
    const callIndex = this.createBufferCallIndex++;
    if (byteLength >= this.bufferPool.sizeThreshold) {
      const buffer = createStorageBuffer(this.device, byteLength);
      this.transientBuffers.add(buffer);
      return buffer;
    }
    const key = `${this.node.name}::${callIndex}`;
    let slots = this.bufferPool.pools.get(key);
    if (!slots) {
      slots = new Array<GPUBuffer | null>(this.bufferPool.depth).fill(null);
      this.bufferPool.pools.set(key, slots);
    }
    const slotIndex = this.bufferPool.callGeneration % this.bufferPool.depth;
    let buffer = slots[slotIndex];
    if (!buffer) {
      buffer = createStorageBuffer(this.device, byteLength);
      slots[slotIndex] = buffer;
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

  private getPipeline(kernelName: string): GPUComputePipeline {
    let pipeline = this.pipelineCache.get(kernelName);
    if (!pipeline) {
      const source = this.kernels.get(kernelName);
      if (!source) {
        throw new KumaManifestError(`.iph package is missing kernel source "kernels/${kernelName}".`);
      }
      const module = this.device.createShaderModule({ code: source });
      pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelineCache.set(kernelName, pipeline);
    }
    return pipeline;
  }

  private bindAndDispatch(kernelName: string, buffers: GPUBuffer[], gridX: number, gridY: number, gridZ: number): void {
    const pipeline = this.getPipeline(kernelName);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bindGroup);
    this.pass.dispatchWorkgroups(gridX, gridY, gridZ);
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
    this.bindAndDispatch(kernelName, buffers, grid.x, grid.y, 1);
  }

  /** Like `dispatchKernel`, but for a kernel whose `@builtin(workgroup_id)` is read
   * directly as grid coordinates (2D or 3D) rather than folded into a linear element
   * index -- `gridX`/`gridY`/`gridZ` are workgroup *counts* along each axis (e.g.
   * ceil(N/TILE), ceil(M/TILE) for a tiled matmul; or ceil(W/TILE), ceil(H/TILE),
   * batch*channels for a tiled conv), not element counts. `gridZ` defaults to 1 for
   * kernels that only need a 2D grid. */
  dispatchKernelGrid(kernelName: string, buffers: GPUBuffer[], gridX: number, gridY: number, gridZ: number = 1): void {
    if (gridX > MAX_WORKGROUPS_PER_DIMENSION || gridY > MAX_WORKGROUPS_PER_DIMENSION || gridZ > MAX_WORKGROUPS_PER_DIMENSION) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") needs a ${gridX}x${gridY}x${gridZ} workgroup grid, ` +
          `exceeding the ${MAX_WORKGROUPS_PER_DIMENSION}-per-dimension dispatch limit.`,
      );
    }
    this.bindAndDispatch(kernelName, buffers, Math.max(1, gridX), Math.max(1, gridY), Math.max(1, gridZ));
  }
}

export type OpHandler = (ctx: OpContext) => void;
