import type { GraphNode, KumaManifest, NodeRef } from "../types/manifest.js";
import { OpContext, type ResolvedTensor, createBufferPoolState, acquireGenerationSlot, releaseGenerationSlot } from "./context.js";
import { numElements } from "./shape.js";
import { opRegistry, findLinearWeightElisions } from "../ops/index.js";
import { KumaManifestError, KumaUnsupportedOpError } from "../errors.js";
import { readBuffers } from "../gpu/buffers.js";
import type { SnippetFn } from "./snippets.js";
import { resolveSwitches, type RunGraphParams } from "./scheduler.js";
import { synthesizeWeight } from "./synthetic_weights.js";

export interface OpTiming {
  name: string;
  target: string;
  microseconds: number;
}

export interface TargetTiming {
  target: string;
  totalMicroseconds: number;
  count: number;
  avgMicroseconds: number;
}

export interface ProfileReport {
  totalMicroseconds: number;
  /** One entry per dispatched node, in graph order. */
  perNode: OpTiming[];
  /** Same data grouped by aten target (op "class"), sorted by total time descending --
   * this is the "where do I focus" view. */
  byTarget: TargetTiming[];
}

async function readTimestamps(device: GPUDevice, queryBuffer: GPUBuffer, count: number): Promise<BigUint64Array> {
  const byteLength = count * 8; // GPUQuerySet timestamp entries are u64 nanoseconds
  const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(queryBuffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = new BigUint64Array(staging.getMappedRange(0, byteLength).slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}

/**
 * Same graph walk as runGraph, but every dispatched node gets its *own* compute pass
 * (instead of sharing one) with `timestampWrites` recording GPU-side start/end
 * timestamps -- still just one encoder and one submit, so this doesn't introduce the
 * CPU/GPU sync stalls a naive "submit + await after every node" approach would (that
 * was DEBUG_GRAPH's old per-node submit/await, removed once the NaN hunt that needed it
 * was over -- this reuses the lesson: multiple *passes* in one encoder cost only a
 * little pass-transition overhead, not a full pipeline drain). Requires the
 * "timestamp-query" GPU feature (see gpu/device.ts) -- most desktop Chrome/Edge GPUs
 * have it, but it's not universal, so this throws a clear error rather than silently
 * producing zeros when it's missing.
 *
 * Per-dispatch passes are a hard WebGPU constraint, not a tuning choice -- timestampWrites
 * can only be attached at pass boundaries, never inserted mid-pass between dispatches, so
 * there is no way to get per-node timing without one pass per node. That forces every one
 * of these 211+ dispatches to run in isolation instead of getting the overlap/pipelining
 * consecutive dispatches get when they share one pass (as a real run()/runToGpu() call
 * does) -- in practice this has measured 3-4x *higher* than real per-frame GPU time, not
 * "a bit higher". Treat byTarget's percentages as the signal; don't compare
 * totalMicroseconds against real frame time directly -- see KumaModel.profile(), which
 * reports a real comparable number (realMilliseconds) alongside this for exactly that
 * reason.
 */
export async function profileGraph(params: RunGraphParams): Promise<ProfileReport> {
  const { device, manifest, kernels, pipelineCache, weightBuffers, inputBuffers } = params;
  if (!device.features.has("timestamp-query")) {
    throw new Error(
      "profileGraph requires the WebGPU 'timestamp-query' feature, which this adapter/browser doesn't support.",
    );
  }
  const rawInputs = params.rawInputs ?? new Map<string, Float32Array>();
  const snippets = params.snippets ?? new Map<string, string>();
  const snippetCache = params.snippetCache ?? new Map<string, SnippetFn>();
  const constantCache = params.constantCache ?? new Map<string, GPUBuffer>();
  const bufferPool = params.bufferPool ?? createBufferPoolState();

  // Must happen before any pooled buffer (via OpContext.createBuffer below) gets
  // reused for this call -- shares the same pool/generation sequence as runGraph, so
  // a profile() call and a run()/runRaw()/runToGpu() call interleaved in time are
  // still safe relative to each other.
  await acquireGenerationSlot(device, bufferPool);

  const { nodes, seeded } = resolveSwitches(device, manifest.graph.nodes, rawInputs, snippets, snippetCache);
  const elisions = findLinearWeightElisions(nodes);
  const resolved = new Map<string, ResolvedTensor>(seeded);

  // Pass 1: just count the nodes that will actually dispatch a kernel, so the query
  // set can be sized exactly upfront. Deliberately doesn't resolve anything -- an
  // elided transpose's alias target (see ops/linear.ts's findLinearWeightElisions)
  // is its *immediate* input, which may itself be a genuine dispatch node (e.g. a
  // detach_ passthrough between the weight and the transpose), not a placeholder.
  // Resolving eagerly here, before that dependency has actually run, threw "elided
  // transpose aliases ... before it was resolved". Resolution happens in pass 2
  // below instead, interleaved with dispatch in node order -- same structure as
  // scheduler.ts's runGraph -- so any alias target is guaranteed already resolved
  // by the time it's needed.
  let dispatchCount = 0;
  for (const node of nodes) {
    if (node.op === "placeholder" || node.op === "output" || elisions.has(node.name)) continue;
    if (!opRegistry.get(node.target)) {
      throw new KumaUnsupportedOpError(node.target, node.name);
    }
    dispatchCount++;
  }

  const querySet = device.createQuerySet({ type: "timestamp", count: dispatchCount * 2 });
  const queryBuffer = device.createBuffer({
    size: dispatchCount * 2 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  const encoder = device.createCommandEncoder();
  // See scheduler.ts / OpContext: fresh non-pooled buffers (incl. op-internal
  // intermediates never in `resolved`) are collected here so cleanup can free them.
  const transientBuffers = new Set<GPUBuffer>();

  // Pass 2: single interleaved walk resolving placeholders/elided aliases inline and
  // dispatching a timestamped GPU pass for every other node.
  const dispatchNodes: GraphNode[] = [];
  let outputNode: GraphNode | undefined;
  let dispatchIndex = 0;

  for (const node of nodes) {
    if (node.op === "placeholder") {
      if (node.kind === "parameter" || node.kind === "buffer") {
        const tensor = weightBuffers.get(node.weight_name!)
          ?? synthesizeWeight(device, constantCache, node.weight_name!, node.meta.shape ?? []);
        if (!tensor) {
          throw new KumaManifestError(`Manifest references weight "${node.weight_name}" with no matching entry in weights[].`);
        }
        resolved.set(node.name, tensor);
      } else {
        const tensor = resolved.get(node.name) ?? inputBuffers.get(node.name);
        if (!tensor) {
          throw new KumaManifestError(`Missing input for placeholder "${node.name}" — pass it to KumaModel.run().`);
        }
        resolved.set(node.name, tensor);
      }
      continue;
    }
    if (node.op === "output") {
      outputNode = node;
      continue;
    }
    const aliasTarget = elisions.get(node.name);
    if (aliasTarget !== undefined) {
      const tensor = resolved.get(aliasTarget);
      if (!tensor) {
        throw new KumaManifestError(`Internal: elided transpose "${node.name}" aliases "${aliasTarget}" before it was resolved.`);
      }
      resolved.set(node.name, tensor);
      continue;
    }

    const handler = opRegistry.get(node.target)!;
    const pass = encoder.beginComputePass({
      timestampWrites: { querySet, beginningOfPassWriteIndex: dispatchIndex * 2, endOfPassWriteIndex: dispatchIndex * 2 + 1 },
    });
    const ctx = new OpContext(device, kernels, pipelineCache, pass, resolved, node, constantCache, bufferPool, transientBuffers);
    handler(ctx);
    pass.end();

    const isMultiOutput = node.meta.outputs !== undefined;
    if (!isMultiOutput && !resolved.has(node.name)) {
      throw new KumaManifestError(`Internal: op handler for "${node.target}" did not register an output for node "${node.name}".`);
    }

    dispatchNodes.push(node);
    dispatchIndex++;
  }

  encoder.resolveQuerySet(querySet, 0, dispatchNodes.length * 2, queryBuffer, 0);
  device.queue.submit([encoder.finish()]);
  releaseGenerationSlot(device, bufferPool);

  if (!outputNode) {
    throw new KumaManifestError('Manifest graph has no "output" node.');
  }
  const outputRefs = (outputNode.args[0] as NodeRef[] | undefined) ?? [];
  const reads = outputRefs.map((ref) => {
    const tensor = resolved.get(ref.node_ref);
    if (!tensor) {
      throw new KumaManifestError(`Output references "${ref.node_ref}" which was never computed.`);
    }
    return { buffer: tensor.buffer, byteLength: numElements(tensor.shape) * 4 };
  });
  // Triggers/awaits the readback so this measures a *complete* frame (including the
  // final copy-out), same as a real model.run() call would actually pay for.
  await readBuffers(device, reads);

  const timestamps = await readTimestamps(device, queryBuffer, dispatchNodes.length * 2);
  querySet.destroy();
  queryBuffer.destroy();

  // ProfileReport doesn't hand any buffer back to the caller, so everything except
  // weights and pooled buffers (which persist by design -- see context.ts's
  // BufferPoolState) is safe to destroy now that the readback above is done.
  const keep = new Set<GPUBuffer>();
  for (const t of weightBuffers.values()) {
    keep.add(t.buffer);
    if (t.imag) keep.add(t.imag);
  }
  for (const buffer of constantCache.values()) {
    keep.add(buffer);
  }
  for (const slots of bufferPool.pools.values()) {
    for (const buffer of slots) {
      if (buffer) keep.add(buffer);
    }
  }
  const destroyed = new Set<GPUBuffer>();
  const destroyIfTemporary = (buffer: GPUBuffer): void => {
    if (keep.has(buffer) || destroyed.has(buffer)) return;
    destroyed.add(buffer);
    buffer.destroy();
  };
  for (const tensor of resolved.values()) {
    destroyIfTemporary(tensor.buffer);
    if (tensor.imag) destroyIfTemporary(tensor.imag);
  }
  for (const buffer of transientBuffers) {
    destroyIfTemporary(buffer);
  }

  const perNode: OpTiming[] = dispatchNodes.map((node, i) => {
    const start = timestamps[i * 2]!;
    const end = timestamps[i * 2 + 1]!;
    return { name: node.name, target: node.target, microseconds: Number(end - start) / 1000 };
  });

  const byTargetMap = new Map<string, { total: number; count: number }>();
  for (const t of perNode) {
    const entry = byTargetMap.get(t.target) ?? { total: 0, count: 0 };
    entry.total += t.microseconds;
    entry.count += 1;
    byTargetMap.set(t.target, entry);
  }
  const byTarget: TargetTiming[] = Array.from(byTargetMap.entries())
    .map(([target, { total, count }]) => ({ target, totalMicroseconds: total, count, avgMicroseconds: total / count }))
    .sort((a, b) => b.totalMicroseconds - a.totalMicroseconds);

  const totalMicroseconds = perNode.reduce((sum, t) => sum + t.microseconds, 0);
  return { totalMicroseconds, perNode, byTarget };
}
