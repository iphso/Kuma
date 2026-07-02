import { isNodeRef, type ArgValue, type GraphNode, type KumaManifest, type NodeRef } from "../types/manifest.js";
import {
  OpContext,
  type ResolvedTensor,
  type BufferPoolState,
  createBufferPoolState,
  acquireGenerationSlot,
  releaseGenerationSlot,
} from "./context.js";
import { numElements } from "./shape.js";
import { opRegistry, findLinearWeightElisions } from "../ops/index.js";
import { KumaManifestError, KumaUnsupportedOpError } from "../errors.js";
import { readBuffers, uploadFloat32 } from "../gpu/buffers.js";
import { getSnippetFn, type SnippetFn } from "./snippets.js";
import { synthesizeWeight } from "./synthetic_weights.js";

export interface RunGraphParams {
  device: GPUDevice;
  manifest: KumaManifest;
  kernels: ReadonlyMap<string, string>;
  pipelineCache: Map<string, GPUComputePipeline>;
  weightBuffers: ReadonlyMap<string, ResolvedTensor>;
  inputBuffers: ReadonlyMap<string, ResolvedTensor>;
  /** Raw (pre-upload) input arrays, keyed the same way as `inputBuffers` — what
   * `js_snippet` nodes evaluate against, since they only ever depend on top-level
   * model inputs (never a GPU-computed intermediate; see engine/snippets.ts). Optional:
   * only needed for manifests that actually contain js_snippet/switch nodes. */
  rawInputs?: ReadonlyMap<string, Float32Array>;
  snippets?: ReadonlyMap<string, string>;
  snippetCache?: Map<string, SnippetFn>;
  /** Shared with OpContext.getOrUploadConstant — persists shape-derived constants (e.g.
   * DFT basis matrices) across nodes and across calls to runGraph, not just within one.
   * Defaults to a fresh (i.e. cold) Map when omitted. */
  constantCache?: Map<string, GPUBuffer>;
  /** Node names to read back and return *in addition to* the manifest's own declared
   * outputs (appended to the same returned array, found by name) -- e.g. the
   * golden-value verifier (engine/verify.ts) wants every node golden.json has an entry
   * for, not just whatever the model itself outputs. Every captured buffer is read back
   * in the same single batch as the real outputs -- no per-node slowdown. Names not
   * present in `resolved` (e.g. a typo, or a node the graph didn't actually reach) are
   * silently omitted from the result rather than failing the whole run. */
  captureNodes?: ReadonlySet<string>;
  /** Skip the GPU->CPU readback for the manifest's own declared outputs -- `data` comes
   * back as an empty Float32Array(0) for them (still real and correctly-ordered: just
   * unread). Measured readback (mapAsync'ing a multi-megabyte buffer back into JS) at
   * 60-70ms for an 11MB frame, dwarfing both the GPU compute itself (~32ms) and the
   * CPU-side cost of encoding every dispatch (~2ms) -- this is mapAsync/IPC overhead in
   * the browser's WebGPU implementation, paid for data that then goes unused. A caller that
   * only wants `buffer` (e.g. to render directly via a GPU render pass, which the GPU
   * itself correctly orders after this submission with no CPU wait needed) should set
   * this. `captureNodes` are unaffected -- those are always read back, since the only
   * current caller of that (the verifier) always needs the data. */
  skipOutputReadback?: boolean;
  /** Shared with OpContext.createBuffer -- persists pooled output buffers across calls,
   * same lifecycle as constantCache. Defaults to a fresh (i.e. cold, unpooled-benefit)
   * state when omitted -- see createBufferPoolState. */
  bufferPool?: BufferPoolState;
  /** Dead intermediate bytes to batch before an early-free submit (see
   * EARLY_FREE_SUBMIT_BUDGET). Defaults to that const; a test can pass 0 to flush per
   * buffer (the old immediate-free behavior). */
  earlyFreeSubmitBudget?: number;
}

export interface RunGraphOutput {
  name: string;
  shape: number[];
  data: Float32Array;
  /** The GPU buffer `data` was read back from -- still valid (not destroyed/reused;
   * this codebase doesn't pool buffers) after runGraph returns, so a caller that wants
   * to render or otherwise consume the result entirely on-GPU (e.g. the demo's canvas)
   * doesn't have to round-trip through `data` just to get a buffer reference back. */
  buffer: GPUBuffer;
  /** Set only for a captured node that happened to be complex-paired -- the manifest's
   * own declared outputs are always real in every model exported so far. */
  imag?: Float32Array;
}

export interface SwitchResolution {
  /** The flattened, GPU-loop-ready node list: js_snippet nodes and JS-routed getitem
   * nodes removed entirely (nothing for the GPU to do), switch nodes replaced by
   * their one chosen branch's nodes plus a synthetic alias so later refs to the
   * switch's own name still resolve. */
  nodes: GraphNode[];
  /** Branch-input bindings, uploaded from JS values — merge into `resolved` before
   * the main loop runs. */
  seeded: Map<string, ResolvedTensor>;
}

/** Recursively collect every NodeRef's node_ref string from an arg value tree. */
function nodeRefNames(arg: ArgValue, out: string[] = []): string[] {
  if (isNodeRef(arg)) out.push(arg.node_ref);
  else if (Array.isArray(arg)) for (const a of arg) nodeRefNames(a, out);
  return out;
}

/**
 * Synchronous pre-pass: evaluates every `js_snippet` node against plain JS values
 * (model inputs, or another snippet's output — never a GPU buffer, per the v1 scope
 * restriction), resolves every `switch` node's selector, and splices the chosen
 * branch's nodes into the effective execution list. Runs entirely before any GPU
 * command encoder exists. See kuma-bart's plan notes for why this is scoped to
 * input-only snippet dependencies (sidesteps mid-graph GPU readback entirely).
 *
 * Exported so engine/profile.ts can run the exact same branch-selection logic
 * runGraph does, without duplicating it.
 */
export function resolveSwitches(
  device: GPUDevice,
  nodes: readonly GraphNode[],
  rawInputs: ReadonlyMap<string, Float32Array>,
  snippetSources: ReadonlyMap<string, string>,
  snippetCache: Map<string, SnippetFn>,
): SwitchResolution {
  const flat: GraphNode[] = [];
  const seeded = new Map<string, ResolvedTensor>();
  // Plain JS-side values only — a js_snippet's (possibly multi-) output, or a getitem
  // extracting one of them. Keyed "<node>::<index>" for multi-output, plus the
  // getitem's own name once extracted (so further refs by either name work).
  const jsOutputs = new Map<string, number[]>();
  const jsNodeNames = new Set<string>();

  function evalJsValue(ref: ArgValue): number[] {
    if (!isNodeRef(ref)) {
      throw new KumaManifestError(`js_snippet expected a value reference, got ${JSON.stringify(ref)}.`);
    }
    const raw = rawInputs.get(ref.node_ref);
    if (raw) return Array.from(raw);
    const fromJs = jsOutputs.get(ref.node_ref);
    if (fromJs) return fromJs;
    throw new KumaManifestError(
      `js_snippet input "${ref.node_ref}" is neither a model input nor a prior JS-snippet value.`,
    );
  }

  for (const node of nodes) {
    if (node.op === "js_snippet") {
      const fn = getSnippetFn(snippetCache, snippetSources, node.target);
      const inputs = node.args.map((arg) => evalJsValue(arg));
      const outputs = fn(inputs);
      outputs.forEach((arr, i) => jsOutputs.set(`${node.name}::${i}`, arr));
      jsNodeNames.add(node.name);
      continue;
    }

    if (node.op === "call_function" && node.target === "getitem") {
      const [sourceRef, indexArg] = node.args as [NodeRef, number];
      if (jsNodeNames.has(sourceRef.node_ref)) {
        const arr = jsOutputs.get(`${sourceRef.node_ref}::${indexArg}`);
        if (!arr) {
          throw new KumaManifestError(
            `getitem "${node.name}" references "${sourceRef.node_ref}"[${indexArg}], which produced no JS-snippet output.`,
          );
        }
        jsOutputs.set(node.name, arr);
        jsNodeNames.add(node.name);
        continue;
      }
      // Not JS-routed (e.g. a normal aten.chunk.default getitem) — flows through
      // to the main loop unchanged, exactly as before this feature existed.
      flat.push(node);
      continue;
    }

    if (node.op === "switch") {
      if (!node.selector) {
        throw new KumaManifestError(`switch "${node.name}" is missing its selector.`);
      }
      const selectorValues = jsOutputs.get(node.selector.node_ref);
      if (!selectorValues || selectorValues.length !== 1) {
        throw new KumaManifestError(
          `switch "${node.name}"'s selector ("${node.selector.node_ref}") must resolve to exactly one JS-side value.`,
        );
      }
      const selectorValue = Math.trunc(selectorValues[0]!);
      const branches = node.branches ?? [];
      const branch = branches[selectorValue];
      if (!branch) {
        throw new KumaManifestError(
          `switch "${node.name}": no branch for selector value ${selectorValue} (have ${branches.length}).`,
        );
      }

      const outerArgs = node.args;
      const branchInputs = branch.inputs;
      if (branchInputs.length !== outerArgs.length) {
        throw new KumaManifestError(
          `switch "${node.name}": branch declares ${branchInputs.length} input(s), but ${outerArgs.length} were provided.`,
        );
      }
      for (let i = 0; i < branchInputs.length; i++) {
        const arr = evalJsValue(outerArgs[i]!);
        const data = new Float32Array(arr);
        seeded.set(branchInputs[i]!.node_ref, { buffer: uploadFloat32(device, data), shape: [data.length] });
      }

      flat.push(...branch.nodes);
      // Synthetic free passthrough so later refs to the switch's own name (e.g. the
      // top-level output node) resolve to whichever branch actually ran.
      flat.push({
        id: -1,
        name: node.name,
        op: "call_function",
        target: "aten.alias.default",
        args: [branch.output],
        kwargs: {},
        meta: node.meta,
      });
      continue;
    }

    flat.push(node);
  }

  return { nodes: flat, seeded };
}

// CPU-side wall-clock breakdown of the phases engine/profile.ts's GPU timestamp queries
// can never see, since all of them happen entirely outside any compute pass: ENCODING
// (building a buffer + uniform buffer + bind group for every dispatch -- measured at
// ~2ms even for 276 nodes, not the bottleneck it looked like it might be), READBACK
// (the final GPU->CPU copy + mapAsync wait -- this WAS the bottleneck, at 60-70ms for
// an 11MB frame, which is why skipOutputReadback/runToGpu exists), and CLEANUP (the
// per-node buffer.destroy() pass -- ruled out as the "real frame time vs.
// GPU-timestamped breakdown" gap's cause, at <1ms; that gap turned out to be a fixed
// ~30ms GPU-completion-notification latency in this browser's implementation,
// independent of work/data size -- confirmed by the GPU-side shared-pass timestamp
// below matching the breakdown almost exactly while CPU-side waits for that same work
// to be reported done took ~30ms regardless. Demo's runAt now pipelines multiple
// frames in flight to amortize that latency instead of paying it every frame -- see
// demo/main.ts's MAX_IN_FLIGHT). Flip on for any new "why is this slow" dive; off by
// default since the per-frame debug GPU timestamp readback below pays that same ~30ms
// itself every call, which would defeat the in-flight pipelining if left on.
const DEBUG_TIMING = false;

/** Opt-in per-frame main-thread timing (enable with `?kumatime` in the URL). Unlike
 * DEBUG_TIMING it adds no GPU-side query overhead -- just wall-clock around the encode
 * loop plus a count of how many queue.submit()s a frame issued, to tell apart an
 * encode-bound frame from a submit-storm-bound one. */
let SCHED_TIME_DEBUG: boolean | undefined;
function schedTimeEnabled(): boolean {
  if (SCHED_TIME_DEBUG === undefined) {
    SCHED_TIME_DEBUG = typeof location !== "undefined" && /[?&]kumatime\b/.test(location.search);
  }
  return SCHED_TIME_DEBUG;
}

/** Dead-but-not-yet-submitted intermediate bytes to let pile up before ending the pass to
 * free them. The early-free liveness pass needs a submitted pass before a buffer's memory
 * is reclaimable, but doing that per-buffer means dozens of queue.submit()s per frame, each
 * with real main-thread validation cost -- a big chunk of per-frame CPU on weak devices.
 * Batching frees into ~this-sized groups cuts that to a handful of submits/frame; the cost
 * is peak memory rising by up to this much (per in-flight frame). */
const EARLY_FREE_SUBMIT_BUDGET = 64 * 1024 * 1024; // 64 MB

/** Walks `manifest.graph.nodes` in order (already topologically sorted by the Python
 * exporter), resolving each node to a GPUBuffer and dispatching its op, then reads back
 * the tensors the `output` node points at. */
export async function runGraph(params: RunGraphParams): Promise<RunGraphOutput[]> {
  const { device, manifest, kernels, pipelineCache, weightBuffers, inputBuffers } = params;
  const rawInputs = params.rawInputs ?? new Map<string, Float32Array>();
  const snippets = params.snippets ?? new Map<string, string>();
  const snippetCache = params.snippetCache ?? new Map<string, SnippetFn>();
  const constantCache = params.constantCache ?? new Map<string, GPUBuffer>();
  const bufferPool = params.bufferPool ?? createBufferPoolState();
  const t0 = DEBUG_TIMING ? performance.now() : 0;

  // Must happen before any pooled buffer (via OpContext.createBuffer below) gets
  // reused for this call -- see acquireGenerationSlot's own docs for what this does
  // and doesn't wait for.
  await acquireGenerationSlot(device, bufferPool);

  const { nodes, seeded } = resolveSwitches(device, manifest.graph.nodes, rawInputs, snippets, snippetCache);
  const elisions = findLinearWeightElisions(nodes);
  const resolved = new Map<string, ResolvedTensor>(seeded);

  // Liveness pre-pass: count how many later nodes consume each intermediate's output,
  // so we can free large non-pooled buffers (≥ bufferPool.sizeThreshold) as soon as their
  // last consumer has been encoded -- instead of keeping all 500+ MB of intermediates
  // alive simultaneously until the end of the encoding loop. Threshold is per-model (a
  // low-memory model lowers it toward 0, freeing nearly everything eagerly).
  const sizeThreshold = bufferPool.sizeThreshold;
  const refCounts = new Map<string, number>();
  for (const node of nodes) {
    for (const arg of node.args) {
      for (const ref of nodeRefNames(arg)) {
        refCounts.set(ref, (refCounts.get(ref) ?? 0) + 1);
      }
    }
  }
  // Buffers that must never be freed early.
  const earlyFreeExclusions = new Set<GPUBuffer>();
  for (const t of weightBuffers.values()) {
    earlyFreeExclusions.add(t.buffer);
    if (t.imag) earlyFreeExclusions.add(t.imag);
  }
  for (const t of inputBuffers.values()) {
    earlyFreeExclusions.add(t.buffer);
    if (t.imag) earlyFreeExclusions.add(t.imag);
  }
  for (const t of seeded.values()) {
    earlyFreeExclusions.add(t.buffer);
    if (t.imag) earlyFreeExclusions.add(t.imag);
  }
  // Synthesized weights (see synthetic_weights.ts) live in constantCache, cached across
  // calls just like real weights -- a one-shot consumer hitting refcount zero must not
  // trigger the early-free destroy() path for these.
  for (const buffer of constantCache.values()) {
    earlyFreeExclusions.add(buffer);
  }
  // Names whose buffers must survive past this call (model outputs and captured nodes).
  const protectedNames = new Set<string>(params.captureNodes ?? []);
  for (const node of nodes) {
    if (node.op === "output") {
      for (const arg of node.args) for (const ref of nodeRefNames(arg)) protectedNames.add(ref);
      break;
    }
  }

  // A single GPUBuffer is routinely shared by several resolved names: alias/passthrough/
  // view ops (aten.alias/contiguous/to/squeeze/unsqueeze, and many reshaping ops) reuse
  // their input's buffer rather than allocating -- see ops/passthrough.ts. So a per-name
  // refcount hitting zero means *that name* is done, NOT that the underlying buffer is:
  // another alias may still have pending consumers, or be a protected output. Freeing on
  // the name's count alone destroys a buffer a later dispatch still binds -- a
  // use-after-destroy. It stays latent at the default 1MB threshold (aliased buffers are
  // typically small) but fires constantly at sizeThreshold 0 (low-memory mode), where
  // every buffer is early-free-eligible. So gate the free on whether *any other* still-
  // live or protected name maps to the same buffer. Scanning the current `resolved` set is
  // sufficient: a future aliaser must reference a name that already holds this buffer (you
  // can't alias a buffer without naming something that holds it), and that name's own
  // refcount is therefore still > 0 here.
  const bufferStillNeeded = (buf: GPUBuffer, excludingRef: string): boolean => {
    for (const [name, t] of resolved) {
      if (name === excludingRef) continue;
      if (t.buffer !== buf && t.imag !== buf) continue;
      if (protectedNames.has(name)) return true;
      const c = refCounts.get(name);
      if (c !== undefined && c > 0) return true;
    }
    return false;
  };

  // Diagnostic: a GPU-side timestamp pair around the *whole* shared pass, to settle
  // directly (no CPU-side ambiguity) whether the shared pass's own GPU execution time
  // actually matches profile.ts's per-node-summed breakdown or runs much longer --
  // gated on DEBUG_TIMING since it's part of the same investigation.
  const debugGpuTiming = DEBUG_TIMING && device.features.has("timestamp-query");
  const debugQuerySet = debugGpuTiming ? device.createQuerySet({ type: "timestamp", count: 2 }) : undefined;

  let encoder = device.createCommandEncoder();
  let pass = encoder.beginComputePass(
    debugQuerySet ? { timestampWrites: { querySet: debugQuerySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
  );

  // Fresh (non-pooled) buffers handed out by OpContext.createBuffer, including op-internal
  // intermediates that never land in `resolved` -- collected so cleanup can free them (see
  // OpContext's constructor note; this is the low-memory per-frame leak fix).
  const transientBuffers = new Set<GPUBuffer>();

  // Early-free batching: dead intermediates accumulate here and are freed in one
  // pass-end/submit/destroy group once they exceed EARLY_FREE_SUBMIT_BUDGET, instead of a
  // separate submit per buffer (see the const's note).
  const pendingFree = new Set<GPUBuffer>();
  let pendingFreeBytes = 0;
  const earlyFreeSubmitBudget = params.earlyFreeSubmitBudget ?? EARLY_FREE_SUBMIT_BUDGET;

  const timeDbg = schedTimeEnabled();
  let submitCount = 0;
  const tEncode0 = timeDbg ? performance.now() : 0;

  let outputNode: GraphNode | undefined;

  for (const node of nodes) {
    if (node.op === "placeholder") {
      if (node.kind === "parameter" || node.kind === "buffer") {
        const tensor = weightBuffers.get(node.weight_name!)
          ?? synthesizeWeight(device, constantCache, node.weight_name!, node.meta.shape ?? []);
        if (!tensor) {
          throw new KumaManifestError(
            `Manifest references weight "${node.weight_name}" with no matching entry in weights[].`,
          );
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

    // call_function
    const aliasTarget = elisions.get(node.name);
    if (aliasTarget !== undefined) {
      const tensor = resolved.get(aliasTarget);
      if (!tensor) {
        throw new KumaManifestError(
          `Internal: elided transpose "${node.name}" aliases "${aliasTarget}" before it was resolved.`,
        );
      }
      resolved.set(node.name, tensor);
      continue;
    }

    const handler = opRegistry.get(node.target);
    if (!handler) {
      throw new KumaUnsupportedOpError(node.target, node.name);
    }

    const ctx = new OpContext(device, kernels, pipelineCache, pass, resolved, node, constantCache, bufferPool, transientBuffers);
    handler(ctx);

    // Multi-output nodes (e.g. aten.chunk.default) register per-index results via
    // setIndexedOutput instead of a single setOutput — nothing to check here for them.
    const isMultiOutput = node.meta.outputs !== undefined;
    if (!isMultiOutput && !resolved.has(node.name)) {
      throw new KumaManifestError(
        `Internal: op handler for "${node.target}" did not register an output for node "${node.name}".`,
      );
    }

    // Decrement ref counts for this node's inputs. Any large intermediate whose last
    // consumer just ran becomes freeable -- accumulate it in `pendingFree` and only end
    // the pass / submit / destroy the whole group once it exceeds the budget, so peak VRAM
    // stays bounded (to the live set plus one budget's worth of dead buffers) without a
    // separate submit per buffer.
    for (const ref of nodeRefNames(node.args.flat() as ArgValue[])) {
      const count = refCounts.get(ref);
      if (count === undefined) continue;
      const newCount = count - 1;
      refCounts.set(ref, newCount);
      if (newCount !== 0) continue;
      const tensor = resolved.get(ref);
      if (!tensor || protectedNames.has(ref)) continue;
      const buf = tensor.buffer;
      if (buf.size >= sizeThreshold && !earlyFreeExclusions.has(buf) && !pendingFree.has(buf)
          && !bufferStillNeeded(buf, ref)) {
        pendingFree.add(buf);
        pendingFreeBytes += buf.size;
      }
      if (tensor.imag && tensor.imag.size >= sizeThreshold && !earlyFreeExclusions.has(tensor.imag)
          && !pendingFree.has(tensor.imag) && !bufferStillNeeded(tensor.imag, ref)) {
        pendingFree.add(tensor.imag);
        pendingFreeBytes += tensor.imag.size;
      }
    }
    if (pendingFree.size > 0 && pendingFreeBytes >= earlyFreeSubmitBudget) {
      pass.end();
      device.queue.submit([encoder.finish()]);
      submitCount++;
      for (const buf of pendingFree) buf.destroy();
      pendingFree.clear();
      pendingFreeBytes = 0;
      encoder = device.createCommandEncoder();
      pass = encoder.beginComputePass();
    }
  }

  pass.end();
  let debugQueryBuffer: GPUBuffer | undefined;
  if (debugQuerySet) {
    debugQueryBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    encoder.resolveQuerySet(debugQuerySet, 0, 2, debugQueryBuffer, 0);
  }
  device.queue.submit([encoder.finish()]);
  submitCount++;
  releaseGenerationSlot(device, bufferPool);
  const t1 = DEBUG_TIMING ? performance.now() : 0;
  if (timeDbg) {
    const encodeMs = performance.now() - tEncode0;
    console.log(`[kuma-sched] encode+submit ${encodeMs.toFixed(1)}ms — ${submitCount} submit(s), ${nodes.length} nodes`);
  }

  if (debugQuerySet && debugQueryBuffer) {
    const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(debugQueryBuffer, 0, staging, 0, 16);
    device.queue.submit([copyEncoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(staging.getMappedRange(0, 16).slice(0));
    staging.unmap();
    staging.destroy();
    debugQuerySet.destroy();
    debugQueryBuffer.destroy();
    const gpuMicroseconds = Number(timestamps[1]! - timestamps[0]!) / 1000;
    console.log(`[kuma] runGraph GPU-only shared-pass time: ${(gpuMicroseconds / 1000).toFixed(2)}ms`);
  }

  if (!outputNode) {
    throw new KumaManifestError('Manifest graph has no "output" node.');
  }

  const outputRefs = (outputNode.args[0] as NodeRef[] | undefined) ?? [];
  const outputTensors = outputRefs.map((ref) => {
    const tensor = resolved.get(ref.node_ref);
    if (!tensor) {
      throw new KumaManifestError(`Output references "${ref.node_ref}" which was never computed.`);
    }
    return tensor;
  });

  const capturedNames: string[] = [];
  const capturedTensors: ResolvedTensor[] = [];
  for (const name of params.captureNodes ?? []) {
    const tensor = resolved.get(name);
    if (tensor) {
      capturedNames.push(name);
      capturedTensors.push(tensor);
    }
  }

  const tensorsToRead = params.skipOutputReadback ? capturedTensors : [...outputTensors, ...capturedTensors];
  const reads = tensorsToRead.flatMap((tensor) => {
    const byteLength = numElements(tensor.shape) * 4;
    const entries = [{ buffer: tensor.buffer, byteLength }];
    if (tensor.imag) entries.push({ buffer: tensor.imag, byteLength });
    return entries;
  });
  const datas = reads.length > 0 ? await readBuffers(device, reads) : [];
  const t2 = DEBUG_TIMING ? performance.now() : 0;

  let cursor = 0;
  const nextRead = (tensor: ResolvedTensor, skip: boolean): { data: Float32Array; imag?: Float32Array } => {
    if (skip) return { data: new Float32Array(0) };
    const data = datas[cursor++]!;
    const imag = tensor.imag ? datas[cursor++]! : undefined;
    return { data, imag };
  };

  const outputs: RunGraphOutput[] = manifest.outputs.map((spec, i) => {
    const tensor = outputTensors[i]!;
    const { data, imag } = nextRead(tensor, !!params.skipOutputReadback);
    return { name: spec.name, shape: tensor.shape, data, buffer: tensor.buffer, imag };
  });
  for (let i = 0; i < capturedNames.length; i++) {
    const tensor = capturedTensors[i]!;
    const { data, imag } = nextRead(tensor, false);
    outputs.push({ name: capturedNames[i]!, shape: tensor.shape, data, buffer: tensor.buffer, imag });
  }

  // call_function node outputs now come from OpContext.createBuffer's pool (see
  // BUFFER_POOL_DEPTH) rather than a fresh allocation every call, *except* for
  // anything a node builds beyond what its pool covers (shouldn't normally happen,
  // since every createBuffer call goes through the pool) and anything never routed
  // through createBuffer at all (input/seeded buffers, uploaded fresh per call by the
  // caller/resolveSwitches, not by op handlers). For those, `.destroy()` while GPU work
  // referencing the buffer is still in flight is well-defined per spec -- the
  // underlying memory isn't actually reclaimed until that work completes, so this
  // remains safe even though nothing here awaited onSubmittedWorkDone(). This is also
  // still what protects against the original OOM crash (VK_ERROR_OUT_OF_DEVICE_MEMORY,
  // then a lost device) this session, for whatever doesn't go through the pool. Keep:
  // weights (persist for the model's lifetime), every pooled buffer (persists by
  // design -- destroying one here would be reusing it into oblivion, not freeing
  // anything), and whatever's actually being handed back to the caller in `outputs`.
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
  for (const t of [...outputTensors, ...capturedTensors]) {
    keep.add(t.buffer);
    if (t.imag) keep.add(t.imag);
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
  // Op-internal intermediates that never entered `resolved` -- the resolved scan above
  // can't see them. `destroyIfTemporary` skips any already destroyed by the early-free
  // pass and any that are an output/pooled/weight buffer (in `keep`).
  for (const buffer of transientBuffers) {
    destroyIfTemporary(buffer);
  }
  if (DEBUG_TIMING) {
    const t3 = performance.now();
    console.log(
      `[kuma] runGraph timing: resolve+encode+submit=${(t1 - t0).toFixed(1)}ms readback=${(t2 - t1).toFixed(1)}ms cleanup=${(t3 - t2).toFixed(1)}ms total=${(t3 - t0).toFixed(1)}ms (${nodes.length} nodes, ${reads.length} buffer read(s), ${destroyed.size} buffer(s) destroyed${params.skipOutputReadback ? ", output readback skipped" : ""})`,
    );
  }

  return outputs;
}
