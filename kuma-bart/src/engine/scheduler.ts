import { isNodeRef, nodeRefsIn, type ArgValue, type GraphNode, type KumaManifest, type NodeRef } from "../types/manifest.js";
import {
  OpContext,
  EncodeEnv,
  pooledAllocator,
  type ResolvedTensor,
  type BufferPoolState,
  createBufferPoolState,
  acquireGenerationSlot,
  releaseGenerationSlot,
} from "./context.js";
import type { TransientSet } from "./transients.js";
import { numElements } from "./shape.js";
import { opRegistry, findLinearWeightElisions } from "../ops/index.js";
import { KumaManifestError, KumaUnsupportedOpError } from "../errors.js";
import { readBuffers, uploadFloat32 } from "webgpu-compute-kit";
import { getSnippetFn, type SnippetFn } from "./snippets.js";
import { synthesizeWeight } from "./synthetic_weights.js";

/** What every graph walk needs: the model's static state plus this call's inputs. */
export interface GraphParams {
  device: GPUDevice;
  manifest: KumaManifest;
  kernels: ReadonlyMap<string, string>;
  pipelineCache: Map<string, GPUComputePipeline>;
  weightBuffers: ReadonlyMap<string, ResolvedTensor>;
  /** Owned by the caller: runGraph never destroys them (it destroys only what it
   * allocated itself), so the same input buffer may be reused across calls. */
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
   * in the same single batch as the real outputs. Names not present in the run (e.g. a
   * node the graph didn't reach) are silently omitted from the result. */
  captureNodes?: ReadonlySet<string>;
}

/** GraphParams plus the knobs of the submit-and-read-back path (encodeGraph/runGraph). */
export interface RunGraphParams extends GraphParams {
  /** Skip the GPU->CPU readback for the manifest's own declared outputs -- `data` comes
   * back as an empty Float32Array(0) for them. Measured readback (mapAsync'ing a
   * multi-megabyte buffer back into JS) at 60-70ms for an 11MB frame, dwarfing the GPU
   * compute itself (~32ms) and dispatch encoding (~2ms). A caller that only wants
   * `buffer` (e.g. to render directly via a GPU render pass, which the GPU orders after
   * this submission with no CPU wait) should set this. `captureNodes` are always read. */
  skipOutputReadback?: boolean;
  /** Shared with OpContext.createBuffer -- persists pooled output buffers across calls,
   * same lifecycle as constantCache. Defaults to a fresh state when omitted. */
  bufferPool?: BufferPoolState;
  /** Dead intermediate bytes to batch before an early-free submit (see
   * EARLY_FREE_SUBMIT_BUDGET). A test can pass 0 to flush per buffer. */
  earlyFreeSubmitBudget?: number;
}

export interface RunGraphOutput {
  name: string;
  shape: number[];
  data: Float32Array;
  /** The GPU buffer `data` was read back from, still valid after runGraph returns -- for
   * a caller that renders or otherwise consumes the result on-GPU. It is either a pooled
   * buffer (reused by a later call `bufferPool.depth` calls from now), a weight/input
   * buffer the output aliases, or a fresh allocation this call made (outputs at or
   * above the pool's sizeThreshold). Call `release()` once done with it. */
  buffer: GPUBuffer;
  /** Set only for a captured node that happened to be complex-paired -- the manifest's
   * own declared outputs are always real in every model exported so far. */
  imag?: Float32Array;
  /** Gives `buffer` back. From runGraph: destroys it (and its imaginary-part buffer, if
   * any) iff this run allocated it fresh; a no-op for pooled, weight, and caller-owned
   * input buffers. From KumaModel.run, runRaw and runToGpu: once every output of that call has been
   * released, the call's scratch set returns to the model and `buffer` may be reused by
   * the next call. Idempotent, and safe to call while GPU work using the buffer is still
   * in flight (reuse and destruction are queue-ordered). */
  release(): void;
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

/**
 * Synchronous pre-pass: evaluates every `js_snippet` node against plain JS values
 * (model inputs, or another snippet's output — never a GPU buffer), resolves every
 * `switch` node's selector, and splices the chosen branch's nodes into the effective
 * execution list. Runs entirely before any GPU command encoder exists. Snippets may only
 * depend on model inputs so that branch selection never needs a mid-graph GPU readback.
 * The returned `seeded` buffers come from `upload` (by default a fresh uploadFloat32 the
 * caller owns).
 */
export function resolveSwitches(
  device: GPUDevice,
  nodes: readonly GraphNode[],
  rawInputs: ReadonlyMap<string, Float32Array>,
  snippetSources: ReadonlyMap<string, string>,
  snippetCache: Map<string, SnippetFn>,
  upload: (data: Float32Array) => GPUBuffer = (data) => uploadFloat32(device, data),
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
      // to the main loop unchanged.
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
        seeded.set(branchInputs[i]!.node_ref, { buffer: upload(data), shape: [data.length] });
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

/** Dead-but-not-yet-submitted intermediate bytes to let pile up before ending the pass to
 * free them. The early-free liveness pass needs a submitted pass before a buffer's memory
 * is reclaimable, but doing that per-buffer means dozens of queue.submit()s per frame, each
 * with real main-thread validation cost -- a big chunk of per-frame CPU on weak devices.
 * Batching frees into ~this-sized groups cuts that to a handful of submits/frame; the cost
 * is peak memory rising by up to this much (per in-flight frame). */
const EARLY_FREE_SUBMIT_BUDGET = 64 * 1024 * 1024; // 64 MB

export interface EncodeOptions {
  /** Free large intermediates this call allocated as soon as their last consumer has
   * been encoded (batched per EARLY_FREE_SUBMIT_BUDGET), instead of holding every one
   * until the end of the call. Only meaningful with a shared pass. */
  earlyFree?: boolean;
  /** Called once, before anything is encoded, with every node that will dispatch. */
  prepare?: (dispatchNodes: readonly GraphNode[]) => void;
  /** Gives each dispatching node its own compute pass (ended right after the node);
   * when omitted, every node shares one pass. */
  nodePass?: (encoder: GPUCommandEncoder, node: GraphNode) => GPUComputePassEncoder;
  /** Last chance to record into the final encoder before it is submitted. */
  beforeSubmit?: (encoder: GPUCommandEncoder) => void;
}

export interface EncodedGraph {
  /** Nodes that dispatched, in encode order. */
  dispatchedNodes: GraphNode[];
  /** The manifest's declared outputs, in `output`-node order. */
  outputTensors: ResolvedTensor[];
  /** `captureNodes` that were actually resolved, in the caller's iteration order. */
  captured: Array<{ name: string; tensor: ResolvedTensor }>;
  /** Buffers this call allocated (fresh transients and switch-seeded inputs) that have
   * not been destroyed yet -- weights, inputs, pooled and cached buffers never are. */
  owned: Set<GPUBuffer>;
  /** Destroys every buffer in `owned` except `keep`. */
  freeOwned(keep: ReadonlySet<GPUBuffer>): void;
}

/** The flattened node list plus everything derived from it before any GPU work. */
interface PreparedGraph {
  nodes: GraphNode[];
  seeded: Map<string, ResolvedTensor>;
  elisions: Map<string, string>;
  dispatchNodes: GraphNode[];
}

/** Switch resolution + registry check: everything that can fail before touching the
 * encoder. `upload` seeds switch-branch inputs (see resolveSwitches). */
function prepareGraph(params: GraphParams, upload?: (data: Float32Array) => GPUBuffer): PreparedGraph {
  const { device, manifest } = params;
  const rawInputs = params.rawInputs ?? new Map<string, Float32Array>();
  const snippets = params.snippets ?? new Map<string, string>();
  const snippetCache = params.snippetCache ?? new Map<string, SnippetFn>();
  const { nodes, seeded } = resolveSwitches(device, manifest.graph.nodes, rawInputs, snippets, snippetCache, upload);
  const elisions = findLinearWeightElisions(nodes);
  const dispatchNodes = nodes.filter((n) => n.op !== "placeholder" && n.op !== "output" && !elisions.has(n.name));
  for (const node of dispatchNodes) {
    if (!opRegistry.has(node.target)) throw new KumaUnsupportedOpError(node.target, node.name);
  }
  return { nodes, seeded, elisions, dispatchNodes };
}

/** How a graph walk talks to whoever owns the command encoder. */
interface WalkHooks {
  /** The pass to record `node`'s dispatches into. */
  pass(node: GraphNode): GPUComputePassEncoder;
  /** Called right after `node`'s handler returned, with the pass it recorded into. */
  afterNode?(node: GraphNode, pass: GPUComputePassEncoder): void;
  /** `buffers` were handed to a node that dispatched nothing, so it is relying on them
   * being zero-filled (e.g. aten.zeros_like) -- which a recycled buffer is not. The owner
   * of the encoder must `clearBuffer` them, outside any open pass. */
  clear(buffers: readonly GPUBuffer[]): void;
  /** Liveness (when `track` is set): a buffer this walk resolved whose last consumer has
   * just been encoded and that no other live or protected name still aliases. Weights,
   * inputs and cached constants are reported too -- the hook decides what it owns. */
  dead?(buffer: GPUBuffer): void;
  /** After a node's liveness bookkeeping -- a chance to flush pending frees. */
  afterLiveness?(): void;
}

interface WalkResult {
  dispatchedNodes: GraphNode[];
  outputTensors: ResolvedTensor[];
  captured: Array<{ name: string; tensor: ResolvedTensor }>;
}

/** Walks the flattened graph in order (already topologically sorted by the exporter),
 * resolving placeholders and dispatching each node's op through its handler, with the
 * refcount liveness pass that drives early-free / transient recycling. Synchronous and
 * encoder-agnostic: the hooks own passes, clears and frees. The shared core of
 * encodeGraph (ring pool, submits) and recordGraph (caller's encoder, transient set). */
function walkGraph(params: GraphParams, prepared: PreparedGraph, env: EncodeEnv, track: boolean, hooks: WalkHooks): WalkResult {
  const { device, weightBuffers, inputBuffers } = params;
  const { nodes, elisions } = prepared;
  const resolved = env.resolved;
  const constantCache = env.constantCache;

  // Liveness pre-pass: how many later nodes consume each name, and which names must
  // survive the call (declared outputs and captured nodes).
  const refCounts = new Map<string, number>();
  const protectedNames = new Set<string>(params.captureNodes ?? []);
  for (const node of nodes) {
    for (const ref of nodeRefsIn(node.args)) refCounts.set(ref, (refCounts.get(ref) ?? 0) + 1);
    if (node.op === "output") for (const ref of nodeRefsIn(node.args)) protectedNames.add(ref);
  }

  // A single GPUBuffer is routinely shared by several resolved names (alias/passthrough/
  // view ops reuse their input's buffer -- see ops/passthrough.ts), so a name's refcount
  // reaching zero does not mean its buffer is dead: another alias may still have
  // consumers, or be protected. Scanning `resolved` is sufficient, because a future
  // aliaser must reference a name that already holds the buffer, whose count is still > 0.
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
  const reportDead = (buf: GPUBuffer | undefined, ref: string): void => {
    if (!buf || bufferStillNeeded(buf, ref)) return;
    hooks.dead?.(buf);
  };

  let outputNode: GraphNode | undefined;
  const dispatchedNodes: GraphNode[] = [];

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

    const handler = opRegistry.get(node.target)!;
    const pass = hooks.pass(node);
    const ctx = new OpContext(env, pass, node);
    handler(ctx);
    hooks.afterNode?.(node, pass);
    dispatchedNodes.push(node);

    // Multi-output nodes (e.g. aten.chunk.default) register per-index results via
    // setIndexedOutput instead of a single setOutput — nothing to check here for them.
    if (node.meta.outputs === undefined && !resolved.has(node.name)) {
      throw new KumaManifestError(
        `Internal: op handler for "${node.target}" did not register an output for node "${node.name}".`,
      );
    }
    // A handler that allocated but never dispatched is treating its buffers as zero-
    // filled; a recycled buffer isn't, so make it so.
    if (ctx.dispatchCount === 0 && ctx.createdBuffers.length > 0) hooks.clear(ctx.createdBuffers);

    if (!track) continue;
    for (const ref of nodeRefsIn(node.args)) {
      const count = refCounts.get(ref);
      if (count === undefined) continue;
      refCounts.set(ref, count - 1);
      if (count - 1 !== 0 || protectedNames.has(ref)) continue;
      const tensor = resolved.get(ref);
      if (!tensor) continue;
      reportDead(tensor.buffer, ref);
      reportDead(tensor.imag, ref);
    }
    hooks.afterLiveness?.();
  }

  if (!outputNode) throw new KumaManifestError('Manifest graph has no "output" node.');
  const outputRefs = (outputNode.args[0] as NodeRef[] | undefined) ?? [];
  const outputTensors = outputRefs.map((ref) => {
    const tensor = resolved.get(ref.node_ref);
    if (!tensor) {
      throw new KumaManifestError(`Output references "${ref.node_ref}" which was never computed.`);
    }
    return tensor;
  });

  const captured: Array<{ name: string; tensor: ResolvedTensor }> = [];
  for (const name of params.captureNodes ?? []) {
    const tensor = resolved.get(name);
    if (tensor) captured.push({ name, tensor });
  }

  return { dispatchedNodes, outputTensors, captured };
}

/** Resolves switches, walks the flattened graph dispatching each node's op into one
 * command encoder of its own, and submits it. The core of runGraph and profileGraph.
 * Scratch comes from the per-node ring pool (`params.bufferPool`, generation-rotated
 * across calls via acquireGenerationSlot); buffers at or above the pool's size threshold
 * are allocated fresh. Ownership rule: only buffers this call allocated are ever
 * destroyed here or by `freeOwned`. For recording into a caller's encoder without
 * submitting, see recordGraph. */
export async function encodeGraph(params: RunGraphParams, options: EncodeOptions = {}): Promise<EncodedGraph> {
  const { device, kernels, pipelineCache } = params;
  const constantCache = params.constantCache ?? new Map<string, GPUBuffer>();
  const bufferPool = params.bufferPool ?? createBufferPoolState();

  const prepared = prepareGraph(params);
  const owned = new Set<GPUBuffer>();
  for (const t of prepared.seeded.values()) {
    owned.add(t.buffer);
    if (t.imag) owned.add(t.imag);
  }
  const destroyOwned = (buffer: GPUBuffer): void => {
    if (owned.delete(buffer)) buffer.destroy();
  };
  const freeOwned = (keep: ReadonlySet<GPUBuffer>): void => {
    for (const buffer of [...owned]) if (!keep.has(buffer)) destroyOwned(buffer);
  };

  try {
    options.prepare?.(prepared.dispatchNodes);
  } catch (err) {
    freeOwned(new Set());
    throw err;
  }

  // Must happen before any pooled buffer (via OpContext.createBuffer) gets reused for
  // this call -- see acquireGenerationSlot's docs for what this does and doesn't wait for.
  await acquireGenerationSlot(device, bufferPool);

  const env = new EncodeEnv(device, kernels, pipelineCache, pooledAllocator(device, bufferPool, owned), constantCache, prepared.seeded);
  const earlyFree = (options.earlyFree ?? false) && !options.nodePass;

  // Early-free is an allowlist: only buffers this call allocated (`owned`) are
  // eligible, never weights, inputs, pooled or constant-cache buffers.
  const pendingFree = new Set<GPUBuffer>();
  let pendingFreeBytes = 0;
  const earlyFreeSubmitBudget = params.earlyFreeSubmitBudget ?? EARLY_FREE_SUBMIT_BUDGET;

  let encoder = device.createCommandEncoder();
  let sharedPass = options.nodePass ? undefined : encoder.beginComputePass();
  const hooks: WalkHooks = {
    pass: (node) => sharedPass ?? options.nodePass!(encoder, node),
    afterNode: (_node, pass) => {
      if (!sharedPass) pass.end();
    },
    clear: (buffers) => {
      sharedPass?.end();
      for (const buffer of buffers) encoder.clearBuffer(buffer);
      if (sharedPass) sharedPass = encoder.beginComputePass();
    },
    dead: (buf) => {
      if (!owned.has(buf) || pendingFree.has(buf)) return;
      pendingFree.add(buf);
      pendingFreeBytes += buf.size;
    },
    afterLiveness: () => {
      if (pendingFree.size === 0 || pendingFreeBytes < earlyFreeSubmitBudget) return;
      sharedPass!.end();
      device.queue.submit([encoder.finish()]);
      for (const buf of pendingFree) destroyOwned(buf);
      pendingFree.clear();
      pendingFreeBytes = 0;
      encoder = device.createCommandEncoder();
      sharedPass = encoder.beginComputePass();
    },
  };

  let walked: WalkResult;
  try {
    walked = walkGraph(params, prepared, env, earlyFree, hooks);
  } catch (err) {
    releaseGenerationSlot(device, bufferPool);
    freeOwned(new Set());
    throw err;
  }

  sharedPass?.end();
  options.beforeSubmit?.(encoder);
  device.queue.submit([encoder.finish()]);
  releaseGenerationSlot(device, bufferPool);

  return { ...walked, owned, freeOwned };
}

export interface RecordGraphOptions {
  /** The caller's encoder: every dispatch is recorded into it, and nothing is ever
   * submitted here. The caller submits (or discards) it. */
  encoder: GPUCommandEncoder;
  /** Scratch for this encode: every intermediate and switch-seeded input comes from here
   * and is recycled within the encode as soon as it is dead. Must be held exclusively
   * by this encode until the encoder has been submitted; the caller then `reset()`s it. */
  transients: TransientSet;
}

export interface RecordedGraph {
  /** Nodes that dispatched, in encode order. */
  dispatchedNodes: GraphNode[];
  /** The manifest's declared outputs, in `output`-node order. Buffers from
   * `transients` (or a weight/input a passthrough aliased) -- valid until the set is
   * reset. */
  outputTensors: ResolvedTensor[];
  /** `captureNodes` that were actually resolved, in the caller's iteration order. */
  captured: Array<{ name: string; tensor: ResolvedTensor }>;
}

/** Records the whole graph into `options.encoder` as one compute pass (split only around
 * buffer clears) and returns, synchronously, without submitting or waiting: the shared-
 * device path, where a caller batches several model encodes and its own kernels into
 * one submission. Scratch is `options.transients` only, so two recordGraph calls into
 * one encoder with two different sets never touch the same buffer. The ring pool and
 * generation counter are not involved. */
export function recordGraph(params: GraphParams, options: RecordGraphOptions): RecordedGraph {
  const { device, kernels, pipelineCache } = params;
  const { encoder, transients } = options;
  const constantCache = params.constantCache ?? new Map<string, GPUBuffer>();

  const prepared = prepareGraph(params, (data) => {
    const buffer = transients.acquire("switch-seed", data.byteLength);
    device.queue.writeBuffer(buffer, 0, data as unknown as Float32Array<ArrayBuffer>);
    return buffer;
  });
  const env = new EncodeEnv(device, kernels, pipelineCache, transients, constantCache, prepared.seeded);

  let pass = encoder.beginComputePass();
  const hooks: WalkHooks = {
    pass: () => pass,
    clear: (buffers) => {
      pass.end();
      for (const buffer of buffers) encoder.clearBuffer(buffer);
      pass = encoder.beginComputePass();
    },
    dead: (buf) => transients.free(buf),
  };
  try {
    const walked = walkGraph(params, prepared, env, true, hooks);
    pass.end();
    return walked;
  } catch (err) {
    pass.end();
    throw err;
  }
}

/** Encodes and submits the graph (see encodeGraph), reads back the declared outputs plus
 * any `captureNodes`, and frees every buffer this call allocated except the ones handed
 * back in the result (release those via `RunGraphOutput.release`). */
export async function runGraph(params: RunGraphParams): Promise<RunGraphOutput[]> {
  const { device, manifest } = params;
  const run = await encodeGraph(params, { earlyFree: true });
  const { outputTensors, captured, owned } = run;

  const capturedTensors = captured.map((c) => c.tensor);
  const tensorsToRead = params.skipOutputReadback ? capturedTensors : [...outputTensors, ...capturedTensors];
  const reads = tensorsToRead.flatMap((tensor) => {
    const byteLength = numElements(tensor.shape) * 4;
    const entries = [{ buffer: tensor.buffer, byteLength }];
    if (tensor.imag) entries.push({ buffer: tensor.imag, byteLength });
    return entries;
  });
  const datas = reads.length > 0 ? await readBuffers(device, reads) : [];

  let cursor = 0;
  const nextRead = (tensor: ResolvedTensor, skip: boolean): { data: Float32Array; imag?: Float32Array } => {
    if (skip) return { data: new Float32Array(0) };
    const data = datas[cursor++]!;
    const imag = tensor.imag ? datas[cursor++]! : undefined;
    return { data, imag };
  };
  const releaser = (tensor: ResolvedTensor) => (): void => {
    for (const buffer of [tensor.buffer, tensor.imag]) {
      if (buffer && owned.delete(buffer)) buffer.destroy();
    }
  };

  const outputs: RunGraphOutput[] = manifest.outputs.map((spec, i) => {
    const tensor = outputTensors[i]!;
    const { data, imag } = nextRead(tensor, !!params.skipOutputReadback);
    return { name: spec.name, shape: tensor.shape, data, buffer: tensor.buffer, imag, release: releaser(tensor) };
  });
  for (const { name, tensor } of captured) {
    const { data, imag } = nextRead(tensor, false);
    outputs.push({ name, shape: tensor.shape, data, buffer: tensor.buffer, imag, release: releaser(tensor) });
  }

  // destroy() while submitted work still references a buffer is well-defined: the memory
  // is reclaimed once that work completes, so no completion wait is needed here.
  const keep = new Set<GPUBuffer>();
  for (const t of [...outputTensors, ...capturedTensors]) {
    keep.add(t.buffer);
    if (t.imag) keep.add(t.imag);
  }
  run.freeOwned(keep);
  return outputs;
}

/** Releases every output's fresh buffer (see RunGraphOutput.release). */
export function releaseOutputs(outputs: readonly RunGraphOutput[]): void {
  for (const out of outputs) out.release();
}
