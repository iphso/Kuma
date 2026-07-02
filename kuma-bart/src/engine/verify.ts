import type { GraphNode, KumaManifest } from "../types/manifest.js";
import type { GoldenBranch, GoldenData, GoldenTensorStats } from "../types/golden.js";
import type { ResolvedTensor, BufferPoolState } from "./context.js";
import type { SnippetFn } from "./snippets.js";
import { runGraph } from "./scheduler.js";
import { summarize } from "./stats.js";
import { uploadFloat32 } from "../gpu/buffers.js";
import { KumaManifestError } from "../errors.js";
import { findLinearWeightElisions } from "../ops/index.js";

// allclose-style tolerance: |actual - golden| <= ATOL + RTOL * |golden|. Loose enough to
// absorb ordinary float32 reduction-order differences between a GPU kernel and eager
// CPU PyTorch (DFT matmuls especially), tight enough to still catch a real bug -- the
// kind of divergence we've been chasing by hand shows up as differences of 10-1000x, not
// a few ULPs.
//
// ATOL=4e-4 (not 1e-4): measured directly against the tiled linear.wgsl + 256-thread
// groupnorm.wgsl rewrite, whose worst observed deviations from golden were ~2x the old
// 1e-4 on small-magnitude (~1e-3) intermediate values, where ATOL is the dominant term
// anyway (RTOL*|golden| is negligible at that scale) -- summing the same K terms in
// 16-wide tiles, or reducing 256 elements at a time instead of 64, is a different but
// equally valid float32 summation order from both the old kernels and from whatever
// order PyTorch's own BLAS uses, and floating-point addition isn't associative. This
// barely loosens the check for larger-magnitude values, where RTOL already dominates.
const RTOL = 0.02;
const ATOL = 4e-4;

function approxEqual(golden: number, actual: number): boolean {
  if (Number.isNaN(golden) || Number.isNaN(actual)) return Number.isNaN(golden) === Number.isNaN(actual);
  if (!Number.isFinite(golden) || !Number.isFinite(actual)) return golden === actual;
  return Math.abs(actual - golden) <= ATOL + RTOL * Math.abs(golden);
}

export interface FieldDiff {
  field: "n" | "finite" | "mean" | "min" | "max" | `first[${number}]` | `spread[${number}]`;
  golden: number;
  actual: number;
}

export interface NodeDiff {
  node: string;
  part: "re" | "im";
  diffs: FieldDiff[];
}

export interface BranchVerifyReport {
  branch: number;
  nodesChecked: number;
  /** In golden.json but never resolved during this run -- the node either doesn't
   * exist under this name anymore, or the graph never reached it. Either way, not the
   * same kind of finding as a value mismatch, reported separately. */
  nodesMissing: string[];
  /** Every node with at least one mismatching field, in graph dispatch order -- the
   * first entry is the most useful one (everything after it may just be inherited). */
  mismatches: NodeDiff[];
}

export interface VerifyReport {
  ok: boolean;
  branches: BranchVerifyReport[];
}

function inputShapeFor(manifest: KumaManifest, name: string, length: number): number[] {
  const placeholder = manifest.graph.nodes.find((n) => n.op === "placeholder" && n.name === name);
  const shape = placeholder?.meta.shape;
  if (shape) {
    const n = shape.reduce((a, b) => a * b, 1);
    if (n === length) return shape;
  }
  return [length];
}

function compareTensor(node: string, part: "re" | "im", golden: GoldenTensorStats, actualData: Float32Array): NodeDiff | undefined {
  const diffs: FieldDiff[] = [];
  if (golden.n !== actualData.length) {
    diffs.push({ field: "n", golden: golden.n, actual: actualData.length });
    return { node, part, diffs };
  }

  const actual = summarize(actualData, golden.first.length);
  // Exact, not approxEqual -- a NaN/Inf appearing where golden has none (or vice
  // versa) is exactly the class of bug this project has hit before (see norm.ts's eps
  // fallback history), and ordinary float32 reduction-order differences between a GPU
  // kernel and eager PyTorch don't flip a normal finite value into NaN/Infinity or
  // back -- that takes an actual operation like 0/0 or overflow, not summation order.
  // mean/min/max alone can miss this: a handful of stray NaNs in an otherwise-large,
  // otherwise-correct-looking tensor can hide inside `mean` (NaN poisons it, which
  // would already be caught) but a small *count* mismatch with finite values still in
  // range elsewhere isn't guaranteed to surface through first[]/spread[]'s sampling.
  if (golden.finite !== actual.finite) diffs.push({ field: "finite", golden: golden.finite, actual: actual.finite });
  if (!approxEqual(golden.mean, actual.mean)) diffs.push({ field: "mean", golden: golden.mean, actual: actual.mean });
  if (!approxEqual(golden.min, actual.min)) diffs.push({ field: "min", golden: golden.min, actual: actual.min });
  if (!approxEqual(golden.max, actual.max)) diffs.push({ field: "max", golden: golden.max, actual: actual.max });

  golden.first.forEach((expected, i) => {
    const got = actual.first[i]!;
    if (!approxEqual(expected, got)) diffs.push({ field: `first[${i}]`, golden: expected, actual: got });
  });
  golden.spread_indices.forEach((flatIdx, i) => {
    const expected = golden.spread[i]!;
    const got = actualData[flatIdx]!;
    if (!approxEqual(expected, got)) diffs.push({ field: `spread[${i}]`, golden: expected, actual: got });
  });

  return diffs.length > 0 ? { node, part, diffs } : undefined;
}

export interface VerifyContext {
  device: GPUDevice;
  kernels: ReadonlyMap<string, string>;
  pipelineCache: Map<string, GPUComputePipeline>;
  weightBuffers: ReadonlyMap<string, ResolvedTensor>;
  snippets: ReadonlyMap<string, string>;
  snippetCache: Map<string, SnippetFn>;
  constantCache: Map<string, GPUBuffer>;
  bufferPool: BufferPoolState;
}

/** Runs one branch's subgraph directly against golden's own recorded reference input(s)
 * -- bypassing the time-based router entirely, so there's no need to reverse-engineer
 * which global `t` would have routed here. `nodes` must already include the `output`
 * node (the manifest's own graph already does; a switch branch's `nodes` list doesn't,
 * by construction -- see kuma.branching.compile_branching -- so the caller synthesizes
 * one referencing `branch.output`). */
async function verifyBranch(
  ctx: VerifyContext,
  branchIndex: number,
  manifest: KumaManifest,
  golden: GoldenBranch,
): Promise<BranchVerifyReport> {
  const inputBuffers = new Map<string, ResolvedTensor>();
  const rawInputs = new Map<string, Float32Array>();
  for (const [name, values] of Object.entries(golden.inputs)) {
    const data = new Float32Array(values);
    const shape = inputShapeFor(manifest, name, data.length);
    rawInputs.set(name, data);
    inputBuffers.set(name, { buffer: uploadFloat32(ctx.device, data), shape });
  }

  const ignoredNodes = new Set(findLinearWeightElisions(manifest.graph.nodes).keys());
  const captureNodes = new Set(Object.keys(golden.nodes).filter((name) => !ignoredNodes.has(name)));
  const outputs = await runGraph({
    device: ctx.device,
    manifest,
    kernels: ctx.kernels,
    pipelineCache: ctx.pipelineCache,
    weightBuffers: ctx.weightBuffers,
    inputBuffers,
    rawInputs,
    snippets: ctx.snippets,
    snippetCache: ctx.snippetCache,
    constantCache: ctx.constantCache,
    bufferPool: ctx.bufferPool,
    captureNodes,
  });

  // Diagnostic: log the first few captured values to help narrow down zeros-in-verify.
  // The first non-output entry in `outputs` is capturedTensors[0] (earliest golden node
  // in graph order). If this is already 0.0 for a node that should be non-zero, the
  // issue is either in dispatch or in readback, not in downstream propagation.
  if (outputs.length > 1) {
    const sampleNames = outputs.slice(1, 4).map((o) => o.name);
    const sampleMeans = outputs.slice(1, 4).map((o) => {
      if (!o.data.length) return "empty";
      const sum = Array.from(o.data).reduce((a, b) => a + b, 0);
      return (sum / o.data.length).toFixed(6);
    });
    console.log(
      `[kuma verify] branch ${branchIndex}: ${outputs.length - 1} captures, ` +
      `input keys=[${Object.keys(golden.inputs).join(",")}], ` +
      `first 3 captured nodes: ${sampleNames.map((n, i) => `${n}=mean(${sampleMeans[i]})`).join(", ")}`,
    );
  }

  const byName = new Map(outputs.map((o) => [o.name, o]));
  const mismatches: NodeDiff[] = [];
  const nodesMissing: string[] = [];
  let nodesChecked = 0;
  const graphOrder = new Map(manifest.graph.nodes.map((node, i) => [node.name, i]));
  const goldenEntries = Object.entries(golden.nodes).sort(
    ([a], [b]) => (graphOrder.get(a) ?? Number.MAX_SAFE_INTEGER) - (graphOrder.get(b) ?? Number.MAX_SAFE_INTEGER),
  );

  for (const [name, stats] of goldenEntries) {
    if (ignoredNodes.has(name)) {
      continue;
    }
    const actual = byName.get(name);
    if (!actual) {
      nodesMissing.push(name);
      continue;
    }
    nodesChecked++;

    const reDiff = compareTensor(name, "re", stats.re, actual.data);
    if (reDiff) mismatches.push(reDiff);

    if (stats.im) {
      if (!actual.imag) {
        mismatches.push({ node: name, part: "im", diffs: [{ field: "n", golden: stats.im.n, actual: 0 }] });
      } else {
        const imDiff = compareTensor(name, "im", stats.im, actual.imag);
        if (imDiff) mismatches.push(imDiff);
      }
    }
  }

  return { branch: branchIndex, nodesChecked, nodesMissing, mismatches };
}

/** Synthesizes a runnable single-branch "manifest" wrapping a switch branch's bare node
 * list (placeholders + call_function, no output node -- see kuma.branching's docs) with
 * the output node it's missing, referencing the branch's own declared output. */
function manifestForSwitchBranch(branchIndex: number, nodes: GraphNode[], outputRef: { node_ref: string }): KumaManifest {
  const outputNode: GraphNode = {
    id: -1,
    name: "output",
    op: "output",
    target: "output",
    args: [[outputRef]],
    kwargs: {},
    meta: {},
  };
  return {
    format: "kuma",
    format_version: 0,
    weight_file: "",
    endianness: "little",
    inputs: [],
    outputs: [{ name: `branch${branchIndex}_output` }],
    weights: [],
    graph: { node_count: nodes.length + 1, op_counts: {}, nodes: [...nodes, outputNode] },
    warnings: [],
    unsupported_ops: [],
  };
}

/** Verifies a loaded model's actual computed values against golden.json (if the .iph
 * package had one) -- for each branch (or the one top-level graph, for a non-branching
 * model), runs it directly against golden's own recorded reference input and compares
 * every node golden.json has an entry for. See types/golden.ts for what's compared. */
export async function verifyAgainstGolden(ctx: VerifyContext, manifest: KumaManifest, golden: GoldenData): Promise<VerifyReport> {
  const switchNode = manifest.graph.nodes.find((n) => n.op === "switch");
  const branches: BranchVerifyReport[] = [];

  if (!switchNode) {
    if (golden.branches.length !== 1) {
      throw new KumaManifestError(
        `golden.json has ${golden.branches.length} branch(es) but this manifest has no switch node (expected exactly 1).`,
      );
    }
    branches.push(await verifyBranch(ctx, 0, manifest, golden.branches[0]!));
  } else {
    const switchBranches = switchNode.branches ?? [];
    for (let i = 0; i < golden.branches.length; i++) {
      const switchBranch = switchBranches[i];
      if (!switchBranch) {
        throw new KumaManifestError(`golden.json has a branch ${i} but the manifest's switch node only has ${switchBranches.length}.`);
      }
      const syntheticManifest = manifestForSwitchBranch(i, switchBranch.nodes, switchBranch.output);
      branches.push(await verifyBranch(ctx, i, syntheticManifest, golden.branches[i]!));
    }
  }

  const ok = branches.every((b) => b.mismatches.length === 0 && b.nodesMissing.length === 0);
  return { ok, branches };
}

async function warmUpBranch(ctx: VerifyContext, manifest: KumaManifest, golden: GoldenBranch): Promise<void> {
  const inputBuffers = new Map<string, ResolvedTensor>();
  const rawInputs = new Map<string, Float32Array>();
  for (const [name, values] of Object.entries(golden.inputs)) {
    const data = new Float32Array(values);
    const shape = inputShapeFor(manifest, name, data.length);
    rawInputs.set(name, data);
    inputBuffers.set(name, { buffer: uploadFloat32(ctx.device, data), shape });
  }

  await runGraph({
    device: ctx.device,
    manifest,
    kernels: ctx.kernels,
    pipelineCache: ctx.pipelineCache,
    weightBuffers: ctx.weightBuffers,
    inputBuffers,
    rawInputs,
    snippets: ctx.snippets,
    snippetCache: ctx.snippetCache,
    constantCache: ctx.constantCache,
    bufferPool: ctx.bufferPool,
    skipOutputReadback: true,
  });
  // runGraph keeps the declared-output buffer alive for the caller by design (it has
  // no way to know this call only cares about the side effect of warming up
  // pipelines/pool/constants) -- this used to explicitly destroy it to avoid a leak,
  // back when every output was a fresh, unpooled allocation. Now that node outputs
  // come from ctx.bufferPool (see context.ts's BufferPoolState), that buffer is the
  // *same* GPUBuffer object this branch's pool will keep reusing forever -- destroying
  // it here would corrupt the pool for every future call against this branch, not
  // free anything. Just let it be (warmUp only ever runs once per branch anyway, so
  // there's no repeated-leak concern the old comment was guarding against).
}

/** Runs every branch a model has once, directly against golden's own recorded
 * reference input (bypassing the router, same setup as verifyBranch) -- purely for
 * the side effect of populating pipelineCache/constantCache for every kernel/shape
 * every branch could possibly need, before the user starts interacting with the
 * model. Without this, the *first* time a given branch (or a given kernel-routing
 * decision within it -- e.g. ops/conv2d.ts's depthwise/pointwise/general split, which
 * different branches can hit differently depending on their own channel counts)
 * actually gets exercised during interactive playback -- e.g. scrubbing across a
 * segment boundary into a branch that's never rendered yet -- pays for real shader
 * compilation right then: a genuine, user-visible stutter landing at exactly the
 * worst possible moment. Output values are discarded (skipOutputReadback) since
 * correctness isn't what this is for -- see verifyAgainstGolden for that. No-op for a
 * model with no golden.json, since there's nothing this could warm up that a first
 * real run() call wouldn't anyway. */
export async function warmUp(ctx: VerifyContext, manifest: KumaManifest, golden: GoldenData): Promise<void> {
  const switchNode = manifest.graph.nodes.find((n) => n.op === "switch");

  if (!switchNode) {
    const branch = golden.branches[0];
    if (branch) await warmUpBranch(ctx, manifest, branch);
    return;
  }

  const switchBranches = switchNode.branches ?? [];
  for (let i = 0; i < golden.branches.length; i++) {
    const switchBranch = switchBranches[i];
    const goldenBranch = golden.branches[i];
    if (!switchBranch || !goldenBranch) continue;
    const syntheticManifest = manifestForSwitchBranch(i, switchBranch.nodes, switchBranch.output);
    await warmUpBranch(ctx, syntheticManifest, goldenBranch);
  }
}
