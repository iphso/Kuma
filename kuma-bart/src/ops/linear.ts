import type { OpContext, ResolvedTensor } from "../engine/context.js";
import { isNodeRef, type ArgValue, type GraphNode } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";

/** Core matmul dispatch shared by linearHandler and the Tucker mode-product chain
 * (ops/einsum.ts): out(M,N) = x(...,K) @ weight(N,K)^T + bias(N). `x` may be >2D (e.g.
 * nn.Linear applied channel-last to a [B,H,W,C] tensor) — every leading dim collapses
 * into M (free reshape, already contiguous), only the last dim is the contracted K.
 * Doesn't call ctx.setOutput — callers that chain several of these per node (einsum)
 * only want the *last* result registered, and the caller's own node.meta.shape is the
 * authoritative (possibly >2D) shape to relabel the result with anyway. */
const LINEAR_TILE = 64;

export function dispatchLinear(ctx: OpContext, x: ResolvedTensor, weight: ResolvedTensor, bias: ResolvedTensor): ResolvedTensor {
  const k = x.shape[x.shape.length - 1]!;
  const m = numElements(x.shape) / k;
  const n = weight.shape[0]!;
  const outShape = [m, n];
  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([m, k, n]);
  // linear.wgsl is a 64x64-tiled matmul -- one workgroup per output tile, read directly
  // via workgroup_id, not folded into a linear dispatchElements count (see its header).
  const gridX = Math.ceil(n / LINEAR_TILE);
  const gridY = Math.ceil(m / LINEAR_TILE);
  ctx.dispatchKernelGrid("linear.wgsl", [x.buffer, weight.buffer, bias.buffer, out, params], gridX, gridY);
  return { buffer: out, shape: outShape };
}

const TRANSPOSE_TARGETS = new Set(["aten.t.default", "aten.transpose.int", "aten.permute.default"]);
const PASSTHROUGH_TARGETS = new Set([
  "aten.alias.default",
  "aten.contiguous.default",
  "aten.detach.default",
  "aten.detach_.default",
  "aten.lift_fresh_copy.default",
  "aten.to.device",
  "aten.to.dtype",
]);

/** Index of the weight-like argument for each matmul target — addmm(bias, x, weight),
 * mm(x, weight). Elision only applies when the transpose node is bound at exactly this
 * position; if a "weight" placeholder transpose ended up feeding `x` instead, eliding it
 * would silently swap in the wrong tensor instead of failing loudly. */
const WEIGHT_ARG_INDEX: Record<string, number> = {
  "aten.addmm.default": 2,
  "aten.mm.default": 1,
  "aten.matmul.default": 1,
};

function collectNodeRefs(arg: ArgValue, into: Set<string>): void {
  if (isNodeRef(arg)) {
    into.add(arg.node_ref);
    return;
  }
  if (Array.isArray(arg)) {
    for (const a of arg) collectNodeRefs(a, into);
  }
}

function tracePassthroughSource(nodesByName: ReadonlyMap<string, GraphNode>, ref: string): string {
  let current = ref;
  const seen = new Set<string>();
  while (!seen.has(current)) {
    seen.add(current);
    const node = nodesByName.get(current);
    if (!node || node.op !== "call_function" || !PASSTHROUGH_TARGETS.has(node.target)) break;
    const next = node.args[0];
    if (!next || !isNodeRef(next)) break;
    current = next.node_ref;
  }
  return current;
}

/**
 * UNVERIFIED (no Docker access to generate a real Linear-bearing manifest this session
 * — see kuma-bart's plan notes): nn.Linear may decompose to `addmm(bias, x, t(weight))`
 * rather than a fused op, and there is no WGSL kernel for standalone `t`/`transpose`/
 * `permute`. linear.wgsl already expects weight in nn.Linear's native
 * (out_features, in_features) layout and does the transpose arithmetically inside the
 * shader, so when a transpose node's *sole* consumer is an addmm/mm and its own input is
 * a literal weight parameter, we elide the transpose: the matmul binds the original
 * weight buffer directly and the transpose node is never dispatched. Anything else (a
 * transpose used elsewhere, or feeding a non-weight tensor) is left alone and will fail
 * loudly as an unsupported op if actually dispatched, rather than silently mis-executing.
 */
export function findLinearWeightElisions(nodes: readonly GraphNode[]): Map<string, string> {
  const nodesByName = new Map(nodes.map((n) => [n.name, n]));
  const consumers = new Map<string, GraphNode[]>();
  const refs = new Set<string>();

  for (const n of nodes) {
    refs.clear();
    for (const a of n.args) collectNodeRefs(a, refs);
    for (const v of Object.values(n.kwargs)) collectNodeRefs(v, refs);
    for (const ref of refs) {
      const list = consumers.get(ref) ?? [];
      list.push(n);
      consumers.set(ref, list);
    }
  }

  const elisions = new Map<string, string>();
  for (const n of nodes) {
    if (n.op !== "call_function" || !TRANSPOSE_TARGETS.has(n.target)) continue;
    const sourceArg = n.args[0];
    if (!sourceArg || !isNodeRef(sourceArg)) continue;
    const sourceRef = tracePassthroughSource(nodesByName, sourceArg.node_ref);
    const source = nodesByName.get(sourceRef);
    if (!source || source.op !== "placeholder" || (source.kind !== "parameter" && source.kind !== "buffer")) continue;

    const myConsumers = consumers.get(n.name) ?? [];
    if (myConsumers.length !== 1) continue;
    const consumer = myConsumers[0]!;
    const weightIdx = WEIGHT_ARG_INDEX[consumer.target];
    if (weightIdx === undefined) continue;
    const consumerWeightArg = consumer.args[weightIdx];
    if (!consumerWeightArg || !isNodeRef(consumerWeightArg) || consumerWeightArg.node_ref !== n.name) continue;

    // Runtime should alias the transpose's immediate input, not the traced-through
    // placeholder. The trace above only proves this is ultimately a weight/buffer; the
    // scheduler's liveness pass may have already released the original placeholder name
    // after a passthrough chain, while the immediate source remains the live alias.
    elisions.set(n.name, sourceArg.node_ref);
  }
  return elisions;
}

/** Handles aten.addmm.default / aten.mm.default / aten.linear.default — all dispatched
 * against linear.wgsl, which expects weight:(N,K) (nn.Linear's native layout). */
export function linearHandler(ctx: OpContext): void {
  const node = ctx.node;
  let xRef: ArgValue;
  let weightRef: ArgValue;
  let biasRef: ArgValue | undefined;

  if (node.target === "aten.addmm.default") {
    [biasRef, xRef, weightRef] = node.args as [ArgValue, ArgValue, ArgValue];
  } else if (node.target === "aten.mm.default" || node.target === "aten.matmul.default") {
    [xRef, weightRef] = node.args as [ArgValue, ArgValue];
  } else if (node.target === "aten.linear.default") {
    xRef = node.args[0] as ArgValue;
    weightRef = node.args[1] as ArgValue;
    biasRef = node.args[2] as ArgValue | undefined;
  } else {
    throw new KumaUnsupportedOpError(node.target, node.name);
  }

  const x = ctx.resolve(xRef);
  const weight = ctx.resolve(weightRef);

  if (weight.shape.length !== 2 || x.shape.length < 1 || x.shape[x.shape.length - 1] !== weight.shape[1]) {
    throw new KumaShapeError(
      `Op "${node.target}" (node "${node.name}") expected x:(...,K) and weight:(N,K) with matching K, got ` +
        `x=${JSON.stringify(x.shape)} weight=${JSON.stringify(weight.shape)}. If this came from nn.Linear, ` +
        `kuma-bart's weight-transpose elision didn't apply — see ops/linear.ts.`,
    );
  }

  const n = weight.shape[0]!;
  const bias =
    biasRef === undefined || biasRef === null ? { buffer: ctx.getOrCreateZeroBuffer(`zero-bias:${n}`, n * 4), shape: [n] } : ctx.resolve(biasRef);

  const result = dispatchLinear(ctx, x, weight, bias);
  const outShape = node.meta.shape ?? result.shape;
  ctx.setOutput(result.buffer, outShape);
}
