import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";

function normalizeDims(dims: readonly number[], rank: number): number[] {
  return dims.map((d) => (d < 0 ? d + rank : d)).sort((a, b) => a - b);
}

/** Dispatches sum.wgsl over a contiguous block of axes (sum.wgsl's own constraint —
 * the reduced axes must be a contiguous memory block). Returns a flat `[n]`-shaped
 * result; callers that need the "real" multi-dim shape use their own node's
 * authoritative manifest shape instead of trying to recompute it here. */
function dispatchSum(
  ctx: OpContext,
  target: string,
  nodeName: string,
  inputBuffer: GPUBuffer,
  inputShape: readonly number[],
  dimArg: readonly number[],
): { buffer: GPUBuffer; n: number } {
  const dims = normalizeDims(dimArg, inputShape.length);
  for (let i = 1; i < dims.length; i++) {
    if (dims[i] !== dims[i - 1]! + 1) {
      throw new KumaUnsupportedOpError(
        target,
        nodeName,
        `sum.wgsl can only reduce a contiguous block of axes, got dims=${JSON.stringify(dimArg)} on a rank-${inputShape.length} tensor.`,
      );
    }
  }
  const first = dims[0]!;
  const last = dims[dims.length - 1]!;
  const outer = numElements(inputShape.slice(0, first));
  const reduce = numElements(inputShape.slice(first, last + 1));
  const inner = numElements(inputShape.slice(last + 1));
  const n = outer * inner;

  const out = ctx.createBuffer([n]);
  const params = ctx.uniform([outer, reduce, inner, n]);
  ctx.dispatchKernel("sum.wgsl", [inputBuffer, out, params], n);
  return { buffer: out, n };
}

/** aten.sum.dim_IntList args: (input, dim, keepdim, dtype=None). */
export function sumHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, dimArg] = node.args as [ArgValue, number[]];
  const input = ctx.resolve(inputRef);
  const { buffer } = dispatchSum(ctx, node.target, node.name, input.buffer, input.shape, dimArg);

  const outShape = node.meta.shape;
  if (!outShape) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") is missing an output shape in the manifest.`);
  }
  const imag = input.imag
    ? dispatchSum(ctx, node.target, node.name, input.imag, input.shape, dimArg).buffer
    : undefined;
  ctx.setOutput(buffer, outShape, imag);
}

/** aten.linalg_vector_norm.default args: (input, ord, dim, keepdim, dtype=None).
 * Composed from existing kernels: square (mul self) -> sum -> sqrt. Only ord=2 (L2
 * norm) is supported — the only value seen in this model — anything else fails loudly
 * rather than guessing a generalized p-norm implementation. */
export function linalgVectorNormHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, ordArg, dimArg] = node.args as [ArgValue, number, number[]];
  if (ordArg !== 2) {
    throw new KumaUnsupportedOpError(node.target, node.name, `only ord=2 (L2 norm) is supported, got ord=${ordArg}.`);
  }
  const input = ctx.resolve(inputRef);
  const n0 = numElements(input.shape);

  const squared = ctx.createBuffer([n0]);
  const squareParams = ctx.uniform([n0]);
  ctx.dispatchKernel("mul.wgsl", [input.buffer, input.buffer, squared, squareParams], n0);

  const { buffer: summed } = dispatchSum(ctx, node.target, node.name, squared, input.shape, dimArg);

  const outShape = node.meta.shape;
  if (!outShape) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") is missing an output shape in the manifest.`);
  }
  const n = numElements(outShape);
  const out = ctx.createBuffer(outShape);
  const sqrtParams = ctx.uniform([n]);
  ctx.dispatchKernel("sqrt.wgsl", [summed, out, sqrtParams], n);
  ctx.setOutput(out, outShape);
}
