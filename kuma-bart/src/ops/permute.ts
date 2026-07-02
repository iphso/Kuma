import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { contiguousStrides, numElements } from "../engine/shape.js";

const RANK_CAP = 4;

/** Shared by permute, transpose, expand, and the Tucker mode-product chain (ops/
 * einsum.ts): all reduce to "gather output element `i` from
 * `input[sum(coord[d] * inStrides[d])]`", padded to rank 4 with extent=1/stride=0 at the
 * front (matching permute.wgsl's fixed vec4 Params layout). Returns the result rather
 * than calling ctx.setOutput, since callers like the mode-product chain dispatch this
 * several times per node and only want the *last* result registered. */
export function dispatchPermute(
  ctx: OpContext,
  inputBuffer: GPUBuffer,
  inputShape: readonly number[],
  outShape: readonly number[],
  dimsForOutputAxis: readonly number[],
): ResolvedTensor {
  if (outShape.length > RANK_CAP) {
    throw new KumaUnsupportedOpError(
      ctx.node.target,
      ctx.node.name,
      `permute.wgsl supports rank <= ${RANK_CAP}, got rank ${outShape.length}.`,
    );
  }
  const inStrides = contiguousStrides(inputShape);
  const pad = RANK_CAP - outShape.length;
  const paddedOutShape = new Array(pad).fill(1).concat(outShape);
  const paddedInStrides = new Array(pad).fill(0).concat(dimsForOutputAxis.map((d) => inStrides[d]!));
  const n = numElements(outShape);

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([...paddedOutShape, ...paddedInStrides, n]);
  ctx.dispatchKernel("permute.wgsl", [inputBuffer, out, params], n);
  return { buffer: out, shape: [...outShape] };
}

/** Permutes a tensor's real part, and — if it's complex-paired — its imaginary part
 * identically (same dims/shape; the two parts are always geometrically in lockstep).
 * dispatchPermute itself only ever sees one plain GPUBuffer, by design (it's also used
 * standalone, e.g. by ops/expand.ts, against a buffer that's never complex) — this is
 * the complex-aware wrapper for callers (the handlers below, einsum's pairwise-
 * contraction permutes) that need to permute a full ResolvedTensor and not lose the
 * imaginary half in the process. */
export function permuteMaybeComplex(
  ctx: OpContext,
  tensor: ResolvedTensor,
  outShape: readonly number[],
  dims: readonly number[],
): ResolvedTensor {
  const re = dispatchPermute(ctx, tensor.buffer, tensor.shape, outShape, dims);
  if (!tensor.imag) return re;
  const im = dispatchPermute(ctx, tensor.imag, tensor.shape, outShape, dims);
  return { buffer: re.buffer, shape: re.shape, imag: im.buffer };
}

export function permuteHandler(ctx: OpContext): void {
  const [inputRef, dimsArg] = ctx.node.args as [ArgValue, number[]];
  const input = ctx.resolve(inputRef);
  const rank = input.shape.length;
  const dims = dimsArg.map((d) => (d < 0 ? d + rank : d));
  const outShape = dims.map((d) => input.shape[d]!);
  const result = permuteMaybeComplex(ctx, input, outShape, dims);
  ctx.setOutput(result.buffer, result.shape, result.imag);
}

export function transposeHandler(ctx: OpContext): void {
  const [inputRef, rawDim0Arg, rawDim1Arg] = ctx.node.args as [ArgValue, number | undefined, number | undefined];
  const input = ctx.resolve(inputRef);
  const rank = input.shape.length;
  const dim0Arg = rawDim0Arg ?? -2;
  const dim1Arg = rawDim1Arg ?? -1;
  const dim0 = dim0Arg < 0 ? dim0Arg + rank : dim0Arg;
  const dim1 = dim1Arg < 0 ? dim1Arg + rank : dim1Arg;

  const dims = Array.from({ length: rank }, (_, i) => i);
  const tmp = dims[dim0]!;
  dims[dim0] = dims[dim1]!;
  dims[dim1] = tmp;

  const outShape = dims.map((d) => input.shape[d]!);
  const result = permuteMaybeComplex(ctx, input, outShape, dims);
  ctx.setOutput(result.buffer, result.shape, result.imag);
}
