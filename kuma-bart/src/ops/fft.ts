import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { dispatchPermute } from "./permute.js";
import { dispatchLinear } from "./linear.js";
import { complexIfftBasis, irfftBasis } from "../engine/dft.js";
import { complexProduct, dispatchBinary } from "./complex_math.js";

function moveAxisToLast(ctx: OpContext, tensor: ResolvedTensor, axis: number): { result: ResolvedTensor; swapDims: number[] } {
  const rank = tensor.shape.length;
  const dims = Array.from({ length: rank }, (_, i) => i);
  if (axis === rank - 1) {
    return { result: tensor, swapDims: dims };
  }
  dims[axis] = rank - 1;
  dims[rank - 1] = axis;
  const outShape = dims.map((d) => tensor.shape[d]!);
  const result = dispatchPermute(ctx, tensor.buffer, tensor.shape, outShape, dims);
  return { result, swapDims: dims };
}

/** A single-axis swap is its own inverse — applying the same `swapDims` again restores
 * the original axis order. */
function moveAxisBack(ctx: OpContext, tensor: ResolvedTensor, swapDims: number[]): ResolvedTensor {
  const outShape = swapDims.map((d) => tensor.shape[d]!);
  return dispatchPermute(ctx, tensor.buffer, tensor.shape, outShape, swapDims);
}

function matmulBasis(ctx: OpContext, x: ResolvedTensor, basis: ResolvedTensor): ResolvedTensor {
  const n = basis.shape[0]!;
  const zeroBias = { buffer: ctx.getOrCreateZeroBuffer(`zeroBias:${n}`, n * 4), shape: [n] };
  return dispatchLinear(ctx, x, basis, zeroBias);
}

/** Length-preserving complex IFFT ('ortho') along one axis, every other axis held
 * fixed — used for fft_irfft2's non-halved axis (must run *before* the real/Hermitian
 * axis below, since that one collapses complex -> real). */
function complexIfftAlongAxis(ctx: OpContext, real: ResolvedTensor, imag: ResolvedTensor, axis: number): { real: ResolvedTensor; imag: ResolvedTensor } {
  const n = real.shape[axis]!;
  // The bases depend only on the transform length, so they're cached for the model's
  // lifetime (see OpContext.getOrUploadConstant).
  const cosBuffer = ctx.getOrUploadConstant(`complexIfft:cos:${n}`, () => complexIfftBasis(n).cos);
  const sinBuffer = ctx.getOrUploadConstant(`complexIfft:sin:${n}`, () => complexIfftBasis(n).sin);
  const cosTensor: ResolvedTensor = { buffer: cosBuffer, shape: [n, n] };
  const sinTensor: ResolvedTensor = { buffer: sinBuffer, shape: [n, n] };

  const { result: realMoved, swapDims } = moveAxisToLast(ctx, real, axis);
  const { result: imagMoved } = moveAxisToLast(ctx, imag, axis);

  const k = realMoved.shape[realMoved.shape.length - 1]!;
  const m = realMoved.shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const real2d = { buffer: realMoved.buffer, shape: [m, k] };
  const imag2d = { buffer: imagMoved.buffer, shape: [m, k] };

  // (cos + i·sin) ⊗ (Xr + i·Xi): Re = cos@Xr - sin@Xi, Im = sin@Xr + cos@Xi (see engine/dft.ts).
  const basisTimes = (basis: ResolvedTensor, x: ResolvedTensor) => matmulBasis(ctx, x, basis).buffer;
  const { re: outRe, im: outIm } = complexProduct(ctx, { re: cosTensor, im: sinTensor }, { re: real2d, im: imag2d }, basisTimes, [m, n]);

  const movedShape = [...realMoved.shape.slice(0, -1), n];
  const reRestored = moveAxisBack(ctx, { buffer: outRe, shape: movedShape }, swapDims);
  const imRestored = moveAxisBack(ctx, { buffer: outIm!, shape: movedShape }, swapDims);
  return { real: reRestored, imag: imRestored };
}

/** Real/Hermitian-reconstruction IFFT ('ortho') along the last axis: whalf complex bins
 * -> outputLength real values. Must run last (collapses complex -> real). */
function realIfftLastAxis(ctx: OpContext, real: ResolvedTensor, imag: ResolvedTensor, outputLength: number): ResolvedTensor {
  const whalf = real.shape[real.shape.length - 1]!;
  const aBuffer = ctx.getOrUploadConstant(`irfft:a:${outputLength}`, () => irfftBasis(outputLength).a);
  const bBuffer = ctx.getOrUploadConstant(`irfft:b:${outputLength}`, () => irfftBasis(outputLength).b);
  const aTensor: ResolvedTensor = { buffer: aBuffer, shape: [outputLength, whalf] };
  const bTensor: ResolvedTensor = { buffer: bBuffer, shape: [outputLength, whalf] };

  const m = real.shape.slice(0, -1).reduce((x, y) => x * y, 1);
  const real2d = { buffer: real.buffer, shape: [m, whalf] };
  const imag2d = { buffer: imag.buffer, shape: [m, whalf] };

  const aXr = matmulBasis(ctx, real2d, aTensor);
  const bXi = matmulBasis(ctx, imag2d, bTensor);
  const outShape = [...real.shape.slice(0, -1), outputLength];
  return { buffer: dispatchBinary(ctx, "add.wgsl", aXr.buffer, bXi.buffer, outShape), shape: outShape };
}

/**
 * aten.fft_irfft2.default args: (input, s, dim, norm). Hard-matched against the one
 * pattern this model uses — dim=[-2,-1] (axis -2: full complex IFFT; axis -1, listed
 * last: the real/Hermitian-halved axis), norm='ortho' — anything else fails loudly
 * rather than guessing. `s` (explicit output size) is ignored in favor of this node's
 * own authoritative manifest output shape, which already reflects whatever `s` would
 * have produced.
 */
export function fftIrfft2Handler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, , dimArg, normArg] = node.args as [ArgValue, ArgValue, number[], string];

  if (!Array.isArray(dimArg) || dimArg.length !== 2) {
    throw new KumaUnsupportedOpError(node.target, node.name, `expected dim=[axis,axis], got ${JSON.stringify(dimArg)}.`);
  }
  if (normArg !== "ortho") {
    throw new KumaUnsupportedOpError(node.target, node.name, `only norm="ortho" is supported, got "${normArg}".`);
  }

  const input = ctx.resolve(inputRef);
  if (!input.imag) {
    throw new KumaShapeError(
      `Op "${node.target}" (node "${node.name}") expected a complex input (with an imaginary part).`,
    );
  }

  const rank = input.shape.length;
  const complexAxis = dimArg[0]! < 0 ? dimArg[0]! + rank : dimArg[0]!;
  const realAxis = dimArg[1]! < 0 ? dimArg[1]! + rank : dimArg[1]!;
  if (realAxis !== rank - 1) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `expected the real/Hermitian axis (dim[1]) to be the last axis, got axis ${realAxis} of rank ${rank}.`,
    );
  }

  const outShape = node.meta.shape;
  if (!outShape) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") is missing an output shape in the manifest.`);
  }
  const outputLength = outShape[realAxis]!;

  const step1 = complexIfftAlongAxis(
    ctx,
    { buffer: input.buffer, shape: input.shape },
    { buffer: input.imag, shape: input.shape },
    complexAxis,
  );
  const result = realIfftLastAxis(ctx, step1.real, step1.imag, outputLength);
  ctx.setOutput(result.buffer, outShape);
}
