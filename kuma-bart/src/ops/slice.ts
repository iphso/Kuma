import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { clampSliceBounds, normalizeDim, numElements, outerInner } from "../engine/shape.js";

function dispatchSlice(
  ctx: OpContext,
  inputBuffer: GPUBuffer,
  inputShape: readonly number[],
  dimArg: number,
  startArg: number | null,
  endArg: number | null,
  stepArg: number | null | undefined,
): { buffer: GPUBuffer; shape: number[] } {
  const dim = normalizeDim(dimArg, inputShape.length);
  const dimSize = inputShape[dim]!;
  const { start, end } = clampSliceBounds(startArg, endArg, dimSize);
  const step = stepArg ?? 1;
  const outExtent = step > 0 ? Math.max(0, Math.ceil((end - start) / step)) : 0;

  const outShape = inputShape.slice();
  outShape[dim] = outExtent;
  const { outer, inner } = outerInner(inputShape, dim);
  const n = numElements(outShape);

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([outer, inner, dimSize, outExtent, start, step, n]);
  ctx.dispatchKernel("slice.wgsl", [inputBuffer, out, params], n);
  return { buffer: out, shape: outShape };
}

/** aten.slice.Tensor args: (input, dim, start, end, step). PyTorch FX commonly encodes
 * "to the end" as a huge sentinel int for `end` and `null` for default start/end. */
export function sliceHandler(ctx: OpContext): void {
  const node = ctx.node;
  const args = node.args;
  const inputRef = args[0] as ArgValue;
  const dimArg = args[1] as number;
  const startArg = args[2] as number | null;
  const endArg = args[3] as number | null;
  const stepArg = args[4] as number | null | undefined;
  const input = ctx.resolve(inputRef);
  const result = dispatchSlice(ctx, input.buffer, input.shape, dimArg, startArg, endArg, stepArg);

  let outImag: GPUBuffer | undefined;
  if (input.imag) {
    outImag = dispatchSlice(ctx, input.imag, input.shape, dimArg, startArg, endArg, stepArg).buffer;
  }

  ctx.setOutput(result.buffer, result.shape, outImag);
}

/** aten.slice_multi.default args: (input, axes, starts, ends, steps).
 * ONNX Slice can describe several axes at once; Kuma's kernel slices one axis at a
 * time, so compose those static slices in ONNX's provided order. */
export function sliceMultiHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, axes, starts, ends, steps] = node.args as [ArgValue, number[], number[], number[], number[]];
  const input = ctx.resolve(inputRef);

  let buffer = input.buffer;
  let imag = input.imag;
  let shape = input.shape;

  for (let i = 0; i < axes.length; i++) {
    const result = dispatchSlice(ctx, buffer, shape, axes[i]!, starts[i]!, ends[i]!, steps[i]);
    buffer = result.buffer;
    if (imag) {
      imag = dispatchSlice(ctx, imag, shape, axes[i]!, starts[i]!, ends[i]!, steps[i]).buffer;
    }
    shape = result.shape;
  }

  ctx.setOutput(buffer, node.meta.shape ?? shape, imag);
}
