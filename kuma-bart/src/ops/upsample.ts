import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";
import { numElements } from "../engine/shape.js";

/** Output [N, C, out_h, out_w] for a 2D upsample node: the manifest's own shape when
 * present, else derived from `output_size` or `scale_factors` (the two ways
 * F.interpolate is traced). */
function upsampleOutShape(
  ctx: OpContext,
  inputShape: readonly number[],
  outputSize: number[] | null | undefined,
  scales: number[] | null | undefined,
): number[] {
  const node = ctx.node;
  if (node.meta.shape) return node.meta.shape;
  if (inputShape.length !== 4) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") expects a 4D NCHW input, got rank ${inputShape.length}.`);
  }
  const [n, c, h, w] = inputShape as [number, number, number, number];
  if (outputSize && outputSize.length === 2) return [n, c, outputSize[0]!, outputSize[1]!];
  if (scales && scales.length === 2) return [n, c, Math.floor(h * scales[0]!), Math.floor(w * scales[1]!)];
  throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") has neither an output shape nor output_size/scale_factors.`);
}

/** aten.upsample_nearest2d.vec args: (input, output_size|None, scale_factors|None);
 * aten.upsample_nearest2d.default args: (input, output_size, scales_h=None,
 * scales_w=None). Both run upsample_nearest2d.wgsl, whose source index is
 * floor(out_idx * in_size / out_size) -- PyTorch's `nearest` rule for the integer
 * factors used in practice (a grouped transposed conv with a 2x2 ones kernel, as in
 * the EDM2 U-Net 'up' resample, is exactly this at factor 2). */
export function upsampleNearest2dHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const outputSize = node.args[1] as number[] | null | undefined;
  const second = node.args[2];
  // `.vec` carries a [sh, sw] list; `.default` carries scalar scales_h then scales_w.
  const scales = Array.isArray(second)
    ? (second as number[])
    : typeof second === "number"
      ? [second, (node.args[3] as number | undefined) ?? second]
      : null;

  const input = ctx.resolve(inputRef);
  const outShape = upsampleOutShape(ctx, input.shape, outputSize, scales);
  if (input.shape.length !== 4 || outShape.length !== 4) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") expects 4D NCHW input/output shapes.`);
  }
  const [batch, channels, inH, inW] = input.shape as [number, number, number, number];
  const [, , outH, outW] = outShape as [number, number, number, number];

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([batch, channels, inH, inW, outH, outW]);
  ctx.dispatchKernel("upsample_nearest2d.wgsl", [input.buffer, out, params], numElements(outShape));
  ctx.setOutput(out, outShape);
}
