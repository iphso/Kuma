import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";

const DEPTHWISE_TILE = 16;
// Must match conv2d_depthwise.wgsl's PATCH_CAPACITY exactly -- the largest
// (TILE-1)*stride+k halo patch (in floats) its shared-memory buffer can hold
// (WebGPU's guaranteed-minimum maxComputeWorkgroupStorageSize, 16KB / 4 bytes).
const DEPTHWISE_PATCH_CAPACITY = 4096;
const POINTWISE_TILE = 64;

/** in_channels_per_group === 1 (every output channel depends on exactly one input
 * channel, no cross-channel reduction) routes to the halo-tiled depthwise kernel
 * instead of the general one, *if* its halo patch fits in shared memory -- an unusually
 * large stride/kernel combination falls back to the general kernel below rather than
 * guessing or failing. See conv2d_depthwise.wgsl's header for why this kernel exists
 * (the general kernel's per-output-pixel reads have no cross-channel redundancy to
 * exploit here, only spatial overlap between neighboring receptive fields, which this
 * kernel's input-patch caching targets directly). Returns false (dispatched nothing)
 * when the patch doesn't fit, so the caller falls through to the general path. */
function tryDispatchConv2dDepthwise(
  ctx: OpContext,
  input: ResolvedTensor,
  weight: ResolvedTensor,
  bias: ResolvedTensor,
  outShape: number[],
  batch: number,
  inChannels: number,
  inH: number,
  inW: number,
  outChannels: number,
  outH: number,
  outW: number,
  kh: number,
  kw: number,
  strideH: number,
  strideW: number,
  padH: number,
  padW: number,
  groups: number,
): boolean {
  const patchH = (DEPTHWISE_TILE - 1) * strideH + kh;
  const patchW = (DEPTHWISE_TILE - 1) * strideW + kw;
  if (patchH * patchW > DEPTHWISE_PATCH_CAPACITY) {
    return false;
  }

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([batch, inChannels, inH, inW, outChannels, outH, outW, kh, kw, strideH, strideW, padH, padW, groups]);
  const gridX = Math.ceil(outW / DEPTHWISE_TILE);
  const gridY = Math.ceil(outH / DEPTHWISE_TILE);
  const gridZ = batch * outChannels;
  ctx.dispatchKernelGrid("conv2d_depthwise.wgsl", [input.buffer, weight.buffer, bias.buffer, out, params], gridX, gridY, gridZ);
  ctx.setOutput(out, outShape);
  return true;
}

/** kh===kw===1 && groups===1 (plain pointwise/channel-mixing conv) is a GEMM over
 * rows=(batch, out_h, out_w), cols=out_channels, k=in_channels. conv2d_pointwise.wgsl
 * executes that GEMM directly against NCHW input/output strides, avoiding the large
 * NHWC layout buffers that a literal dispatchLinear route would need. */
function tryDispatchConv2dPointwise(
  ctx: OpContext,
  input: ResolvedTensor,
  weight: ResolvedTensor,
  bias: ResolvedTensor,
  outShape: number[],
  batch: number,
  inChannels: number,
  inH: number,
  inW: number,
  outChannels: number,
  outH: number,
  outW: number,
  kh: number,
  kw: number,
  strideH: number,
  strideW: number,
  padH: number,
  padW: number,
  groups: number,
): boolean {
  if (kh !== 1 || kw !== 1 || groups !== 1) {
    return false;
  }

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([batch, inChannels, inH, inW, outChannels, outH, outW, strideH, strideW, padH, padW]);
  const rows = batch * outH * outW;
  ctx.dispatchKernelGrid(
    "conv2d_pointwise.wgsl",
    [input.buffer, weight.buffer, bias.buffer, out, params],
    Math.ceil(outChannels / POINTWISE_TILE),
    Math.ceil(rows / POINTWISE_TILE),
  );
  ctx.setOutput(out, outShape);
  return true;
}

function dispatchConv2d(
  ctx: OpContext,
  input: ResolvedTensor,
  weight: ResolvedTensor,
  biasRef: ArgValue,
  strideArg: number[],
  paddingArg: number[],
  dilationArg: number[],
  groups: number,
): void {
  const node = ctx.node;
  if (dilationArg[0] !== 1 || dilationArg[1] !== 1) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `dilation ${JSON.stringify(dilationArg)} is not supported — conv2d.wgsl has no dilation parameter.`,
    );
  }

  const outShape = node.meta.shape;
  if (!outShape || outShape.length !== 4) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") is missing a 4D output shape in the manifest.`);
  }
  const [batch, outChannels, outH, outW] = outShape as [number, number, number, number];
  const inChannels = input.shape[1]!;
  const inH = input.shape[2]!;
  const inW = input.shape[3]!;
  const kh = weight.shape[2]!;
  const kw = weight.shape[3]!;
  const strideH = strideArg[0]!;
  const strideW = strideArg[1]!;
  const padH = paddingArg[0]!;
  const padW = paddingArg[1]!;

  const bias = biasRef === null ? { buffer: ctx.getOrCreateZeroBuffer(`zero-bias:${outChannels}`, outChannels * 4), shape: [outChannels] } : ctx.resolve(biasRef);

  const inChannelsPerGroup = inChannels / groups;
  if (
    inChannelsPerGroup === 1 &&
    tryDispatchConv2dDepthwise(
      ctx,
      input,
      weight,
      bias,
      outShape,
      batch,
      inChannels,
      inH,
      inW,
      outChannels,
      outH,
      outW,
      kh,
      kw,
      strideH,
      strideW,
      padH,
      padW,
      groups,
    )
  ) {
    return;
  }
  if (
    tryDispatchConv2dPointwise(
      ctx,
      input,
      weight,
      bias,
      outShape,
      batch,
      inChannels,
      inH,
      inW,
      outChannels,
      outH,
      outW,
      kh,
      kw,
      strideH,
      strideW,
      padH,
      padW,
      groups,
    )
  ) {
    return;
  }

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([batch, inChannels, inH, inW, outChannels, outH, outW, kh, kw, strideH, strideW, padH, padW, groups]);
  ctx.dispatchKernel("conv2d.wgsl", [input.buffer, weight.buffer, bias.buffer, out, params], numElements(outShape));
  ctx.setOutput(out, outShape);
}

/** aten.convolution.default args: (input, weight, bias, stride[2], padding[2],
 * dilation[2], transposed:bool, output_padding[2], groups:int). Trailing args at their
 * schema default (stride=1, padding=0, dilation=1, groups=1) are sometimes omitted by
 * the tracer entirely rather than serialized explicitly — same as aten.conv2d.default
 * below — so every arg past bias is read positionally with an explicit fallback. */
export function conv2dHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const weightRef = node.args[1] as ArgValue;
  const biasRef = (node.args[2] as ArgValue | undefined) ?? null;
  const strideArg = (node.args[3] as number[] | undefined) ?? [1, 1];
  const paddingArg = (node.args[4] as number[] | undefined) ?? [0, 0];
  const dilationArg = (node.args[5] as number[] | undefined) ?? [1, 1];
  const transposedArg = (node.args[6] as boolean | undefined) ?? false;
  const groupsArg = (node.args[8] as number | undefined) ?? 1;

  if (transposedArg) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      "ConvTranspose2d (transposed convolution) has no WGSL kernel yet.",
    );
  }

  const input = ctx.resolve(inputRef);
  const weight = ctx.resolve(weightRef);
  dispatchConv2d(ctx, input, weight, biasRef, strideArg, paddingArg, dilationArg, groupsArg);
}

/** aten.conv2d.default — the simpler non-transposed-capable overload:
 * (input, weight, bias, stride[2], padding[2], dilation[2], groups:int). Same kernel,
 * shorter arg list (no transposed/output_padding fields at all).
 *
 * Confirmed against the real model: most calls here are the bare 3-arg form
 * (input, weight, bias) with stride=1/padding=0/dilation=1/groups=1 all omitted
 * because they're at their PyTorch default — the tracer only serializes args up to
 * the last one actually passed non-default, not the full canonical schema. Every
 * trailing arg needs its own explicit default, not just a positional destructure. */
export function conv2dSimpleHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const weightRef = node.args[1] as ArgValue;
  const biasRef = (node.args[2] as ArgValue | undefined) ?? null;
  const strideArg = (node.args[3] as number[] | undefined) ?? [1, 1];
  const paddingArg = (node.args[4] as number[] | undefined) ?? [0, 0];
  const dilationArg = (node.args[5] as number[] | undefined) ?? [1, 1];
  const groupsArg = (node.args[6] as number | undefined) ?? 1;

  const input = ctx.resolve(inputRef);
  const weight = ctx.resolve(weightRef);
  dispatchConv2d(ctx, input, weight, biasRef, strideArg, paddingArg, dilationArg, groupsArg);
}
