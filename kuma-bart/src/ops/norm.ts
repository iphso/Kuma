import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";
import { numElements } from "../engine/shape.js";
import { KERNEL_WORKGROUP_SIZE } from "../engine/dispatch.js";

/** aten.group_norm.default args: (input, num_groups, weight, bias, eps, cudnn_enabled).
 * The single-tensor-output wrapper, not the 3-tuple aten.native_group_norm.default.
 * Confirmed against the real model: eps is consistently omitted (every call has 4
 * args, not 5) because it's always called at its schema default — same "tracer skips
 * trailing default args" pattern as aten.conv2d.default (see ops/conv2d.ts). */
export function groupNormHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const numGroupsArg = node.args[1] as number;
  const weightRef = node.args[2] as ArgValue;
  const biasRef = node.args[3] as ArgValue;
  const epsArg = (node.args[4] as number | undefined) ?? 1e-5;
  const input = ctx.resolve(inputRef);
  if (input.shape.length < 3) {
    throw new KumaShapeError(
      `Op "${node.target}" (node "${node.name}") expects an N,C,... input, got shape ${JSON.stringify(input.shape)}.`,
    );
  }
  const [batch, channels] = input.shape as [number, number, ...number[]];
  const spatial = numElements(input.shape.slice(2));
  const groups = numGroupsArg;
  const channelsPerGroup = channels / groups;
  const rows = batch * groups;

  const weight = ctx.resolve(weightRef);
  const bias = ctx.resolve(biasRef);
  const shape = node.meta.shape ?? input.shape;

  const out = ctx.createBuffer(shape);
  const params = ctx.uniformTyped([
    { u32: batch },
    { u32: channels },
    { u32: spatial },
    { u32: groups },
    { u32: channelsPerGroup },
    { f32: epsArg },
    { u32: rows },
  ]);
  // groupnorm.wgsl uses one *workgroup* per row (256 threads cooperatively reducing
  // that row's mean/variance via shared memory), not one thread per row -- the original
  // version left the GPU almost entirely idle whenever rows (batch*groups) was small
  // relative to how much data each row covers (measured at 49% of a whole frame's GPU
  // time for this model). The KERNEL_WORKGROUP_SIZE (64) below is dispatchKernel's own
  // element-count-to-workgroup-count conversion factor, not this shader's thread count
  // (which is fixed at 256 inside groupnorm.wgsl, independent of this call) -- passing
  // rows*64 here always yields exactly `rows` workgroups dispatched.
  ctx.dispatchKernel("groupnorm.wgsl", [input.buffer, weight.buffer, bias.buffer, out, params], rows * KERNEL_WORKGROUP_SIZE);
  ctx.setOutput(out, shape);
}

/** aten.layer_norm.default args: (input, normalized_shape, weight, bias, eps,
 * cudnn_enable). The single-tensor-output wrapper, not the 3-tuple
 * aten.native_layer_norm.default. */
export function layerNormHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, normalizedShapeArg, weightRef, biasRef, epsArgRaw] = node.args as [
    ArgValue,
    number[],
    ArgValue,
    ArgValue,
    number | undefined,
  ];
  // Same "tracer skips trailing default args" elision as aten.group_norm.default above —
  // without this fallback, an omitted eps packs as `undefined` -> NaN into the uniform
  // buffer (DataView.setFloat32(offset, undefined) coerces to NaN), which poisons every
  // element of variance + eps and therefore the entire output.
  const epsArg = epsArgRaw ?? 1e-5;
  const input = ctx.resolve(inputRef);
  const normSize = numElements(normalizedShapeArg);
  const rows = numElements(input.shape) / normSize;

  const weight = ctx.resolve(weightRef);
  const bias = ctx.resolve(biasRef);
  const shape = node.meta.shape ?? input.shape;

  const out = ctx.createBuffer(shape);
  const params = ctx.uniformTyped([{ u32: rows }, { u32: normSize }, { f32: epsArg }]);
  ctx.dispatchKernel("layernorm.wgsl", [input.buffer, weight.buffer, bias.buffer, out, params], rows);
  ctx.setOutput(out, shape);
}
