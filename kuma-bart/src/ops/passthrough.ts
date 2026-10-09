import type { OpContext, OpHandler } from "../engine/context.js";
import { numElements } from "../engine/shape.js";

/** Metadata-only passthrough: same buffer, shape taken straight from this node's own
 * (authoritative) manifest meta — no kernel dispatch. Covers aten.alias.default,
 * aten.contiguous.default, aten.to.device, aten.to.dtype, aten.squeeze.dim,
 * aten.unsqueeze.default. `to.dtype` is included here even for the float32->int64 case
 * seen in this model (an index tensor feeding aten.index.Tensor) — kuma-bart keeps
 * "integer" tensors as f32-encoded values throughout and only truncates to u32 at the
 * point of use inside gather.wgsl, so no actual cast/copy is ever needed. */
export const passthroughHandler: OpHandler = (ctx) => {
  const [inputRef] = ctx.node.args;
  const input = ctx.resolve(inputRef!);
  const shape = ctx.node.meta.shape ?? input.shape;
  ctx.setOutput(input.buffer, shape, input.imag);
};

/** aten.zeros_like.default — same shape as input, freshly zero-initialized (a brand new
 * GPUBuffer is zero-initialized per the WebGPU spec, so this is also dispatch-free). */
export const zerosLikeHandler: OpHandler = (ctx) => {
  const [inputRef] = ctx.node.args;
  const input = ctx.resolve(inputRef!);
  const shape = ctx.node.meta.shape ?? input.shape;
  ctx.setOutput(ctx.createBuffer(shape), shape);
};

/** aten.ones_like.default — same shape as input, all ones. The value is a pure function
 * of the (static) shape, never of input data, so it is uploaded once per node and
 * cached for the model's lifetime: no kernel dispatch and no per-run buffer. */
export const onesLikeHandler: OpHandler = (ctx) => {
  const [inputRef] = ctx.node.args;
  const input = ctx.resolve(inputRef!);
  const shape = ctx.node.meta.shape ?? input.shape;
  const n = numElements(shape);
  const buffer = ctx.getOrUploadConstant(`ones_like:${ctx.node.name}:${n}`, () => new Float32Array(n).fill(1));
  ctx.setOutput(buffer, shape);
};
