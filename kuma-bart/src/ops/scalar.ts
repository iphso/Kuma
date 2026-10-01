import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";
import { dispatchBinary, dispatchUnary } from "./complex_math.js";

/** aten.clamp.default args: (input, min=None, max=None). clamp.wgsl wants concrete
 * float bounds — its own header comment documents passing ±inf for an unbounded side. */
export function clampHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, minArg, maxArg] = node.args as [ArgValue, number | null, number | null];
  const input = ctx.resolve(inputRef);
  const shape = node.meta.shape ?? input.shape;
  const n = numElements(shape);

  const out = ctx.createBuffer(shape);
  const params = ctx.uniformTyped([{ u32: n }, { f32: minArg ?? -Infinity }, { f32: maxArg ?? Infinity }]);
  ctx.dispatchKernel("clamp.wgsl", [input.buffer, out, params], n);
  ctx.setOutput(out, shape);
}

/** aten.pow.Tensor_Scalar args: (input, exponent). */
export function powScalarHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, exponentArg] = node.args as [ArgValue, number];
  const input = ctx.resolve(inputRef);
  const shape = node.meta.shape ?? input.shape;
  const n = numElements(shape);

  const out = ctx.createBuffer(shape);
  const params = ctx.uniformTyped([{ u32: n }, { f32: exponentArg }]);
  ctx.dispatchKernel("pow_scalar.wgsl", [input.buffer, out, params], n);
  ctx.setOutput(out, shape);
}

/** aten.rsub.Scalar args: (input, scalar, alpha=1) — computes scalar - input*alpha.
 * No dedicated kernel: the scalar is a literal known at this point (not a tensor), so
 * it's broadcast into a same-shape constant buffer and fed through the existing
 * sub.wgsl as `sub(scalarBuffer, input)`. Only alpha=1 (the only value seen in
 * practice) is supported — anything else fails loudly rather than guessing.
 * Complex-aware: scalar - (re+im*i) = (scalar-re) - im*i — the scalar has no
 * imaginary part, so the input's imaginary part just carries through negated. */
export function rsubScalarHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const scalarArg = node.args[1] as number;
  const alphaArg = node.args[2] as number | undefined;
  const alpha = alphaArg ?? 1;
  if (alpha !== 1) {
    throw new KumaUnsupportedOpError(node.target, node.name, `only alpha=1 is supported, got alpha=${alpha}.`);
  }

  const input = ctx.resolve(inputRef);
  const shape = node.meta.shape ?? input.shape;
  const n = numElements(shape);

  // A literal fixed at export time, never input data, so it is cached for the model's
  // lifetime; the element count is part of the key so a node re-run at a different shape
  // never binds a wrong-size buffer.
  const scalarBuffer = ctx.getOrUploadConstant(`rsub:${node.name}:${n}`, () => new Float32Array(n).fill(scalarArg));
  const out = dispatchBinary(ctx, "sub.wgsl", scalarBuffer, input.buffer, shape);
  const outImag = input.imag ? dispatchUnary(ctx, "neg.wgsl", input.imag, shape) : undefined;
  ctx.setOutput(out, shape, outImag);
}
