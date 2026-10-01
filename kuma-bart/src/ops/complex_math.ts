import type { OpContext } from "../engine/context.js";
import { numElements } from "../engine/shape.js";

/** out = kernel(x, y), for the flat binary kernels (add/sub/mul/div/minimum.wgsl), whose
 * bindings are (a, b, out, Params{n}). Operands must already be `shape`-sized. */
export function dispatchBinary(ctx: OpContext, kernelName: string, x: GPUBuffer, y: GPUBuffer, shape: readonly number[]): GPUBuffer {
  const n = numElements(shape);
  const out = ctx.createBuffer(shape);
  ctx.dispatchKernel(kernelName, [x, y, out, ctx.uniform([n])], n);
  return out;
}

/** out = kernel(x), for the flat unary kernels (relu/neg/exp/...wgsl), whose bindings are
 * (input, out, Params{n}). */
export function dispatchUnary(ctx: OpContext, kernelName: string, x: GPUBuffer, shape: readonly number[]): GPUBuffer {
  const n = numElements(shape);
  const out = ctx.createBuffer(shape);
  ctx.dispatchKernel(kernelName, [x, out, ctx.uniform([n])], n);
  return out;
}

/** One complex operand as a real part plus an optional imaginary part, in whatever form
 * the product function takes (a buffer, or a shaped tensor for matmul-like products). */
export interface ComplexParts<T> {
  re: T;
  im?: T;
}

/** (a_re + a_im·i) ⊗ (b_re + b_im·i) = (a_re⊗b_re − a_im⊗b_im) + (a_re⊗b_im + a_im⊗b_re)·i
 * for any real bilinear product ⊗ (elementwise mul, batched matmul, basis matmul), whose
 * results are `outShape`-sized. A missing imaginary part is treated as zero, so a
 * real⊗real product costs exactly one `mul` and no combine dispatches. */
export function complexProduct<T>(
  ctx: OpContext,
  a: ComplexParts<T>,
  b: ComplexParts<T>,
  mul: (x: T, y: T) => GPUBuffer,
  outShape: readonly number[],
): { re: GPUBuffer; im?: GPUBuffer } {
  const reRe = mul(a.re, b.re);
  if (a.im === undefined && b.im === undefined) return { re: reRe };
  if (b.im === undefined) return { re: reRe, im: mul(a.im!, b.re) };
  if (a.im === undefined) return { re: reRe, im: mul(a.re, b.im) };
  const re = dispatchBinary(ctx, "sub.wgsl", reRe, mul(a.im, b.im), outShape);
  const im = dispatchBinary(ctx, "add.wgsl", mul(a.re, b.im), mul(a.im, b.re), outShape);
  return { re, im };
}
