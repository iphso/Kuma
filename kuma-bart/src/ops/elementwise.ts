import type { OpContext, OpHandler, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";
import { broadcastTensor } from "./expand.js";

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

/** Most call sites pass two tensors, but torch.export sometimes records e.g.
 * `tensor * 32` as aten.mul.Tensor with a literal Python number as the second arg
 * (rather than the .Scalar overload) — broadcast that as a same-shape constant.
 *
 * PyTorch's binary ops broadcast implicitly (ATen's own kernels handle it internally),
 * so torch.export traces e.g. a [1,40]-vs-[35,40] aten.div.Tensor with no expand node
 * in between — the smaller operand arrives here at its original, un-broadcast shape.
 * This project's binary WGSL kernels are flat/broadcast-unaware, so an operand whose
 * shape doesn't already match the op's output shape needs to be materialized up to it
 * first (same machinery aten.expand.default uses) — skipping this previously meant the
 * kernel read past the smaller buffer's end for every output index beyond its own
 * length, landing on whatever this GPU's robust-buffer-access default returns for an
 * out-of-bounds storage read (here: 0), turning a real-but-small divisor into 0 and the
 * result into +-Infinity. */
function resolveOperand(ctx: OpContext, arg: ArgValue, shape: readonly number[], label: "a" | "b"): ResolvedTensor {
  if (typeof arg === "number") {
    // arg is graph-structure-derived (a literal fixed at manifest-build time), never
    // from actual input data -- same caching rationale as ctx.uniform/uniformTyped
    // (see context.ts), so cached forever per-node rather than rebuilt and
    // re-uploaded every call. label disambiguates a-side vs b-side in case a single
    // node's binary op ever has both operands as literals.
    const buffer = ctx.getOrUploadConstant(`operand:${ctx.node.name}:${label}`, () =>
      new Float32Array(numElements(shape)).fill(arg),
    );
    return { buffer, shape: [...shape] };
  }
  const tensor = ctx.resolve(arg);
  if (sameShape(tensor.shape, shape)) {
    return tensor;
  }
  const buffer = broadcastTensor(ctx, tensor, shape);
  // imag (if present) needs the exact same broadcast, or it'd silently go missing.
  const imag = tensor.imag ? broadcastTensor(ctx, { buffer: tensor.imag, shape: tensor.shape }, shape) : undefined;
  return { buffer, shape: [...shape], imag };
}

function dispatchBinary(ctx: OpContext, kernelName: string, x: GPUBuffer, y: GPUBuffer, shape: readonly number[], n: number): GPUBuffer {
  const out = ctx.createBuffer(shape);
  ctx.dispatchKernel(kernelName, [x, y, out, ctx.uniform([n])], n);
  return out;
}

function negate(ctx: OpContext, x: GPUBuffer, shape: readonly number[], n: number): GPUBuffer {
  const out = ctx.createBuffer(shape);
  ctx.dispatchKernel("neg.wgsl", [x, out, ctx.uniform([n])], n);
  return out;
}

/** out_im = a_im (+/-) b_im, treating a missing side as zero — so a real operand
 * combined with a complex one just passes the complex side's imaginary part through
 * (negated for the b-side of a subtraction). Used by add/sub, where the imaginary part
 * combines exactly like the real part does, with no cross terms. */
function combineImagLinear(
  ctx: OpContext,
  kernelName: "add.wgsl" | "sub.wgsl",
  aImag: GPUBuffer | undefined,
  bImag: GPUBuffer | undefined,
  shape: readonly number[],
  n: number,
): GPUBuffer | undefined {
  if (!aImag && !bImag) return undefined;
  if (aImag && !bImag) return aImag;
  if (!aImag && bImag) {
    return kernelName === "add.wgsl" ? bImag : negate(ctx, bImag, shape, n);
  }
  return dispatchBinary(ctx, kernelName, aImag!, bImag!, shape, n);
}

/** (a_re+a_im*i)*(b_re+b_im*i) = (a_re*b_re - a_im*b_im) + (a_re*b_im + a_im*b_re)*i.
 * `realOut` is already a_re*b_re (the plain real dispatch every call site does
 * regardless of complexity) — this only computes the corrections needed on top. */
function combineMultiplyComplex(
  ctx: OpContext,
  a: ResolvedTensor,
  b: ResolvedTensor,
  realOut: GPUBuffer,
  shape: readonly number[],
  n: number,
): { re: GPUBuffer; im?: GPUBuffer } {
  if (a.imag && b.imag) {
    const imIm = dispatchBinary(ctx, "mul.wgsl", a.imag, b.imag, shape, n);
    const re = dispatchBinary(ctx, "sub.wgsl", realOut, imIm, shape, n);
    const reIm = dispatchBinary(ctx, "mul.wgsl", a.buffer, b.imag, shape, n);
    const imRe = dispatchBinary(ctx, "mul.wgsl", a.imag, b.buffer, shape, n);
    const im = dispatchBinary(ctx, "add.wgsl", reIm, imRe, shape, n);
    return { re, im };
  }
  if (a.imag && !b.imag) {
    return { re: realOut, im: dispatchBinary(ctx, "mul.wgsl", a.imag, b.buffer, shape, n) };
  }
  if (!a.imag && b.imag) {
    return { re: realOut, im: dispatchBinary(ctx, "mul.wgsl", a.buffer, b.imag, shape, n) };
  }
  return { re: realOut };
}

type ComplexPolicy = "linear" | "multiply" | "divide" | "unsupported";

function binaryElementwise(kernelName: string, complexPolicy: ComplexPolicy = "unsupported"): OpHandler {
  return (ctx: OpContext): void => {
    const node = ctx.node;
    const [aArg, bArg] = node.args;
    const shape = node.meta.shape;
    if (!shape) {
      throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") is missing an output shape in the manifest.`);
    }
    const a = resolveOperand(ctx, aArg!, shape, "a");
    const b = resolveOperand(ctx, bArg!, shape, "b");
    const n = numElements(shape);
    const realOut = dispatchBinary(ctx, kernelName, a.buffer, b.buffer, shape, n);

    if (!a.imag && !b.imag) {
      ctx.setOutput(realOut, shape);
      return;
    }
    if (complexPolicy === "unsupported") {
      throw new KumaUnsupportedOpError(node.target, node.name, "complex (imaginary-paired) operands aren't supported for this op.");
    }
    if (complexPolicy === "linear") {
      const im = combineImagLinear(ctx, kernelName as "add.wgsl" | "sub.wgsl", a.imag, b.imag, shape, n);
      ctx.setOutput(realOut, shape, im);
      return;
    }
    if (complexPolicy === "divide") {
      // (a_re+a_im*i)/b_re = a_re/b_re + (a_im/b_re)*i -- only a real denominator is
      // supported; a complex one would need the full conjugate-multiplication formula,
      // which nothing observed so far actually needs.
      if (b.imag) {
        throw new KumaUnsupportedOpError(node.target, node.name, "division by a complex denominator isn't supported (only complex ÷ real).");
      }
      const im = a.imag ? dispatchBinary(ctx, "div.wgsl", a.imag, b.buffer, shape, n) : undefined;
      ctx.setOutput(realOut, shape, im);
      return;
    }
    const { re, im } = combineMultiplyComplex(ctx, a, b, realOut, shape, n);
    ctx.setOutput(re, shape, im);
  };
}

function unaryElementwise(kernelName: string): OpHandler {
  return (ctx: OpContext): void => {
    const [inputRef] = ctx.node.args;
    const input = ctx.resolve(inputRef!);
    const shape = ctx.node.meta.shape ?? input.shape;
    const n = numElements(shape);
    const out = ctx.createBuffer(shape);
    const params = ctx.uniform([n]);
    ctx.dispatchKernel(kernelName, [input.buffer, out, params], n);
    ctx.setOutput(out, shape);
  };
}

export const addHandler = binaryElementwise("add.wgsl", "linear");
export const mulHandler = binaryElementwise("mul.wgsl", "multiply");
export const subHandler = binaryElementwise("sub.wgsl", "linear");
export const divHandler = binaryElementwise("div.wgsl", "divide");
export const minimumHandler = binaryElementwise("minimum.wgsl");
export const reluHandler = unaryElementwise("relu.wgsl");
export const geluHandler = unaryElementwise("gelu.wgsl");
export const sqrtHandler = unaryElementwise("sqrt.wgsl");
export const cosHandler = unaryElementwise("cos.wgsl");
export const sinHandler = unaryElementwise("sin.wgsl");
export const floorHandler = unaryElementwise("floor.wgsl");
export const erfHandler = unaryElementwise("erf.wgsl");
export const reshapeHandler = unaryElementwise("reshape.wgsl");

/** aten.div.Tensor_mode args: (a, b, rounding_mode). Only "floor" is supported (the
 * only value seen in this model) — composed from existing kernels: divide, then
 * floor. Anything else fails loudly rather than guessing. */
export function divModeHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [aRef, bRef, modeArg] = node.args as [ArgValue, ArgValue, string];
  if (modeArg !== "floor") {
    throw new KumaUnsupportedOpError(node.target, node.name, `only rounding_mode="floor" is supported, got "${modeArg}".`);
  }
  const a = ctx.resolve(aRef);
  const b = ctx.resolve(bRef);
  const shape = node.meta.shape ?? a.shape;
  const n = numElements(shape);

  const divided = ctx.createBuffer(shape);
  ctx.dispatchKernel("div.wgsl", [a.buffer, b.buffer, divided, ctx.uniform([n])], n);

  const out = ctx.createBuffer(shape);
  ctx.dispatchKernel("floor.wgsl", [divided, out, ctx.uniform([n])], n);
  ctx.setOutput(out, shape);
}
