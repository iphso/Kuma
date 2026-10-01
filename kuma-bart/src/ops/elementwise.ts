import type { OpContext, OpHandler, ResolvedTensor } from "../engine/context.js";
import { isNodeRef, type ArgValue } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";
import { broadcastTensor } from "./expand.js";
import { complexProduct, dispatchBinary, dispatchUnary } from "./complex_math.js";

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

/** Materializes one binary-op operand at the op's output `shape`.
 *
 * A literal Python number (torch.export sometimes records `tensor * 32` as
 * aten.mul.Tensor with a scalar second arg) becomes a same-shape constant buffer.
 *
 * PyTorch's binary ops broadcast implicitly, so torch.export traces e.g. a
 * [1,40]-vs-[35,40] aten.div.Tensor with no expand node in between. kuma's binary WGSL
 * kernels are flat and broadcast-unaware -- reading past a smaller buffer's end returns
 * 0 under robust buffer access, silently turning a divisor into 0 -- so any operand whose
 * shape differs from the output shape is broadcast up to it first (same machinery as
 * aten.expand.default). */
function resolveOperand(ctx: OpContext, arg: ArgValue, shape: readonly number[], label: "a" | "b"): ResolvedTensor {
  if (typeof arg === "number") {
    // A literal fixed at export time, never input data, so it is cached for the model's
    // lifetime. `label` separates the two sides of a node with two literal operands; the
    // element count keeps a node re-run at a different shape from reusing a wrong-size
    // buffer.
    const n = numElements(shape);
    const buffer = ctx.getOrUploadConstant(`operand:${ctx.node.name}:${label}:${n}`, () => new Float32Array(n).fill(arg));
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

/** Output shape for a binary op: the manifest's own, falling back to the first tensor
 * operand's shape. */
function binaryOutputShape(ctx: OpContext, aArg: ArgValue, bArg: ArgValue): number[] {
  const shape = ctx.node.meta.shape;
  if (shape) return shape;
  const tensorArg = isNodeRef(aArg) ? aArg : isNodeRef(bArg) ? bArg : undefined;
  if (!tensorArg) {
    throw new KumaShapeError(`Op "${ctx.node.target}" (node "${ctx.node.name}") is missing an output shape in the manifest.`);
  }
  return ctx.resolve(tensorArg).shape;
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

    if (complexPolicy === "multiply") {
      const mul = (x: GPUBuffer, y: GPUBuffer) => dispatchBinary(ctx, kernelName, x, y, shape);
      const { re, im } = complexProduct(ctx, { re: a.buffer, im: a.imag }, { re: b.buffer, im: b.imag }, mul, shape);
      ctx.setOutput(re, shape, im);
      return;
    }

    const realOut = dispatchBinary(ctx, kernelName, a.buffer, b.buffer, shape);
    if (!a.imag && !b.imag) {
      ctx.setOutput(realOut, shape);
      return;
    }
    if (complexPolicy === "unsupported") {
      throw new KumaUnsupportedOpError(node.target, node.name, "complex (imaginary-paired) operands aren't supported for this op.");
    }
    if (complexPolicy === "linear") {
      // out_im = a_im ± b_im with a missing side treated as zero -- no cross terms.
      let im: GPUBuffer;
      if (a.imag && b.imag) im = dispatchBinary(ctx, kernelName, a.imag, b.imag, shape);
      else if (a.imag) im = a.imag;
      else im = kernelName === "add.wgsl" ? b.imag! : dispatchUnary(ctx, "neg.wgsl", b.imag!, shape);
      ctx.setOutput(realOut, shape, im);
      return;
    }
    // "divide": (a_re+a_im*i)/b_re = a_re/b_re + (a_im/b_re)*i -- only a real denominator
    // is supported; a complex one would need the conjugate-multiplication formula.
    if (b.imag) {
      throw new KumaUnsupportedOpError(node.target, node.name, "division by a complex denominator isn't supported (only complex ÷ real).");
    }
    const im = a.imag ? dispatchBinary(ctx, "div.wgsl", a.imag, b.buffer, shape) : undefined;
    ctx.setOutput(realOut, shape, im);
  };
}

function unaryElementwise(kernelName: string): OpHandler {
  return (ctx: OpContext): void => {
    const [inputRef] = ctx.node.args;
    const input = ctx.resolve(inputRef!);
    const shape = ctx.node.meta.shape ?? input.shape;
    ctx.setOutput(dispatchUnary(ctx, kernelName, input.buffer, shape), shape);
  };
}

export const addHandler = binaryElementwise("add.wgsl", "linear");
export const mulHandler = binaryElementwise("mul.wgsl", "multiply");
export const subHandler = binaryElementwise("sub.wgsl", "linear");
export const divHandler = binaryElementwise("div.wgsl", "divide");
export const minimumHandler = binaryElementwise("minimum.wgsl");
export const reluHandler = unaryElementwise("relu.wgsl");
export const sqrtHandler = unaryElementwise("sqrt.wgsl");
export const cosHandler = unaryElementwise("cos.wgsl");
export const sinHandler = unaryElementwise("sin.wgsl");
export const floorHandler = unaryElementwise("floor.wgsl");
export const erfHandler = unaryElementwise("erf.wgsl");
export const sigmoidHandler = unaryElementwise("sigmoid.wgsl");
export const tanhHandler = unaryElementwise("tanh.wgsl");
export const siluHandler = unaryElementwise("silu.wgsl");
export const absHandler = unaryElementwise("abs.wgsl");
export const expHandler = unaryElementwise("exp.wgsl");
export const logHandler = unaryElementwise("log.wgsl");
export const rsqrtHandler = unaryElementwise("rsqrt.wgsl");
export const hardswishHandler = unaryElementwise("hardswish.wgsl");
export const reshapeHandler = unaryElementwise("reshape.wgsl");

const geluTanh = unaryElementwise("gelu.wgsl");
const geluErf = unaryElementwise("gelu_erf.wgsl");
let warnedGeluFallback = false;

/** aten.gelu.default(input, *, approximate="none"|"tanh"). The exact erf form runs
 * gelu_erf.wgsl; approximate="tanh" runs gelu.wgsl. Packages exported before
 * gelu_erf.wgsl existed only ship gelu.wgsl, so they fall back to the tanh approximation
 * (max abs error ~1e-3), with a one-time warning. */
export function geluHandler(ctx: OpContext): void {
  const approximate = ctx.node.kwargs["approximate"] ?? ctx.node.args[1] ?? "none";
  if (approximate === "tanh") {
    geluTanh(ctx);
    return;
  }
  if (ctx.hasKernel("gelu_erf.wgsl")) {
    geluErf(ctx);
    return;
  }
  if (!warnedGeluFallback) {
    warnedGeluFallback = true;
    console.warn(
      "[kuma-bart] this .iph has no kernels/gelu_erf.wgsl (exported before exact GELU was supported); " +
        "running aten.gelu.default with the tanh approximation instead. Re-export to get exact GELU.",
    );
  }
  geluTanh(ctx);
}

/** aten.div.Tensor_mode args: (a, b, rounding_mode). Only "floor" is supported --
 * composed as div then floor, with the same scalar/broadcast operand handling as the
 * plain binary ops. Anything else fails loudly rather than guessing. */
export function divModeHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [aArg, bArg, modeArg] = node.args as [ArgValue, ArgValue, string];
  if (modeArg !== "floor") {
    throw new KumaUnsupportedOpError(node.target, node.name, `only rounding_mode="floor" is supported, got "${modeArg}".`);
  }
  const shape = binaryOutputShape(ctx, aArg, bArg);
  const a = resolveOperand(ctx, aArg, shape, "a");
  const b = resolveOperand(ctx, bArg, shape, "b");
  const divided = dispatchBinary(ctx, "div.wgsl", a.buffer, b.buffer, shape);
  ctx.setOutput(dispatchUnary(ctx, "floor.wgsl", divided, shape), shape);
}
