import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { reductionLayout } from "./reduce.js";

/** aten.softmax.int args: (input, dim, dtype=None); aten._softmax.default args:
 * (input, dim, half_to_float). One softmax.wgsl invocation per (outer, inner) row:
 * max-subtracted exp, normalised in place -- numerically the same as PyTorch's. */
export function softmaxHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, dimArg, third] = node.args as [ArgValue, number, unknown];
  if (third !== undefined && third !== null && third !== false && third !== "float32") {
    throw new KumaUnsupportedOpError(node.target, node.name, `only float32 softmax is supported, got dtype/half_to_float=${JSON.stringify(third)}.`);
  }
  const input = ctx.resolve(inputRef);
  if (input.imag) {
    throw new KumaUnsupportedOpError(node.target, node.name, "complex (imaginary-paired) input isn't supported.");
  }
  const shape = node.meta.shape ?? input.shape;
  const { outer, reduce, inner } = reductionLayout(node.target, node.name, input.shape, [dimArg]);
  const rows = outer * inner;

  const out = ctx.createBuffer(shape);
  const params = ctx.uniform([outer, reduce, inner, rows]);
  ctx.dispatchKernel("softmax.wgsl", [input.buffer, out, params], rows);
  ctx.setOutput(out, shape);
}
