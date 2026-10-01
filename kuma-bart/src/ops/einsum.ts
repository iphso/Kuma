import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";
import { permuteMaybeComplex } from "./permute.js";
import { planEinsum } from "../engine/einsum-plan.js";
import { complexProduct } from "./complex_math.js";

const BMM_TILE = 64;

/** bmm.wgsl: a:(B,M,K), b:(B,K,N) -> out:(B,M,N). */
function dispatchBmm(ctx: OpContext, aBuf: GPUBuffer, bBuf: GPUBuffer, batch: number, m: number, k: number, n: number): GPUBuffer {
  const out = ctx.createBuffer([batch, m, n]);
  const params = ctx.uniform([batch, m, k, n]);
  ctx.dispatchKernelGrid("bmm.wgsl", [aBuf, bBuf, out, params], Math.ceil(n / BMM_TILE), Math.ceil(m / BMM_TILE), batch);
  return out;
}

/** Batched matmul, complex-aware via complexProduct: a single real bmm when neither
 * operand carries an imaginary part (most contraction steps in practice). */
function dispatchComplexBmm(ctx: OpContext, a: ResolvedTensor, b: ResolvedTensor, batch: number, m: number, k: number, n: number): ResolvedTensor {
  const shape = [batch, m, n];
  const bmm = (x: GPUBuffer, y: GPUBuffer) => dispatchBmm(ctx, x, y, batch, m, k, n);
  const { re, im } = complexProduct(ctx, { re: a.buffer, im: a.imag }, { re: b.buffer, im: b.imag }, bmm, shape);
  return { buffer: re, shape, imag: im };
}

/**
 * aten.einsum.default(equation, operands) — general support via engine/einsum-plan.ts:
 * the equation is planned as a sequence of pairwise contractions (permute each operand
 * into [batch, kept, contracted] order, free-reshape to 3D, batched matmul, free-reshape
 * the result back out), then a final permute to match the requested output order.
 * Complex-aware: any operand may be complex-paired (a ResolvedTensor with `.imag`,
 * e.g. from aten.complex.default) — a mix of real and complex operands in the same
 * contraction (e.g. a complex Tucker core times some real, some complex factors) is
 * expected, not an edge case. The plan and the complex-bmm math are both validated
 * against independent brute-force references in test/einsum-plan.test.ts.
 *
 * permute.wgsl caps at rank 4, so any operand/intermediate above that fails loudly via
 * dispatchPermute rather than silently truncating.
 */
export function einsumHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [equationArg, operandsArg] = node.args as [string, ArgValue[]];

  if (!Array.isArray(operandsArg) || operandsArg.length < 2) {
    throw new KumaShapeError(
      `Op "${node.target}" (node "${node.name}") expected 2+ einsum operands, got ` +
        `${Array.isArray(operandsArg) ? operandsArg.length : "a non-list args[1]"}.`,
    );
  }

  const operands = operandsArg.map((ref) => ctx.resolve(ref));
  const plan = planEinsum(equationArg, operands.map((o) => o.shape));

  let acc = operands[0]!;
  for (const step of plan.steps) {
    const permutedAcc = permuteMaybeComplex(ctx, acc, step.accPermutedShape, step.accPermDims);
    const op = operands[step.operandIndex]!;
    const permutedOp = permuteMaybeComplex(ctx, op, step.opPermutedShape, step.opPermDims);

    // Free reshape to the 3D (batch, m/n, k) shape bmm.wgsl expects -- already
    // contiguous from the permute above, just reinterpreted.
    const aFor3d: ResolvedTensor = { buffer: permutedAcc.buffer, shape: [step.batchSize, step.m, step.k], imag: permutedAcc.imag };
    const bFor3d: ResolvedTensor = { buffer: permutedOp.buffer, shape: [step.batchSize, step.k, step.n], imag: permutedOp.imag };

    const bmmOut = dispatchComplexBmm(ctx, aFor3d, bFor3d, step.batchSize, step.m, step.k, step.n);
    acc = { buffer: bmmOut.buffer, shape: step.resultShape, imag: bmmOut.imag };
  }

  const final = permuteMaybeComplex(ctx, acc, plan.outputShape, plan.finalPermDims);
  const outShape = node.meta.shape ?? final.shape;
  ctx.setOutput(final.buffer, outShape, final.imag);
}
