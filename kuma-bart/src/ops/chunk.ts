import type { OpContext } from "../engine/context.js";
import { isNodeRef, type ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";
import { normalizeDim, numElements, outerInner } from "../engine/shape.js";

/** aten.chunk.default args: (input, chunks, dim). Splits into `chunks` pieces along
 * `dim` (the last piece may be smaller if not evenly divisible) — a multi-output node;
 * each piece is dispatched via slice.wgsl and registered under a synthetic per-index
 * key, picked up by a following getitem(thisNode, i) node. No single `setOutput` call —
 * the scheduler knows not to expect one because this node's manifest meta has an
 * `outputs[]` list rather than a single `shape`. */
export function chunkHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, chunksArg, dimArg] = node.args as [ArgValue, number, number];
  const input = ctx.resolve(inputRef);
  const dim = normalizeDim(dimArg, input.shape.length);
  const dimSize = input.shape[dim]!;
  const chunkSize = Math.ceil(dimSize / chunksArg);
  const { outer, inner } = outerInner(input.shape, dim);

  let start = 0;
  for (let i = 0; i < chunksArg; i++) {
    const extent = Math.min(chunkSize, dimSize - start);
    if (extent <= 0) break;
    const outShape = input.shape.slice();
    outShape[dim] = extent;
    const n = numElements(outShape);

    const out = ctx.createBuffer(outShape);
    const params = ctx.uniform([outer, inner, dimSize, extent, start, 1, n]);
    ctx.dispatchKernel("slice.wgsl", [input.buffer, out, params], n);

    let outImag: GPUBuffer | undefined;
    if (input.imag) {
      outImag = ctx.createBuffer(outShape);
      ctx.dispatchKernel("slice.wgsl", [input.imag, outImag, params], n);
    }

    ctx.setIndexedOutput(i, { buffer: out, shape: outShape, imag: outImag });
    start += extent;
  }
}

/** `getitem(node, i)` — FX's surface form for picking one result out of a multi-output
 * node (here, always a preceding aten.chunk.default). Target string is the literal
 * builtin name "getitem", not an aten.* string. */
export function getitemHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [sourceRef, indexArg] = node.args as [ArgValue, number];
  if (!isNodeRef(sourceRef)) {
    throw new KumaShapeError(`Op "getitem" (node "${node.name}") expected a node reference as its first arg.`);
  }
  ctx.setOutputTensor(ctx.resolveIndexed(sourceRef.node_ref, indexArg));
}

/** aten.select.int args: (input, dim, index) — a literal int index, never a tensor.
 * Equivalent to a width-1 slice (reusing slice.wgsl) with the selected dim then dropped
 * — that drop is metadata-only, so we trust this node's own manifest shape for it. */
export function selectHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, dimArg, indexArg] = node.args as [ArgValue, number, number];
  const input = ctx.resolve(inputRef);
  const dim = normalizeDim(dimArg, input.shape.length);
  const dimSize = input.shape[dim]!;
  const index = indexArg < 0 ? indexArg + dimSize : indexArg;

  const { outer, inner } = outerInner(input.shape, dim);
  const slicedShape = input.shape.slice();
  slicedShape[dim] = 1;
  const n = numElements(slicedShape);

  const out = ctx.createBuffer(slicedShape);
  const params = ctx.uniform([outer, inner, dimSize, 1, index, 1, n]);
  ctx.dispatchKernel("slice.wgsl", [input.buffer, out, params], n);

  const finalShape = node.meta.shape ?? slicedShape.filter((_, d) => d !== dim);
  let outImag: GPUBuffer | undefined;
  if (input.imag) {
    outImag = ctx.createBuffer(slicedShape);
    ctx.dispatchKernel("slice.wgsl", [input.imag, outImag, params], n);
  }

  ctx.setOutput(out, finalShape, outImag);
}
