import { KumaShapeError } from "../errors.js";

export function numElements(shape: readonly number[]): number {
  return shape.reduce((acc, d) => acc * d, 1);
}

/** Row-major (C-contiguous) strides, in elements, for a shape. */
export function contiguousStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1]! * shape[i + 1]!;
  }
  return strides;
}

export function normalizeDim(dim: number, rank: number): number {
  const d = dim < 0 ? dim + rank : dim;
  if (d < 0 || d >= rank) {
    throw new KumaShapeError(`dim ${dim} is out of range for a rank-${rank} tensor`);
  }
  return d;
}

/** outer = product of dims before `dim`, inner = product of dims after `dim` — the
 * outer/inner factoring shared by the concat and slice kernels. */
export function outerInner(shape: readonly number[], dim: number): { outer: number; inner: number } {
  return {
    outer: numElements(shape.slice(0, dim)),
    inner: numElements(shape.slice(dim + 1)),
  };
}

/** Per-axis input strides for broadcasting `inputShape` (left-padded with 1s as
 * needed) up to `targetShape` — 0 for an expanded (broadcast) axis, the normal
 * contiguous stride otherwise. Used by aten.expand.default's permute.wgsl-based
 * implementation (stride 0 means every output index along that axis reads the same
 * single input element). Throws if a non-1 axis would need to change size. */
export function expandStrides(inputShape: readonly number[], targetShape: readonly number[]): number[] {
  const rankPad = targetShape.length - inputShape.length;
  if (rankPad < 0) {
    throw new KumaShapeError(
      `can't expand a rank-${inputShape.length} shape down to rank-${targetShape.length}`,
    );
  }
  const paddedInputShape = new Array(rankPad).fill(1).concat(inputShape)
    .map((size, i) => (size === -1 ? targetShape[i]! : size));
  const inputStrides = contiguousStrides(paddedInputShape);
  return targetShape.map((targetSize, d) => {
    const inSize = paddedInputShape[d]!;
    if (inSize === targetSize) return inputStrides[d]!;
    if (inSize === 1) return 0;
    throw new KumaShapeError(`can't expand axis ${d} from ${inSize} to ${targetSize}`);
  });
}

/** PyTorch FX commonly encodes "slice to the end" as a huge sentinel int (e.g.
 * sys.maxsize) and "from the start" / "to the end" as `null`. Clamp both to the real
 * dimension extent and resolve negative indices, mirroring aten.slice.Tensor semantics. */
export function clampSliceBounds(
  start: number | null,
  end: number | null,
  dimSize: number,
): { start: number; end: number } {
  let s = start ?? 0;
  let e = end ?? dimSize;
  if (s < 0) s += dimSize;
  if (e < 0) e += dimSize;
  s = Math.max(0, Math.min(s, dimSize));
  e = Math.max(0, Math.min(e, dimSize));
  return { start: s, end: e };
}
