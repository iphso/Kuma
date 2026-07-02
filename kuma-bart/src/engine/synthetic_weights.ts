import type { ResolvedTensor } from "./context.js";
import { complexIfftBasis, irfftBasis } from "./dft.js";
import { uploadFloat32 } from "../gpu/buffers.js";

export function synthesizeWeight(
  device: GPUDevice,
  cache: Map<string, GPUBuffer>,
  weightName: string,
  shape: readonly number[],
): ResolvedTensor | undefined {
  const match = weightName.match(/\.complex_tucker\.lifted_tensor_(\d+)$/);
  if (!match) return undefined;

  const index = Number(match[1]);
  let data: Float32Array | undefined;

  if ((index === 0 || index === 1) && shape.length === 2 && shape[0] === shape[1]) {
    const basis = complexIfftBasis(shape[0]!);
    data = index === 0 ? basis.cos : basis.sin;
  } else if ((index === 2 || index === 3) && shape.length === 2) {
    const basis = irfftBasis(shape[0]!);
    data = index === 2 ? basis.a : new Float32Array(basis.b).map((v) => -v);
  }

  if (!data) return undefined;

  let buffer = cache.get(weightName);
  if (!buffer) {
    buffer = uploadFloat32(device, data);
    cache.set(weightName, buffer);
  }
  return { buffer, shape: [...shape] };
}
