import type { ResolvedTensor } from "./context.js";
import { complexIfftBasis, irfftBasis } from "./dft.js";
import { uploadFloat32 } from "webgpu-compute-kit";

/** Weights the exporter leaves out of weights.f32.bin because the runtime can rebuild
 * them: a complex Tucker layer's lifted DFT bases (`...complex_tucker.lifted_tensor_<i>`,
 * i = 0/1 complex-IFFT cos/sin, 2/3 irfft a/-b). Cached in `cache` by weight name for the
 * model's lifetime. Returns undefined for any other name. */
export function synthesizeWeight(
  device: GPUDevice,
  cache: Map<string, GPUBuffer>,
  weightName: string,
  shape: readonly number[],
): ResolvedTensor | undefined {
  const match = weightName.match(/\.complex_tucker\.lifted_tensor_(\d+)$/);
  if (!match) return undefined;

  const cached = cache.get(weightName);
  if (cached) return { buffer: cached, shape: [...shape] };

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

  const buffer = uploadFloat32(device, data);
  cache.set(weightName, buffer);
  return { buffer, shape: [...shape] };
}
