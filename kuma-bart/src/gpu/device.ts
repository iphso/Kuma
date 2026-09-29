import { requestComputeDevice, isLikelyLowMemoryDevice } from "webgpu-compute-kit";

export { isLikelyLowMemoryDevice };

/** Requests a GPUDevice via webgpu-compute-kit's requestComputeDevice (adapter/device
 * setup, opportunistic timestamp-query enablement, low-memory heuristic, and an opt-in
 * buffer-leak tracker now toggled via `?wgpumem` / `localStorage.wgpumem`, superseding
 * this project's old `?kumamem` flag now that the instrumentation lives in the shared
 * package). Kept as a thin wrapper (rather than switching every call site to the kit's
 * `requestComputeDevice` directly) so model.ts's public surface doesn't change. */
export async function requestKumaDevice(): Promise<{ device: GPUDevice; isLikelyLowMemory: boolean }> {
  const { device, isLikelyLowMemory } = await requestComputeDevice();
  return { device, isLikelyLowMemory };
}
