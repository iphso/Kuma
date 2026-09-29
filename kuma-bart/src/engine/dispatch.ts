export { computeDispatchGrid, MAX_WORKGROUPS_PER_DIMENSION } from "webgpu-compute-kit";

/** Every kuma kernel using dispatchKernel's linear-dispatch entry point uses this
 * workgroup size and folds its linear element index as
 * `gid.x + gid.y * (num_workgroups.x * 64)` (see kuma's kernels/__init__.py — every
 * kernel was updated to this scheme together). This convention is kuma-specific (kernel
 * authoring choice, not a WebGPU constraint), so it stays here rather than in
 * webgpu-compute-kit, whose PipelineCache.dispatchLinear takes workgroup size as a
 * parameter instead of assuming one. */
export const KERNEL_WORKGROUP_SIZE = 64;
