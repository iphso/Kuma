// Thin re-export of webgpu-compute-kit's generic buffer helpers, under this project's
// original names -- `uploadWeightSlice` is kuma's name for the kit's more general
// `uploadBufferSlice` (uploading a byte range sliced out of a larger blob; kuma always
// uses it for a weight's slice of the packed weights file, hence the name here).
export { createStorageBuffer, createUniformBuffer, uploadFloat32, zeroBuffer, readBuffers } from "webgpu-compute-kit";
export { uploadBufferSlice as uploadWeightSlice } from "webgpu-compute-kit";
