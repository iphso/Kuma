// Same bootstrap as webgpu-compute-kit/test/gpu/setup.ts: pick the least busy NVIDIA GPU
// (MESA_VK_DEVICE_SELECT overrides), then install Dawn's WebGPU globals.
import { installNodeWebGpu, selectNvidiaGpu } from "webgpu-compute-kit/node-gpu";

await selectNvidiaGpu("auto");
await installNodeWebGpu();
