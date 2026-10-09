// Real-GPU test setup: pick the least busy NVIDIA GPU with headroom (or honour an explicit
// MESA_VK_DEVICE_SELECT), then install Dawn's WebGPU globals so requestComputeDevice()
// and the browser-style runtime code run unchanged. The `webgpu` package resolves through
// webgpu-compute-kit's own node_modules (it is an optional peer of both packages).
import { installNodeWebGpu, selectNvidiaGpu } from "webgpu-compute-kit/node-gpu";

await selectNvidiaGpu("auto");
await installNodeWebGpu();
