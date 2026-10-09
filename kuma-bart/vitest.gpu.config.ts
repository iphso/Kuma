import { defineConfig } from "vitest/config";

// Tests that run real .iph packages on a real GPU, in Node (Dawn via the `webgpu`
// package; see webgpu-compute-kit/bench/node-gpu.mjs). `npm run test:gpu`.
export default defineConfig({
  test: {
    include: ["test/gpu/**/*.test.ts"],
    environment: "node",
    setupFiles: ["./test/gpu/setup.ts"],
    testTimeout: 600_000,
    hookTimeout: 600_000,
    // One GPU device per file; files in parallel would just contend for the same GPU.
    fileParallelism: false,
  },
});
