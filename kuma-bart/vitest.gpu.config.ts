import { defineConfig } from "vitest/config";

// Tests that run .iph packages on a real GPU, in Node (Dawn via the `webgpu` package,
// through webgpu-compute-kit/node-gpu). `npm run test:gpu`. Mirrors the kit's own.
export default defineConfig({
  test: {
    include: ["test/gpu/**/*.test.ts"],
    environment: "node",
    setupFiles: ["./test/gpu/setup.ts"],
    testTimeout: 120_000,
    // One GPU device per file; files in parallel would just contend for the same GPU.
    fileParallelism: false,
  },
});
