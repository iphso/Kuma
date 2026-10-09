import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // Real-GPU tests (Dawn in Node) run under vitest.gpu.config.ts: `npm run test:gpu`.
    exclude: [...configDefaults.exclude, "test/gpu/**"],
    environment: "node",
    setupFiles: ["./test/setup.ts"],
  },
});
