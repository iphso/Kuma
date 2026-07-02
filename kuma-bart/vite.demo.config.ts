import { defineConfig, type Plugin } from "vite";
import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";

// process.cwd(), not an import.meta.url-relative path: Vite bundles config files
// through a temporary esbuild step before evaluating them, which can make
// import.meta.url inside a *config* file resolve relative to that temp location
// instead of this file's real location -- a well-known Vite gotcha specific to config
// files (app source under demo/ doesn't have this problem). The dev server's process
// is always started from the package root (kuma-bart/ locally, /workspace in
// Docker -- see Dockerfile's WORKDIR), so cwd-relative is the reliable pattern here.
const ARTIFACTS_DIR = resolve(process.cwd(), "demo/artifacts");

/** Dev-server-only endpoint (GET /api/artifacts -> string[] of filenames) so the demo
 * can populate a dropdown of available .iph files instead of asking for a hand-typed
 * path. No production build counterpart needed -- this demo is never static-built and
 * deployed, only run via `vite`/`vite --host` (see kuma-bart/README.md). */
function artifactsApiPlugin(): Plugin {
  return {
    name: "kuma-bart-artifacts-api",
    configureServer(server) {
      server.middlewares.use("/api/artifacts", (_req, res) => {
        let files: string[] = [];
        try {
          files = readdirSync(ARTIFACTS_DIR)
            .filter((name) => name.endsWith(".iph"))
            .sort();
        } catch (err) {
          // demo/artifacts doesn't exist (e.g. no bind mount outside Docker) -- empty
          // list for the browser, but logged here so a real misconfiguration (wrong
          // path, permissions) doesn't silently look identical to "just no files yet."
          console.error(`[kuma-bart-artifacts-api] couldn't read ${ARTIFACTS_DIR}:`, err);
          files = [];
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(files));
      });
    },
  };
}

/** Serves onnxruntime-web's WASM binaries at root-relative URLs (e.g.
 * /ort-wasm-simd-threaded.jsep.wasm) so the browser can fetch them.  Vite's dev server
 * doesn't serve node_modules as static files by default, so without this the WASM fetch
 * returns a 404 HTML page, which WebAssembly.instantiate() then rejects with "expected
 * magic word 00 61 73 6d".  ort.env.wasm.wasmPaths is set to "/" in OnnxModel (see
 * src/onnx/model.ts) to steer all WASM fetches here. */
function onnxWasmPlugin(): Plugin {
  const wasmDir = join(process.cwd(), "node_modules/onnxruntime-web/dist");
  return {
    name: "onnx-wasm",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // Strip query string (?import, ?v=... etc.) Vite may append to the URL.
        const filename = (req.url ?? "").split("?")[0]!.split("/").pop()!;
        // Only intercept ort's own WASM and JSEP JS files; leave everything else
        // to Vite's module pipeline.
        if (filename.startsWith("ort-") && (filename.endsWith(".wasm") || filename.endsWith(".mjs"))) {
          try {
            const content = readFileSync(join(wasmDir, filename));
            const contentType = filename.endsWith(".wasm") ? "application/wasm" : "text/javascript";
            res.setHeader("Content-Type", contentType);
            res.end(content);
            return;
          } catch {
            // file not in onnxruntime-web/dist — fall through to Vite's handler
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  // demo/artifacts is bind-mounted from the repo root's artifacts/ by docker-compose's
  // bart-demo service, so it's servable directly at /artifacts/... with no fs.allow
  // overrides needed.
  root: "demo",
  plugins: [artifactsApiPlugin(), onnxWasmPlugin()],
  optimizeDeps: {
    // onnxruntime-web must NOT be pre-bundled by Vite's esbuild optimizer.
    // esbuild runs in Node.js where `typeof navigator === "undefined"` is true, so
    // it dead-code-eliminates the entire WebGPU branch — leaving only the
    // `throw new Error("WebGPU is not supported")` path in the output bundle.
    // Excluding the package means the browser gets ort.mjs as-is and evaluates
    // the navigator check at runtime, where navigator.gpu is actually available.
    exclude: ["onnxruntime-web"],
  },
});
