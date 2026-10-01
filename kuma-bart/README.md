# kuma-bart

WebGPU runtime for `.iph` packages produced by the `kuma` Python compiler. For what this project is and how the pipeline works, see the [root README](../README.md). This file covers the runtime internals and design decisions — useful if you're changing something and want to know why it's the way it is.

## Build / test / demo

```bash
# From repo root (recommended)
docker compose run --rm bart-test    # typecheck + vitest
docker compose up bart-demo          # serves demo/ at localhost:5173

# Or directly with npm (requires Node 20+)
cd kuma-bart && npm install
npm run test     # typecheck + vitest
npm run demo     # vite dev server
npm run build    # library build → dist/
```

`bart-demo` bind-mounts `artifacts/` read-only at `demo/artifacts/`. Run `docker compose run --rm export` first to populate it, then select a model from the dropdown.

## Architecture

kuma-bart builds on [`webgpu-compute-kit`](../../webgpu-compute-kit), the shared, graph-agnostic WebGPU plumbing (device setup, buffer helpers, the buffer pool and generation pipelining, the pipeline cache/dispatcher, timestamp profiling, zip-pack loading). It is a `file:../../webgpu-compute-kit` dependency, so it must be checked out as a sibling of this repo (`~/Projects/Kuma` next to `~/Projects/webgpu-compute-kit`) with its `dist/` built (`npm run build` there). Generic WebGPU mechanics belong in the kit; this package keeps only what is specific to `.iph` graphs.

```
src/
  index.ts            Public API (KumaModel, OnnxModel, releaseOutputs, error types,
                      manifest/golden/report types). Does not import the player.
  model.ts            KumaModel: load, run/runRaw/runToGpu, verify, profile, warmUp
  errors.ts           KumaManifestError / KumaShapeError / KumaUnsupportedOpError
  iph/loader.ts       .iph loading on the kit's loadZipPack: manifest, weights,
                      kernels/*.wgsl, snippets/*.js, golden.json; peekIphFormat
  onnx/model.ts       OnnxModel: "onnx"/"onnx-branching" .iph packages via onnxruntime-web

  player/             (separate entry: "kuma-bart/player")
    KumaPlayer.ts     <kuma-player> web component (see root README for usage)
    backends.ts       PlayerBackend: KumaBackend (WebGPU) / OnnxBackend (2D canvas)
    frameScheduler.ts LatestWinsScheduler: in-flight-capped, latest-wins frame submission
    gpuRender.ts      GPU-direct canvas rendering (no CPU pixel round-trip)
    styles.ts         Shadow-DOM styles and template

  engine/
    scheduler.ts      encodeGraph (switch resolution, topological walk, per-node dispatch,
                      early-free) and runGraph (encodeGraph + readback + cleanup)
    context.ts        OpContext — per-node GPU mechanics (pooled buffers, cached
                      constants/uniforms, dispatch) handed to every op handler
    profile.ts        profileGraph — encodeGraph with one timestamped pass per node
    verify.ts         Golden-value verification + warmUp
    synthetic_weights.ts  Rebuilds weights the exporter omits (complex-Tucker DFT bases)
    snippets.ts       Sandboxed eval of js_snippet nodes (branch-routing logic)
    dft.ts            DFT basis matrices for FFT-family ops
    einsum-plan.ts    Einsum subscript → concrete dispatch plan
    shape.ts          Shape/stride helpers
    stats.ts          Tensor summary stats used by verify

  ops/                One file per op family, each exporting an OpHandler registered
                      in ops/index.ts's opRegistry; complex_math.ts holds the shared
                      binary/unary dispatch and complex-product helpers

  gpu/params.ts       Uniform buffer packing (u32-only and mixed u32/f32 structs)
  types/              TypeScript types for manifest.json and golden.json

test/                 vitest unit tests (mock-gpu.ts + graph-builder.ts for graph tests)
demo/                 Interactive player (thin shell around <kuma-player>)
```

## Key design decisions

### Buffer pooling

`OpContext.createBuffer` doesn't allocate a fresh GPU buffer on every call. Each node gets a `BUFFER_POOL_DEPTH`-deep ring of pre-allocated buffers (default depth 3), reused forever. This works because kuma is a static-shape compiler — the same node always produces the same shape, so the same buffer is always the right size. Buffers at or above the pool's size threshold (default 1 MB) bypass the pool and are allocated fresh each call.

Without pooling, a ~270-node model at 60fps allocates and immediately destroys hundreds of `GPUBuffer`/bind-group/uniform objects per frame. That's enough sustained pressure to trigger real V8 GC pauses — confirmed via Chrome DevTools Performance recordings where GC surges correlated directly with frame drops.

A generation counter (the kit's `acquireGenerationSlot`/`releaseGenerationSlot`) keeps reuse safe under multi-frame pipelining: a buffer slot isn't handed out again until `onSubmittedWorkDone()` confirms the previous frame using it has completed. That wait only actually happens when the pool is genuinely exhausted — it's not paid on every call.

### Buffer ownership

One rule: whoever allocates a buffer frees it. `runGraph` destroys only what it allocated itself — fresh above-threshold buffers and switch-seeded branch inputs — never weights, pooled buffers, cached constants, or caller-provided `inputBuffers`. `KumaModel` owns the input buffers it uploads and destroys them after each call. Outputs handed back to the caller may be fresh allocations: call `RunGraphOutput.release()` (or `releaseOutputs`) when done with `buffer`; it is a no-op for pooled, weight, and input buffers. `run()`, `verify()`, `warmUp()` and `profile()` release their own outputs; `runRaw()`/`runToGpu()` callers must.

### Multi-frame pipelining

`device.queue.onSubmittedWorkDone()` has a measured ~30ms latency in at least one common browser WebGPU implementation — independent of how much GPU work is involved. A GPU-side timestamp around a real inference pass showed ~7ms of actual compute; the same 30ms appeared on a 16-byte debug readback.

Waiting for that signal after every frame serializes 30ms of fixed overhead per frame, which caps throughput at ~33fps regardless of GPU speed. Allowing pool-depth frames in flight simultaneously lets that latency overlap with submission of the next frame. `<kuma-player>` caps its in-flight frames at `model.maxInFlight` (the loaded model's pool depth) — submitting more doesn't buy more real concurrency, since `acquireGenerationSlot` would block internally anyway.

### `runToGpu` / skipping the CPU readback

Reading a multi-megabyte output buffer back to the CPU (`mapAsync`) measured at 60–70ms on real hardware — far more than the GPU compute itself. A caller that only needs to render the result (which can stay entirely on GPU — see `player/gpuRender.ts` for a direct-from-storage-buffer render pass) should call `runToGpu()`, not `run()`.

### Pipeline warm-up

The first use of a compute pipeline triggers WebGPU shader compilation. For a multi-branch model, that happens the first time each branch gets exercised during real playback — landing as a stutter exactly at segment boundaries.

`model.warmUp()` runs every switch branch once (against `golden.json`'s reference inputs, bypassing the time-based router) to populate the pipeline cache and buffer pool ahead of interactive use. Shader compilation happens during the already-expected load wait.

## Known gaps

- No general memory planner. Liveness is handled only partially: large (non-pooled) intermediates are freed as soon as their last consumer is encoded (batched into ~64 MB groups), but there is no allocator that reuses memory across nodes with disjoint live ranges, and pooled buffers are a fixed-depth ring per node.
- No automatic kernel fusion. A few op-specific specializations exist (tiled matmul, halo-tiled depthwise conv, parallel-reduction groupnorm) but nothing that crosses op boundaries.
- `ConvTranspose2d` kernel is missing — throws `KumaUnsupportedOpError` if a manifest needs one.

Both a real planner and fusion need new information in the `.iph` format (live-range annotations, fusable-subgraph markers) that the compiler doesn't emit yet. That's Step 2 work on both sides of the pipeline.
