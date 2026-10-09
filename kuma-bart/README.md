# kuma-bart

WebGPU runtime for `.iph` packages produced by the `kuma` Python compiler. For what this project is and how the pipeline works, see the [root README](../README.md). This file covers the runtime internals and design decisions — useful if you're changing something and want to know why it's the way it is.

## Build / test / demo

```bash
# From repo root (recommended)
docker compose run --rm bart-test    # typecheck + vitest
docker compose up bart-demo          # serves demo/ at localhost:5173

# Or directly with npm (requires Node 20+)
cd kuma-bart && npm install
npm run test     # typecheck + vitest (mock GPU)
npm run test:gpu # test/gpu/ on a real GPU: Dawn in Node via webgpu-compute-kit/node-gpu
npm run demo     # vite dev server
npm run build    # library build → dist/
```

`test:gpu` needs the optional `webgpu` package (0.3.10, the newest whose native module loads on Ubuntu 22.04), which resolves through the sibling kit's `node_modules`. It picks the least busy NVIDIA GPU with 8 GB free (`selectNvidiaGpu("auto")`; set `MESA_VK_DEVICE_SELECT` to pin one) and runs against the committed `artifacts/simple.iph` and `artifacts/halfshake.iph`.

`bart-demo` bind-mounts `artifacts/` read-only at `demo/artifacts/`. Run `docker compose run --rm export` first to populate it, then select a model from the dropdown.

## Architecture

kuma-bart builds on [`webgpu-compute-kit`](../../webgpu-compute-kit), the shared, graph-agnostic WebGPU plumbing (device setup, buffer helpers, the buffer pool and generation pipelining, the pipeline cache/dispatcher, timestamp profiling, zip-pack loading). It is a `file:../../webgpu-compute-kit` dependency, so it must be checked out as a sibling of this repo (`~/Projects/Kuma` next to `~/Projects/webgpu-compute-kit`) with its `dist/` built (`npm run build` there). Generic WebGPU mechanics belong in the kit; this package keeps only what is specific to `.iph` graphs.

```
src/
  index.ts            Public API (KumaModel, OnnxModel, releaseOutputs, error types,
                      manifest/golden/report types). Does not import the player.
  model.ts            KumaModel: load (own or caller-provided device), encode (record into a
                      caller's encoder), run/runRaw/runToGpu (encode + submit), verify, profile,
                      warmUp, trimScratch
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
    scheduler.ts      walkGraph (switch resolution, topological walk, per-node dispatch,
                      liveness) under two owners: recordGraph (caller's encoder, TransientSet
                      scratch, never submits) and encodeGraph (own encoder, ring pool, submits)
                      + runGraph (encodeGraph + readback + cleanup)
    transients.ts     TransientSet — one encode's scratch: size-keyed free list, recycled
                      within the encode and reset after submit
    context.ts        OpContext — per-node GPU mechanics (scratch via ScratchAllocator, cached
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
  types/              TypeScript types for manifest.json, golden.json and GpuTensor

test/                 vitest unit tests (mock-gpu.ts + graph-builder.ts for graph tests)
test/gpu/             real-GPU tests (Dawn in Node), see `npm run test:gpu`
demo/                 Interactive player (thin shell around <kuma-player>)
```

## Key design decisions

### Shared-device mode and `encode()`

`KumaModel.load(source, { device })` puts the model on a caller's `GPUDevice` (nothing is requested, and `destroy()` frees the model's buffers without touching the device). `model.encode(inputs, encoder, opts)` then records the whole graph into the caller's `GPUCommandEncoder`, synchronously, and never submits or waits — the caller's own kernels can produce the inputs and consume the outputs inside the same submission, and any number of encodes can go into one encoder. `run()`/`runRaw()`/`runToGpu()` are this path plus the input upload, one submit and (for `run`/`runRaw`) the readback. `GpuTensor` (`types/tensor.ts`) is the contract on both sides; the root README's "Shared-device mode" section is the user-facing description.

What encode() cannot check: that a buffer belongs to `model.gpuDevice`. WebGPU exposes no buffer→device query, so a foreign buffer fails as an asynchronous validation error (`uncapturederror` / `popErrorScope`). Everything else about an input is checked up front and thrown synchronously as `KumaShapeError`/`KumaManifestError`, before anything is recorded (except an offset input's staging copy).

### Scratch sets (buffer pooling)

`OpContext.createBuffer` doesn't allocate a fresh GPU buffer on every call. For `encode()` (and so `run*`), every intermediate comes from a `TransientSet` (`engine/transients.ts`) that the call holds exclusively from `encode()` until `release()`: a free list keyed by buffer size, so a buffer whose last consumer has been recorded is reused by the next same-size node inside the same command buffer (compute passes execute in order within a submission, so this needs no fence), and the whole set is reset on `release()` for the next call. A set's footprint is roughly the encode's peak live activation bytes; sets are kept for reuse and allocated on demand, one per call in flight, which is what makes two encodes recorded into one encoder before submission safe — they never share a buffer. `maxInFlight` (`bufferPoolDepth`, default 3) is the recommended cap, not a block: each unreleased call holds a live activation set. `trimScratch()` frees idle sets after a burst.

Without this, a ~270-node model at 60fps allocates and immediately destroys hundreds of `GPUBuffer`/bind-group/uniform objects per frame. That's enough sustained pressure to trigger real V8 GC pauses — confirmed via Chrome DevTools Performance recordings where GC surges correlated directly with frame drops.

`verify()`, `warmUp()` and `profile()` go through the older `encodeGraph`/`runGraph` path: a per-node ring of `BUFFER_POOL_DEPTH` pre-allocated buffers, reused across calls under the kit's generation counter (`acquireGenerationSlot`/`releaseGenerationSlot`, which waits on `onSubmittedWorkDone()` only when the ring is genuinely exhausted), with buffers at or above the pool's size threshold (default 1 MB) allocated fresh and freed as soon as their last consumer is encoded. That path owns its command encoder, which is why it keeps submitting mid-graph to reclaim memory and why it is not what `encode()` uses.

A handler that allocates an output and dispatches nothing (`aten.zeros_like`) is relying on fresh-buffer zero fill; both paths detect that (`OpContext.dispatchCount`) and `clearBuffer` it, since a recycled buffer is not zero.

### Buffer ownership

One rule: whoever allocates a buffer frees it. `runGraph` destroys only what it allocated itself — fresh above-threshold buffers and switch-seeded branch inputs — never weights, pooled buffers, cached constants, or caller-provided `inputBuffers`. `recordGraph` destroys nothing: its scratch belongs to the `TransientSet` the caller passed. `KumaModel` owns the input buffers it uploads for `run*` and frees them with the call's scratch. Outputs handed back by `runRaw()`/`runToGpu()` are scratch: call every output's `release()` (or `releaseOutputs`) and the call's set goes back to the model. `encode()` outputs are scratch too (valid until `release()`), unless redirected into caller buffers via `opts.outputs`. `run()`, `verify()`, `warmUp()` and `profile()` release their own outputs.

### Multi-frame pipelining

`device.queue.onSubmittedWorkDone()` has a measured ~30ms latency in at least one common browser WebGPU implementation — independent of how much GPU work is involved. A GPU-side timestamp around a real inference pass showed ~7ms of actual compute; the same 30ms appeared on a 16-byte debug readback.

Waiting for that signal after every frame serializes 30ms of fixed overhead per frame, which caps throughput at ~33fps regardless of GPU speed. Allowing pool-depth frames in flight simultaneously lets that latency overlap with submission of the next frame. `<kuma-player>` caps its in-flight frames at `model.maxInFlight` (the loaded model's configured depth) — submitting more doesn't buy more real concurrency, and each extra frame in flight holds another scratch set of live activations.

### `runToGpu` / skipping the CPU readback

Reading a multi-megabyte output buffer back to the CPU (`mapAsync`) measured at 60–70ms on real hardware — far more than the GPU compute itself. A caller that only needs to render the result (which can stay entirely on GPU — see `player/gpuRender.ts` for a direct-from-storage-buffer render pass) should call `runToGpu()`, not `run()`.

### Pipeline warm-up

The first use of a compute pipeline triggers WebGPU shader compilation. For a multi-branch model, that happens the first time each branch gets exercised during real playback — landing as a stutter exactly at segment boundaries.

`model.warmUp()` runs every switch branch once (against `golden.json`'s reference inputs, bypassing the time-based router) to populate the pipeline cache and buffer pool ahead of interactive use. Shader compilation happens during the already-expected load wait.

## Known gaps

- No general memory planner. `encode()`'s scratch set reuses buffers across nodes with disjoint live ranges, but only by exact size (a greedy free list), and the ring-pool path (`verify`/`profile`) only frees large intermediates early.
- No automatic kernel fusion. A few op-specific specializations exist (tiled matmul, halo-tiled depthwise conv, parallel-reduction groupnorm) but nothing that crosses op boundaries.
- f32 only, in weights, activations and the `GpuTensor` contract. f16 would need the ops registry to carry a dtype per kernel variant (every WGSL kernel is written against `array<f32>`), `shader-f16` feature negotiation on the device, half-size weight uploads with a conversion step, and uniform/param packing that knows element size; fusion would additionally need the handlers to expose per-element expressions instead of whole kernels.
- `encode()` creates one bind group per dispatch (the kit's `dispatchLinear`), which is where its CPU cost goes: ~40 µs per node on Dawn (11 ms for a 279-node branch of `halfshake.iph`, 0.12 ms for the 4-node `simple.iph`). Caching bind groups per (pipeline, buffers) within a scratch set would cut that for a model whose set is reused every call.
- `ConvTranspose2d` kernel is missing — throws `KumaUnsupportedOpError` if a manifest needs one.

Fusion needs new information in the `.iph` format (fusable-subgraph markers) that the compiler doesn't emit yet. That's Step 2 work on both sides of the pipeline.
