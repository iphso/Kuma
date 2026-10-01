# Kuma — Kernel-Unified Model Artifact

Run a trained PyTorch model **live in your browser**, on the GPU, with no Python server and no runtime dependencies beyond WebGPU. (The default path doesn't involve ONNX at all; ONNX is only an optional compile-time route — `backend="onnx"` — plus a separate raw-ONNX package format that kuma-bart runs through onnxruntime-web.)

The idea is simple: take a model you already have, compile it once into a self-contained package, and load that package directly into any browser that supports WebGPU. The package carries everything — the computation graph, the weights, and the WGSL compute shaders to execute it. No server-side inference. No heavyweight runtime. Just a file and a browser.

```
PyTorch model
    ↓  torch.export  (captures the computation graph)
kuma  (Python, this repo)
    ↓  packs weights, embeds shaders, serializes graph
model.iph  (a zip — manifest + weights + WGSL kernels)
    ↓  kuma-bart  (TypeScript, also this repo)
WebGPU compute kernels
    ↓
pixels on a <canvas>, at 60fps
```

A real multi-branch ConvNeXt-style model — 4 segments, ~270 dispatched ops — currently plays back interactively at full framerate.

---

## Two halves of the pipeline

**`kuma`** (Python, `src/kuma/`) is the compiler. It takes a PyTorch model, traces it with `torch.export`, packs the weights into a contiguous float32 binary blob, and writes everything out as a `.iph` package. It doesn't run anything — it just produces the file.

**`kuma-bart`** (TypeScript, `kuma-bart/`) is the runtime. It loads a `.iph` package in the browser, uploads the weights to GPU buffers, and walks the graph — dispatching one WebGPU compute pass per node per inference call. It also exposes a `<kuma-player>` web component you can drop into any page.

Neither half knows much about the other. The `.iph` format is the whole contract between them.

---

## Status

Functional, and genuinely fast enough to be useful. The things that work:

- **Full inference loop** — loads a `.iph`, runs it on the GPU, renders the output to a canvas with no CPU round-trip.
- **Multi-branch models** — switch-routed models (e.g. a time-sliced animation with distinct segments) are supported, with branch-specific warm-up to avoid shader-compile stutters at boundary crossings.
- **60fps playback** — the interactive player scrubs a normalized `[0, 1]` time input and plays back at up to 60fps, using multi-frame pipelining to overlap GPU-completion latency with submission of the next frame.
- **Verification** — `model.verify()` checks inference output against eager-PyTorch reference values embedded in the package at export time.
- **Per-op profiling** — `model.profile()` gives a GPU-timestamped breakdown of time by op type.

The things that aren't here yet:

- **No kernel fusion.** Each op is a separate compute pass. A few hand-specialized kernels exist (tiled matmul, halo-tiled depthwise conv, parallel-reduction norm), but nothing that fuses across op boundaries automatically.
- **No general memory/buffer planner.** The buffer pool is a fixed-depth reuse ring per node — it works, and it eliminates the per-frame allocation pressure that was causing real GC pauses, but it isn't a planner that reasons about live ranges across the whole graph. That needs new information in the `.iph` format itself, which the compiler doesn't emit yet.

Both of those are the same Step 2 work, and the reason they're not here is that they require changes to the format (live-range annotations, fusable-subgraph markers) before the runtime can do anything useful with them.

---

## The `.iph` format

A `.iph` file is a zip archive. Unzip it and you get:

| File | What it is |
|---|---|
| `manifest.json` | Inputs, outputs, weight table, full computation graph as JSON, warnings |
| `weights.f32.bin` | One flat, contiguous little-endian float32 blob for all parameters |
| `kernels/*.wgsl` | WGSL compute shader source for every op family |
| `debug_report.md` | Human-readable summary: param count, op table, weight table |

The manifest carries the full FX graph from `torch.export` — every node, its op target (`aten.convolution.default`, etc.), its arguments (tensor args become `{"node_ref": "<node_name>"}`), and a `weight_name` on placeholder nodes that maps back into the weight table. To reconstruct a weight: slice `weights.f32.bin` at `[byte_offset : byte_offset + byte_length]`, interpret as little-endian float32, reshape to `shape`.

The kernels are embedded in every package rather than fetched from a CDN or compiled at runtime — the package is intentionally self-contained, loadable from a plain file server with no build step.

`playback` metadata (`fps`, `duration_seconds`) is optional, only present if the caller provided it at export time. If your model takes a normalized time input and you want the player to know how fast to run it, pass those values through — they're not recoverable from the graph otherwise.

---

## Python: exporting a model

### Install

This repo uses Docker for its own dev/test environment (no local Python required), but to **use** `kuma` from your own project you just pip-install it wherever you have Python + PyTorch:

```bash
# local checkout (recommended while developing both repos)
pip install -e /path/to/kuma

# or from git
pip install git+https://github.com/yourorg/kuma.git
```

Requires `torch >= 2.1`, `numpy >= 1.24` and `onnx >= 1.14` (onnx is a hard dependency: the optional ONNX export paths and their golden capture use it).

### Basic export

```python
import torch
import kuma

model = build_model(cfg)
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)  # must match real input shape and dtype

kuma.export_model(model, example_inputs, out="model.iph")
```

That's the one-call happy path. It runs `torch.export`, validates dtypes and devices (fails loudly on float16, bfloat16, non-CPU tensors, or dynamic shapes — none of those are supported), packs the weights, and writes the package.

### With playback metadata

If your model takes a scalar time input in `[0, 1]` that maps to a physical duration:

```python
kuma.export_model(
    model, example_inputs,
    out="model.iph",
    fps=30.0,
    duration_seconds=total_frames / 30.0,
)
```

The player uses `duration_seconds` to set the sweep speed. Without it, it defaults to a 6-second sweep, which may or may not match your model's intended playback rate.

### Lower-level API

```python
# Start from an ExportedProgram you already have
ep = torch.export.export(model.eval(), example_inputs)
kuma.export_exported_program(ep, out="model.iph")

# Or compile without writing to disk — useful for inspection
package = kuma.compile(ep)
package.save("model.iph")       # the self-contained zip
package.write_dir("debug/")     # loose files for easier inspection
```

`Package` has `manifest` (dict), `weights_blob` (bytes), `debug_report` (str), `kernels` (dict of WGSL source), and `skipped` (list of non-float32 tensors that were dropped, like `bn.num_batches_tracked`).

### Scope

- Inference only. Static shapes. Float32 only.
- Raises `ValueError`/`TypeError` loudly on anything outside that: float16, bfloat16, non-CPU tensors, non-`nn.Module` inputs.
- Op coverage on the compiler side is whatever `torch.export` traces cleanly: Conv2d, ConvTranspose2d, Linear, BatchNorm, LayerNorm, GroupNorm, GELU, ReLU, SiLU, pooling, upsampling, reshape, permute, concat, slice all *compile* into a package. Running is a different question — kuma-bart's opRegistry is the authority, and it currently has **no handler for pooling, upsampling, BatchNorm, or transposed convolution** (the WGSL kernels ship in the package, but nothing dispatches them), so a package using those compiles fine and then fails in kuma-bart with `KumaUnsupportedOpError` when loaded/run, rather than silently producing wrong results.
- The manifest's `unsupported_ops` list is informational only (always empty on the torch path; on the ONNX path it lists ONNX op types that couldn't be converted). kuma-bart does not read it. On the ONNX path, any unconvertible op that's actually reachable from the output is a compile-time `ValueError` instead.

---

## Browser: loading and running a model

### JS API

```ts
import { KumaModel } from "kuma-bart";

const model = await KumaModel.load("/path/to/model.iph");

// Warm up first if you care about smooth first-frame experience
await model.warmUp();

// CPU-readback result (use this if you need the values in JS)
const outputs = await model.run({ time: new Float32Array([0.5]) });

// GPU-only path — no CPU round-trip, for rendering straight to a canvas
const gpuOutputs = await model.runToGpu({ time: new Float32Array([0.5]) });
// gpuOutputs[0].buffer is a GPUBuffer on model.gpuDevice

// Verification against embedded golden values
const report = await model.verify();

// Per-op GPU timing breakdown
const profile = await model.profile({ time: new Float32Array([0.5]) });
```

`KumaModel.load` accepts a path string, a `Response`, or an `ArrayBuffer` — so it works equally well fetching from a server or reading a file the user dropped into the page.

### `<kuma-player>` web component

For interactive playback, there's a drop-in web component. Import it once, then use the element anywhere:

```html
<script type="module">
  import "kuma-bart/player"; // registers <kuma-player>
</script>

<kuma-player src="/artifacts/model.iph"></kuma-player>
```

Attributes:

| Attribute | What it does |
|---|---|
| `src` | Path to a `.iph` file — triggers load when changed |
| `debug` | (boolean) Show the verify/profile debug panel on mount |
| `autoplay` | (boolean) Start playing immediately after load and warm-up |

CSS custom properties for theming:

```css
kuma-player {
  --kp-accent: #7c6ff7;  /* progress bar color, default white */
  --kp-radius: 8px;      /* card corner radius, default 12px */
}
```

In Astro, React, or any other framework — it's a plain custom element, so it works anywhere:

```astro
---
// Astro example
---
<kuma-player src={Astro.props.modelPath}></kuma-player>
<script>import "kuma-bart/player"</script>
```

### Installing kuma-bart

```bash
npm install /path/to/kuma-bart   # local checkout
# or once published:
npm install kuma-bart
```

Build the library first (`npm run build` inside `kuma-bart/`, or `docker compose run --rm bart-test`), then `dist/` is what gets installed.

---

## Running the dev environment

Everything is Docker-based. From the repo root:

```bash
# Build images
docker compose build

# Run the Python test suite
docker compose run --rm test

# Export the example model to artifacts/simple.iph
docker compose run --rm export

# Run kuma-bart's typecheck + vitest suite
docker compose run --rm bart-test

# Serve the interactive demo at localhost:5173
# (loads .iph files from artifacts/ — run the export step first)
docker compose up bart-demo
```

The demo's model picker lists whatever `.iph` files are in `artifacts/`. Several models are checked in there (`beauty.iph`, `bosphorus.iph`, `bunny.iph`, `honey.iph`, `jockey.iph`, `ready.iph`, `shake.iph`, `yacht.iph`, their `half*.iph` variants, and the tiny acceptance-test `simple.iph`), so you can spin up `bart-demo` and start poking around without running the export step first.

---

## How it works (the interesting parts)

### Buffer pooling and why it matters

Without pooling, a real model (~270 dispatched nodes) allocates and immediately destroys hundreds of `GPUBuffer`/bind-group/uniform-buffer objects every frame at 60fps. That's enough sustained pressure to trigger real V8 GC pauses — confirmed with Chrome DevTools Performance recordings where GC surges landed exactly where frame drops appeared.

The fix: every node's output comes from a small ring of pre-allocated GPU buffers (`BUFFER_POOL_DEPTH` deep), reused on every call. This works because kuma is a **static-shape** compiler — the same node always produces the same shape, so the same buffer is always the right size. A generation counter tracks which slots are still in use under multi-frame pipelining, so a buffer is never handed out again until the GPU confirms the previous frame using it has completed.

### The ~30ms GPU-completion latency problem

`device.queue.onSubmittedWorkDone()` has a measured ~30ms latency in at least one major browser's WebGPU implementation — independent of how much actual GPU work was involved. A 16-byte debug readback took the same ~30ms as a full inference frame. Meanwhile, GPU-side timestamps showed actual compute finishing in ~7ms.

Waiting for that notification after every single frame means every frame pays 30ms in fixed overhead, capping throughput at ~33fps regardless of GPU speed. The fix is standard double/triple buffering: allow a small bounded number of frames in flight simultaneously so that 30ms latency overlaps with submission of the next frame instead of blocking it. The bound matches `BUFFER_POOL_DEPTH` so the demo-level cap and the library's internal buffer pool stay in sync.

### Shader warm-up

The first time a compute pipeline gets used, WebGPU compiles the WGSL shader. For a model with multiple branches, that happens the first time each branch is exercised during real playback — producing a visible stutter exactly at segment boundaries, the worst possible moment.

`model.warmUp()` runs every branch once against the embedded reference inputs before any interactive use. Shader compilation happens during the already-expected load wait, not mid-playback.

---

## Repo layout

```
src/kuma/           Python compiler
  __init__.py         Public API: export_model, export_exported_program, compile, Package
  compiler.py         compile(ep) -> Package; the main pipeline
  pack_weights.py     Parameters → contiguous float32 blob
  manifest.py         Manifest dict builder
  golden.py           golden.json capture (eager-run per-node value stats for verify())
  branching.py        compile_branching: N traced branches + JS router -> one switch package
  branch_assembly.py  Router/switch graph + branch namespacing shared by all branching paths
  onnx_backend.py     export_via_onnx / compile_branching_onnx_kuma: torch -> ONNX -> Kuma graph
  onnx_ops.py         ONNX op -> aten target converters
  onnx_compiler.py    compile_onnx / compile_branching_onnx: raw-ONNX packages (onnxruntime-web)
  kernels/            Embedded WGSL shaders (bundled into every .iph)

kuma-bart/          TypeScript WebGPU runtime
  src/
    model.ts          KumaModel: load, run, runToGpu, verify, profile, warmUp
    player/
      KumaPlayer.ts   <kuma-player> web component
    engine/           Interpreter loop, buffer pooling, profiler, verifier
    ops/              One file per op family (conv, linear, norm, activation, ...)
    gpu/              Device init, buffer helpers, params packing
  demo/               Interactive browser demo
  test/               vitest unit tests

examples/           Acceptance-test model and export runner
artifacts/          Output directory — NOT git-ignored: the demo .iph models and simple.iph are committed here
```
