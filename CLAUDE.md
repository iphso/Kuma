# Kuma

PyTorch `torch.export` → `.iph` compiler/exporter (Step 1 of a Torch→WebGPU compiler).

Kuma captures a PyTorch inference graph, packs its weights, and bundles everything into a
self-contained `.iph` package (a zip: `manifest.json`, `weights.f32.bin`, `kernels/*.wgsl`,
`debug_report.md`). Kuma itself is not a runtime, not a training library, and not a WebGPU
framework — the `.iph` file is the contract between this compiler and `kuma-bart`, the WebGPU
runtime that executes it. `kuma-bart/` lives in this same repo (own README at
`kuma-bart/README.md`) — this file (`CLAUDE.md`) covers the Python compiler; see
"kuma-bart (WebGPU runtime)" below for its environment/test/demo commands.

## Environment

**There is no local Python or pip on this machine.**
All Python execution, testing, and validation must happen inside Docker.

```bash
# Build image
docker compose build

# Run the acceptance test (simple model export → artifacts/simple.iph + artifacts/simple/)
docker compose run --rm export

# Run the full test suite
docker compose run --rm test

# Run a specific test file
docker compose run --rm test tests/test_convnext_block.py -v

# Or build+run in one shot
docker compose up --build
```

Artifacts land in `./artifacts/` (bind-mounted at `/workspace/artifacts` in the container).

## Testing a code change

1. Run the test suite first:
   ```bash
   docker compose run --rm test
   ```

2. Then do a manual acceptance run to read the artifacts:
   ```bash
   docker compose run --rm export \
     --out /workspace/artifacts/simple.iph \
     --out-dir /workspace/artifacts/simple
   ```

Spot-check the output:
- `artifacts/simple.iph` — the self-contained package (zip); should contain `manifest.json`,
  `weights.f32.bin`, `kernels/*.wgsl`, `debug_report.md`
- `artifacts/simple/debug_report.md` — ops table, weight table, warnings (no WebGPU-readiness
  guessing here — kuma-bart's own `opRegistry` is the authoritative answer to "is this op
  supported," not a static list in the compiler; see `kuma-bart/README.md`)
- `artifacts/simple/manifest.json` — full bundle descriptor; `byte_offset + byte_length` must be consistent
- `artifacts/simple/weights.f32.bin` — size should match sum of `byte_length` fields in manifest
- `artifacts/simple/exported_graph.json` — raw FX node list with `weight_name` on parameter placeholders (debug-dir only; not part of the `.iph` contract since the manifest already embeds the graph)

## kuma-bart (WebGPU runtime)

`kuma-bart/` is the TypeScript/WebGPU runtime that actually executes a `.iph` package —
full architecture, design rationale, and usage docs are in `kuma-bart/README.md`, not
duplicated here. Same environment situation as the Python side applies on some dev
boxes: no usable local Node (system Node can be too old for vite/vitest) and no Docker
group permissions either, so check which one actually works in your environment before
assuming a command below will run as-is.

```bash
# From repo root -- typecheck + vitest
docker compose run --rm bart-test

# From repo root -- serves the interactive demo at localhost:5173, bind-mounting this
# repo's artifacts/ (read-only) at demo/artifacts/ -- the same place `export` writes to
docker compose up bart-demo
```

The demo's default `.iph` path field expects something under `/artifacts/...` — run
`docker compose run --rm export` first (or point it at any other `.iph` already in
`./artifacts/`) before loading it in the demo.

## Using with a real Niko/Nika model (from another container)

The cleanest integration is to mount this repo into the model container and pip-install it there.

**Option A — bind mount + editable install (recommended for dev):**

In your model repo's `docker-compose.yml`, add:
```yaml
volumes:
  - /home/sl8rv/Projects/Kuma:/kuma
```
Then inside that container (e.g. in an entrypoint or interactively):
```bash
pip install -e /kuma
```

**Option B — install from git (for CI or fixed snapshots):**
```bash
pip install git+https://github.com/yourorg/kuma.git
```

**Direct Python API.** There is no installed CLI (the `export` Docker service just runs
`examples/export_simple.py`); the package exposes several Python entry points:

| Entry point | Input | Output package format |
|---|---|---|
| `kuma.export_model(model, inputs, out, backend="torch"\|"onnx")` | `nn.Module` | `"kuma"` (unknown `backend` raises) |
| `kuma.export_exported_program(ep, out)` / `kuma.compile(ep)` | `ExportedProgram` | `"kuma"` |
| `kuma.compile_branching(...)` | N `ExportedProgram`s + JS router | `"kuma"` with a `switch` node |
| `kuma.export_via_onnx(model, inputs)` | `nn.Module` via ONNX | `"kuma"` |
| `kuma.compile_branching_onnx_kuma(...)` | N modules via ONNX + JS router | `"kuma"` with a `switch` node |
| `kuma.compile_onnx(...)` / `kuma.compile_branching_onnx(...)` | module(s) | `"onnx"` / `"onnx-branching"` (raw ONNX models, run by kuma-bart's onnxruntime-web `OnnxModel` path, not its WGSL interpreter) |

Every entry point that takes an `nn.Module` validates it the same way (tuple of cpu float32 tensors, float32
parameters, float32/int64/bool buffers). The ONNX paths need the `onnx` package, which
is a hard dependency in `pyproject.toml`.

Typical torch-path usage:

```python
import torch
import kuma

model = build_model(cfg)
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)

kuma.export_model(model, example_inputs, out="model.iph")
```

or, starting from an already-captured `ExportedProgram`:

```python
ep = torch.export.export(model.eval(), example_inputs)
kuma.export_exported_program(ep, out="model.iph")
```

or, for lower-level access to the in-memory package before writing it out:

```python
package = kuma.compile(ep)
package.save("model.iph")        # self-contained .iph zip
package.write_dir("debug/")      # loose files for inspection
```

**Wiring up an existing model repo:**

Write a thin `export.py` alongside your model code that calls the API directly:
```python
# mymodel/export.py
import torch
import kuma
from mymodel.build import build_model   # your existing factory

model = build_model(cfg)                # fill in your config however you do it
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)  # match the exact input shape/dtype your model expects

kuma.export_model(model, example_inputs, out="artifacts/mymodel.iph")
```

## Test suite layout

```
tests/
  conftest.py            # run_pipeline() fixture (kuma.compile or export_via_onnx -> Package) + shared assertion helpers
  test_package_iph.py    # .iph zip contract + top-level kuma.export_model/export_exported_program/compile API
  test_artifacts.py      # file existence, size consistency, JSON schema, roundtrip
  test_conv.py           # Conv2d variants, depthwise, 1×1, ConvTranspose2d
  test_linear.py         # Linear, MLP, channel-MLP via 1×1 conv
  test_norm.py           # BatchNorm2d, LayerNorm, GroupNorm, ConvNeXt-style norm
  test_activations.py    # GELU, ReLU, SiLU, Hardswish, Sigmoid, Tanh
  test_residual.py       # plain skip, projected skip, bottleneck block
  test_convnext_block.py # full ConvNeXt block (closest to real Niko/Nika target)
  test_error_cases.py    # float16/non-cpu inputs rejected loudly
  test_blocks.py         # SE block, encoder stage, tiny U-Net, multi-head output
  test_concat.py         # torch.cat / skip-connection fusion patterns
  test_depthwise_separable.py  # MobileNet-style depthwise-separable blocks
  test_elementwise.py    # mul, sub, div, clamp, abs
  test_multi_input.py    # multi-tensor model inputs
  test_pooling.py        # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
  test_shapes.py         # flatten, reshape, permute, transpose, squeeze/unsqueeze
  test_upsample.py       # nn.Upsample, ConvTranspose2d, PixelShuffle
  test_branching.py      # compile_branching (torch), compile_onnx, compile_branching_onnx,
                         # branch validation (lengths/indices/output shapes), torch golden capture
  test_onnx_backend.py   # same models through backend="torch" and backend="onnx"
  test_onnx_branching.py # compile_branching_onnx_kuma structure + golden correctness
  test_onnx_converters.py # per-converter regressions (MatMul/Gemm/Gather/Conv/ConvTranspose/
                         # Flatten/GELU), folding/name-collision/golden-determinism checks
```

Note that the torch-path tests only check the *compiler's* output contract: several ops
covered here (pooling, upsampling, BatchNorm, ConvTranspose2d) compile fine but have no
handler in kuma-bart's opRegistry yet, so such packages fail at load/run time there.

## Project layout

```
src/kuma/
  __init__.py       # public API (see "Entry points" below) + Package
  compiler.py       # compile(ep) -> Package; export_model/export_exported_program
  export.py         # validate_model_and_inputs + torch.export.export wrapper (export_program)
  graph.py          # FX graph → JSON-friendly dict (serialize_graph)
  pack_weights.py   # Parameters/buffers → contiguous f32 blob (pack_weights); the shared
                    # 4-byte alignment helpers (append_aligned/pack_f32_arrays)
  manifest.py       # manifest dict builder (build_manifest, build_playback_meta)
  golden.py         # golden.json capture: per-node value stats from an eager run
                    # (shared stats code for the torch and ONNX paths)
  debug.py          # debug report generator (generate_debug_report)
  package_iph.py    # Package dataclass: .save() writes the .iph zip, .write_dir() writes loose files
  branching.py      # compile_branching: N torch.export'd branches + JS router -> one package
  branch_assembly.py # shared router/switch graph + branch namespacing/weight concatenation
                    # used by all three branching entry points
  onnx_backend.py   # export_via_onnx / compile_branching_onnx_kuma: torch -> ONNX -> Kuma graph
  onnx_ops.py       # ONNX op -> aten target converters used by onnx_backend
  onnx_compiler.py  # compile_onnx / compile_branching_onnx: raw ONNX models in the .iph
                    # (format "onnx" / "onnx-branching"; kuma-bart runs these via onnxruntime-web, kuma-bart/src/onnx/)
  kernels/          # embedded WGSL kernels — elementwise binary/unary ops, matmul/conv
                     # (incl. transposed, plus depthwise/pointwise-specialized fast paths
                     # conv2d_depthwise.wgsl/conv2d_pointwise.wgsl alongside the general
                     # conv2d.wgsl), reductions, norm (layer/batch/group), pooling
                     # (max/avg/adaptive-avg), upsampling (nearest/bilinear), pixel-shuffle,
                     # and shape ops (reshape/permute/concat/slice) — bundled into every
                     # .iph package. See kernels/__init__.py for the full list. GELU has two:
                     # gelu.wgsl (approximate="tanh") and gelu_erf.wgsl (erf form, nn.GELU's
                     # default); the manifest node's `approximate` kwarg says which applies.
examples/
  simple.py         # Conv2d → GELU → Conv2d → residual add (acceptance-test model)
  export_simple.py  # Acceptance-test runner (Docker ENTRYPOINT for the `export` service)
artifacts/          # NOT git-ignored: committed demo models (beauty/bosphorus/bunny/honey/
                    # jockey/ready/shake/yacht.iph, their half*.iph variants) and the
                    # acceptance-test output simple.iph + simple/. `export` overwrites
                    # simple*; don't delete the committed .iph files.
```

### Cross-repo coupling: synthesized DFT weights

`pack_weights` deliberately skips any tensor whose name contains
`.complex_tucker.lifted_tensor_` (the fixed DFT/iDFT basis matrices torch.export lifts out
of the complex-Tucker spectral layers). They are not in `weights.f32.bin`; kuma-bart
regenerates them from name + shape in `kuma-bart/src/engine/synthetic_weights.ts`, which
matches the same `\.complex_tucker\.lifted_tensor_(\d+)$` pattern (indices 0/1 = complex
iFFT cos/sin, 2/3 = irfft bases). Changing the pattern or index meaning on either side
breaks the other silently.

## Key constraints (Step 1 scope)

- Inference only, static shapes, float32 only.
- `torch.export.export` is the primary capture path.
- WGSL kernels are embedded in every `.iph` package, and `kuma-bart` does execute them now
  — but there's still no kernel fusion or general memory/buffer planning anywhere in the
  pipeline (the `.iph` format itself doesn't carry what a real planner would need yet).
- Fail loudly on unsupported dtypes, non-CPU tensors, or dynamic shapes.
- Target models: Conv/ConvNeXt/operator/decoder style (Niko/Nika family).

## What's next (Step 2+)

`kuma-bart` (this repo's WebGPU interpreter) already exists and runs `.iph` packages —
see its own README for current status. What's still genuinely ahead, split across both
halves of the pipeline:

- Op decompositions and ATen → WebGPU op lowering beyond the current kernel set
- A real buffer/memory planner and a general kernel-fusion pass (today: a fixed-depth
  buffer-reuse pool and a few hand-specialized kernels in kuma-bart, not a planner) —
  needs the `.iph` format itself to start carrying information (live ranges, fusable
  subgraphs) it doesn't emit yet, so this is compiler-side work as much as runtime-side
