"""Int8 weight quantization for .iph packages.

A quantized package stores every large float32 tensor as int8 with one float32 scale per
output channel (dim 0), and everything else unchanged:

    weights.f32.bin   float32 tensors that stay float32 (small ones: biases, gains,
                      embeddings) plus every quantized tensor's scales
    weights.q8.bin    the int8 values, row-major like the float32 tensor they replace;
                      each tensor starts on a 4-byte boundary and is zero-padded to a
                      multiple of 4 bytes (so a runtime can bind it as array<u32>, four
                      int8 per word, little-endian)

A quantized tensor's manifest entry:

    {"name", "shape", "n_elements",
     "dtype": "int8", "file": "weights.q8.bin",
     "byte_offset", "byte_length" (= n_elements, unpadded),
     "quant": {"scheme": "int8_symmetric", "group_size": G,
               "scale_dtype": "float32", "scale_offset", "scale_count"}}

Element i dequantizes to float32(q[i]) * scale[i // G] (one f32 multiply; scale_offset is
a byte offset into the manifest's `weight_file`). For a tensor of rank >= 2, G is the
product of its trailing dims (one scale per output channel); a rank-1 tensor gets one
scale (G = n_elements). That is exactly the rounding of Loom's py/quant/fakequant.py int8
mode (symmetric, qmax 127, round half to even, all in float32), so a converted package's
dequantized weights are bit-identical to a fake-quantized float32 package's.

Quantized packages have `format_version` 1 and a top-level `quant_weight_file`; float32
packages are unchanged (format_version 0). The kernels a runtime needs to consume int8
weights directly (kernels/*_q8.wgsl) are added to the package.

CLI (converts an existing package; no torch model needed):

    python -m kuma.quantize models/base_model.iph out/base_model.q8.iph
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kuma.kernels import load_quant_kernels
from kuma.pack_weights import append_aligned
from kuma.package_iph import Package

QUANT_WEIGHT_FILE = "weights.q8.bin"
QUANT_FORMAT_VERSION = 1
SCHEME = "int8_symmetric"
QMAX = 127
# Tensors under this many elements (biases, norm gains, small embeddings) stay float32:
# they are a rounding error of the package size and the most precision-sensitive.
DEFAULT_MIN_ELEMENTS = 4096


def quantize_int8(values: np.ndarray, shape: Iterable[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Symmetric int8 with one scale per dim-0 slice (one for a rank-1 tensor).

    Returns (q: int8[n], scales: float32[rows], group_size). Every step is float32 so the
    result matches fakequant.py bit for bit: scale = max|w| / 127 (1 if the row is all
    zeros), q = clip(round_half_even(w / scale), -127, 127)."""
    shape = list(shape)
    flat = np.asarray(values, dtype="<f4").reshape(-1)
    rows = int(shape[0]) if len(shape) > 1 else 1
    n = int(flat.size)
    if rows <= 0 or n % rows != 0:
        raise ValueError(f"cannot split {n} elements of shape {shape} into {rows} rows")
    r = flat.reshape(rows, n // rows)
    scale = np.abs(r).max(axis=1, keepdims=True) / np.float32(QMAX)
    scale[scale == 0] = np.float32(1)
    q = np.clip(np.round(r / scale), -QMAX, QMAX)
    return q.astype(np.int8).reshape(-1), scale.astype("<f4").reshape(-1), n // rows


def dequantize_int8(q: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """float32(q[i]) * scales[i // group_size] -- what a runtime computes."""
    return (q.astype(np.float32).reshape(-1, group_size) * scales.astype(np.float32)[:, None]).reshape(-1)


def _should_quantize(entry: dict[str, Any], min_elements: int, pattern: re.Pattern[str] | None) -> bool:
    if entry.get("dtype") != "float32" or entry.get("n_elements", 0) < min_elements:
        return False
    return pattern is None or pattern.search(entry["name"]) is not None


def quantize_manifest_and_blob(
    manifest: dict[str, Any],
    weights_blob: bytes,
    *,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
    include: str | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    """The core conversion: (manifest, f32 blob) -> (new manifest, new f32 blob, q8 blob).
    `include` is an optional regex on tensor names (default: every float32 tensor of at
    least `min_elements` elements, the scheme measured in Loom EXPERIMENT_LOG §155)."""
    if manifest.get("format") != "kuma":
        raise ValueError(f"only 'kuma' packages can be quantized, got format {manifest.get('format')!r}")
    if manifest.get("quant_weight_file") or any(w.get("dtype") != "float32" for w in manifest["weights"]):
        raise ValueError("package is already quantized (has non-float32 weight entries)")
    pattern = re.compile(include) if include else None

    f32 = bytearray()
    q8 = bytearray()
    entries: list[dict[str, Any]] = []
    n_quantized = n_kept = 0
    for entry in manifest["weights"]:
        raw = weights_blob[entry["byte_offset"] : entry["byte_offset"] + entry["byte_length"]]
        if not _should_quantize(entry, min_elements, pattern):
            new = dict(entry, byte_offset=append_aligned(f32, raw))
            entries.append(new)
            n_kept += entry["n_elements"]
            continue
        values = np.frombuffer(raw, dtype="<f4")
        q, scales, group = quantize_int8(values, entry["shape"])
        q_raw = q.tobytes()
        q_offset = append_aligned(q8, q_raw + b"\x00" * ((-len(q_raw)) % 4))
        s_raw = scales.tobytes()
        s_offset = append_aligned(f32, s_raw)
        entries.append(
            {
                "name": entry["name"],
                "shape": list(entry["shape"]),
                "n_elements": entry["n_elements"],
                "dtype": "int8",
                "file": QUANT_WEIGHT_FILE,
                "byte_offset": q_offset,
                "byte_length": len(q_raw),
                "quant": {
                    "scheme": SCHEME,
                    "group_size": group,
                    "scale_dtype": "float32",
                    "scale_offset": s_offset,
                    "scale_count": int(scales.size),
                },
            }
        )
        n_quantized += entry["n_elements"]

    out = dict(manifest)
    out["format_version"] = QUANT_FORMAT_VERSION
    out["quant_weight_file"] = QUANT_WEIGHT_FILE
    out["weights"] = entries
    out["quantization"] = {
        "scheme": SCHEME,
        "granularity": "per_output_channel",
        "min_elements": min_elements,
        "include": include,
        "quantized_elements": n_quantized,
        "float32_elements": n_kept,
    }
    return out, bytes(f32), bytes(q8)


_GOLDEN_DROPPED = (
    "golden.json dropped by int8 quantization: it records the float32 model's values, "
    "which an int8 package no longer matches within verify tolerances"
)


def quantize_package(
    package: Package,
    *,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
    include: str | None = None,
    keep_golden: bool = False,
) -> Package:
    """A copy of `package` with int8 weights (see the module docstring) and the q8 kernels
    added. golden.json is dropped unless `keep_golden` (it describes the float32 model)."""
    manifest, f32, q8 = quantize_manifest_and_blob(
        package.manifest, package.weights_blob, min_elements=min_elements, include=include
    )
    warnings = list(manifest.get("warnings", []))
    golden = package.golden
    if golden is not None and not keep_golden:
        golden = None
        warnings.append(_GOLDEN_DROPPED)
    manifest["warnings"] = warnings
    info = manifest["quantization"]
    report = package.debug_report + (
        "\n\n## Int8 weights\n\n"
        f"Scheme `{SCHEME}`, one float32 scale per output channel (dim 0), tensors with >= "
        f"{min_elements} elements: {info['quantized_elements']:,} weights in `{QUANT_WEIGHT_FILE}` "
        f"({len(q8):,} bytes), {info['float32_elements']:,} kept float32.\n"
    )
    return Package(
        manifest=manifest,
        weights_blob=f32,
        graph_data=package.graph_data,
        debug_report=report,
        kernels={**package.kernels, **load_quant_kernels()},
        snippets=dict(package.snippets),
        skipped=list(package.skipped),
        golden=golden,
        models=dict(package.models),
        quant_weights_blob=q8,
    )


def read_package(path: str | Path) -> Package:
    """Load a saved .iph back into an in-memory Package (graph_data is not stored in a
    .iph; it comes back empty). Fails on zip entries Package would not write back."""
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        kernels: dict[str, bytes] = {}
        snippets: dict[str, bytes] = {}
        models: dict[str, bytes] = {}
        golden = None
        report = ""
        weights = b""
        quant = None
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if name == "manifest.json":
                continue
            if name == manifest.get("weight_file", "weights.f32.bin"):
                weights = zf.read(name)
            elif name == manifest.get("quant_weight_file"):
                quant = zf.read(name)
            elif name.startswith("kernels/"):
                kernels[name[len("kernels/") :]] = zf.read(name)
            elif name.startswith("snippets/"):
                snippets[name[len("snippets/") :]] = zf.read(name)
            elif name.startswith("models/"):
                models[name[len("models/") :]] = zf.read(name)
            elif name == "golden.json":
                golden = json.loads(zf.read(name))
            elif name == "debug_report.md":
                report = zf.read(name).decode()
            else:
                raise ValueError(f"{path}: unexpected entry {name!r} (would be lost on rewrite)")
    return Package(
        manifest=manifest,
        weights_blob=weights,
        graph_data={},
        debug_report=report,
        kernels=kernels,
        snippets=snippets,
        golden=golden,
        models=models,
        quant_weights_blob=quant,
    )


def quantize_iph(
    src: str | Path,
    dst: str | Path,
    *,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
    include: str | None = None,
    keep_golden: bool = False,
) -> Path:
    """Convert a saved float32 .iph to an int8 one, without the torch model."""
    package = quantize_package(read_package(src), min_elements=min_elements, include=include, keep_golden=keep_golden)
    return package.save(dst)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m kuma.quantize", description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="float32 .iph package")
    ap.add_argument("dst", help="int8 .iph package to write")
    ap.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS,
                    help=f"quantize float32 tensors with at least this many elements (default {DEFAULT_MIN_ELEMENTS})")
    ap.add_argument("--include", default=None, help="only quantize tensors whose name matches this regex")
    ap.add_argument("--keep-golden", action="store_true", help="keep golden.json (it describes the float32 model)")
    args = ap.parse_args(argv)
    out = quantize_iph(args.src, args.dst, min_elements=args.min_elements, include=args.include, keep_golden=args.keep_golden)
    m = json.loads(zipfile.ZipFile(out).read("manifest.json"))
    info = m["quantization"]
    print(
        f"{out}: {Path(args.src).stat().st_size / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB, "
        f"{info['quantized_elements']:,} weights int8, {info['float32_elements']:,} kept float32"
    )


if __name__ == "__main__":
    main()
