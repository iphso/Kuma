"""Shared fixtures for the Kuma test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kuma.compiler import compile as kuma_compile
from kuma.package_iph import Package


def run_pipeline(
    model: nn.Module,
    example_inputs: tuple,
    out_dir: Path,
    *,
    backend: str = "torch",
) -> dict[str, Any]:
    """
    Run the full compile pipeline against a model and return a dict of all outputs.
    This is the canonical test harness — tests should call this, not reimport internals.

    backend="torch"  — direct torch.export path (default)
    backend="onnx"   — export via ONNX as an intermediary

    Writes both the loose debug artifacts (for assert_artifacts_exist) and a model.iph
    package into out_dir.
    """
    model.eval()

    if backend == "onnx":
        from kuma.onnx_backend import export_via_onnx
        package = export_via_onnx(model, example_inputs)
        package.write_dir(out_dir)
        iph_path = package.save(out_dir / "model.iph")
        return {
            "ep": None,
            "package": package,
            "graph_data": package.graph_data,
            "blob": package.weights_blob,
            "weight_entries": package.manifest["weights"],
            "skipped": package.skipped,
            "manifest": package.manifest,
            "out_dir": out_dir,
            "iph_path": iph_path,
        }

    with torch.no_grad():
        ep = torch.export.export(model, example_inputs)

    package = kuma_compile(ep)
    package.write_dir(out_dir)
    iph_path = package.save(out_dir / "model.iph")

    return {
        "ep": ep,
        "package": package,
        "graph_data": package.graph_data,
        "blob": package.weights_blob,
        "weight_entries": package.manifest["weights"],
        "skipped": package.skipped,
        "manifest": package.manifest,
        "out_dir": out_dir,
        "iph_path": iph_path,
    }


# ── Common assertion helpers ─────────────────────────────────────────────────

def assert_artifacts_exist(out_dir: Path) -> None:
    for fname in ["exported_graph.json", "weights.f32.bin", "manifest.json", "debug_report.md"]:
        assert (out_dir / fname).exists(), f"Missing artifact: {fname}"


def assert_weight_sizes_match(weight_entries: list[dict], blob: bytes) -> None:
    """weights.f32.bin size and every byte_offset+byte_length must be consistent."""
    for w in weight_entries:
        end = w["byte_offset"] + w["byte_length"]
        assert end <= len(blob), (
            f"Weight '{w['name']}' claims offset {w['byte_offset']} + "
            f"length {w['byte_length']} = {end} but blob is only {len(blob)} bytes"
        )
        expected_bytes = w["n_elements"] * 4  # float32
        assert w["byte_length"] == expected_bytes, (
            f"Weight '{w['name']}': byte_length {w['byte_length']} != "
            f"n_elements {w['n_elements']} * 4 = {expected_bytes}"
        )


def assert_manifest_schema(manifest: dict) -> None:
    for key in ("format", "format_version", "inputs", "outputs", "weights", "graph", "warnings"):
        assert key in manifest, f"manifest.json missing key: {key!r}"
    assert manifest["format"] == "kuma"
    assert manifest["format_version"] == 0
    assert isinstance(manifest["weights"], list)
    assert "nodes" in manifest["graph"]
    assert "op_counts" in manifest["graph"]


def assert_ops_present(manifest: dict, *expected_ops: str) -> None:
    found = set(manifest["graph"]["op_counts"].keys())
    for op in expected_ops:
        assert op in found, f"Expected op '{op}' not found in graph. Got: {sorted(found)}"


def roundtrip_weight(weight_entries: list[dict], blob: bytes, name: str) -> torch.Tensor:
    """Read a named weight back out of the blob and return it as a float32 tensor."""
    entry = next(w for w in weight_entries if w["name"] == name)
    raw = blob[entry["byte_offset"] : entry["byte_offset"] + entry["byte_length"]]
    import numpy as np
    arr = np.frombuffer(raw, dtype="<f4").copy()
    return torch.from_numpy(arr).reshape(entry["shape"])
