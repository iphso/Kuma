"""ONNX-backed .iph compiler: same Package/zip container as the Kuma path, but the
model is stored as ONNX rather than a Kuma graph of WGSL kernels.

The runtime distinguishes the two by manifest["format"]:
  "kuma"            — standard Kuma/WGSL graph (kuma.compiler.compile)
  "onnx"            — single ONNX model at models/model.onnx
  "onnx-branching"  — per-segment ONNX models + JS router (compile_branching_onnx)

Weights are embedded inside the ONNX model itself, so weights.f32.bin is always
empty for ONNX packages. The JS routing logic, playback metadata, and the zip
container structure are identical to the Kuma branching path.
"""

from __future__ import annotations

import io
from typing import Any

import torch
import torch.nn as nn

from kuma.manifest import build_playback_meta
from kuma.package_iph import Package


def _to_onnx_bytes(
    wrapper: nn.Module,
    example_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    opset_version: int = 17,
) -> bytes:
    buf = io.BytesIO()
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        example_inputs,
        buf,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
        opset_version=opset_version,
    )
    return buf.getvalue()


def compile_onnx(
    wrapper: nn.Module,
    example_inputs: tuple,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    fps: float | None = None,
    duration_seconds: float | None = None,
    opset_version: int = 17,
) -> Package:
    """Export `wrapper` to ONNX and package it as a single-model .iph.

    input_names/output_names default to ["input"] / ["output"] when not given;
    callers that care about the names a downstream runtime will see should pass them.
    """
    in_names = input_names or ["input"]
    out_names = output_names or ["output"]

    onnx_bytes = _to_onnx_bytes(wrapper, example_inputs, in_names, out_names, opset_version)

    manifest: dict[str, Any] = {
        "format": "onnx",
        "format_version": 0,
        "model_file": "models/model.onnx",
        "inputs": [{"name": n} for n in in_names],
        "outputs": [{"name": n} for n in out_names],
    }
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    return Package(
        manifest=manifest,
        weights_blob=b"",
        graph_data={},
        debug_report=(
            f"# Kuma ONNX Export\n\n"
            f"- Model size: {len(onnx_bytes):,} bytes\n"
            f"- Inputs: {in_names}\n"
            f"- Outputs: {out_names}\n"
        ),
        models={"model.onnx": onnx_bytes},
    )


def compile_branching_onnx(
    router_snippet_name: str,
    router_snippet_source: str,
    router_input_names: list[str],
    router_output_specs: list[dict[str, Any]],
    selector_output_index: int,
    branch_input_output_index: int,
    branch_wrappers: list[nn.Module],
    branch_example_inputs: list[tuple],
    *,
    branch_input_names: list[str] | None = None,
    branch_output_names: list[str] | None = None,
    fps: float | None = None,
    duration_seconds: float | None = None,
    opset_version: int = 17,
) -> Package:
    """Export each branch wrapper to ONNX and assemble a branching .iph.

    The router JS snippet and switch-node graph structure mirror compile_branching
    exactly. Each branch's "nodes" list is replaced by a {"model_file": "models/..."}
    reference so a runtime dispatches to the correct ONNX model for the selected branch.
    """
    if not branch_wrappers:
        raise ValueError("compile_branching_onnx requires at least one branch")

    in_names = branch_input_names or ["input"]
    out_names = branch_output_names or ["output"]

    models: dict[str, bytes] = {}
    branch_specs: list[dict[str, Any]] = []
    total_onnx_bytes = 0
    for i, (wrapper, example_inp) in enumerate(zip(branch_wrappers, branch_example_inputs)):
        onnx_bytes = _to_onnx_bytes(wrapper, example_inp, in_names, out_names, opset_version)
        filename = f"segment_{i}.onnx"
        models[filename] = onnx_bytes
        total_onnx_bytes += len(onnx_bytes)
        branch_specs.append({"model_file": f"models/{filename}"})

    # Router node + getitem unpacking (identical structure to compile_branching)
    router_node_name = "router"
    router_node: dict[str, Any] = {
        "id": 0,
        "name": router_node_name,
        "op": "js_snippet",
        "target": router_snippet_name,
        "args": [{"node_ref": name} for name in router_input_names],
        "kwargs": {},
        "meta": {
            "outputs": [
                {"shape": spec["shape"], "dtype": spec.get("dtype", "float32")}
                for spec in router_output_specs
            ]
        },
    }

    getitem_names: list[str] = []
    getitem_nodes: list[dict[str, Any]] = []
    for i, spec in enumerate(router_output_specs):
        name = f"router_out_{i}"
        getitem_names.append(name)
        getitem_nodes.append({
            "id": i + 1,
            "name": name,
            "op": "call_function",
            "target": "getitem",
            "args": [{"node_ref": router_node_name}, i],
            "kwargs": {},
            "meta": {},
        })

    # Switch node: branches reference ONNX files instead of inlined node lists
    switch_node: dict[str, Any] = {
        "id": 1_000_000,
        "name": "switch_0",
        "op": "switch",
        "target": "switch",
        "args": [{"node_ref": getitem_names[branch_input_output_index]}],
        "kwargs": {},
        "meta": {},
        "selector": {"node_ref": getitem_names[selector_output_index]},
        "branches": branch_specs,
    }
    output_node: dict[str, Any] = {
        "id": 1_000_001,
        "name": "output",
        "op": "output",
        "target": "output",
        "args": [[{"node_ref": "switch_0"}]],
        "kwargs": {},
        "meta": {},
    }

    all_nodes = [router_node, *getitem_nodes, switch_node, output_node]

    manifest: dict[str, Any] = {
        "format": "onnx-branching",
        "format_version": 0,
        "inputs": [{"name": name} for name in router_input_names],
        "outputs": [{"name": "output"}],
        "graph": {
            "node_count": len(all_nodes),
            "nodes": all_nodes,
        },
    }
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    debug_report = (
        f"# Kuma ONNX Branching Export\n\n"
        f"- Branches: {len(branch_wrappers)}\n"
        f"- Router snippet: {router_snippet_name}\n"
        f"- Total ONNX model bytes: {total_onnx_bytes:,}\n"
    )

    return Package(
        manifest=manifest,
        weights_blob=b"",
        graph_data={"format_version": 0, "nodes": all_nodes},
        debug_report=debug_report,
        snippets={router_snippet_name: router_snippet_source.encode()},
        models=models,
    )
