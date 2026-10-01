"""ONNX backend: torch.nn.Module → ONNX → kuma Package.

An alternative compilation path where the model is first exported to ONNX (via
torch.onnx.export), then converted to kuma's native manifest+WGSL format. The ONNX
op -> aten-target converters live in kuma.onnx_ops; this module handles everything
around them (ONNX capture, static-value folding, graph walking/pruning, weight
packing, manifest/golden assembly).

Entry points:
  export_via_onnx(model, example_inputs, ...) → Package
  compile_branching_onnx_kuma(...)            → Package (router + switch, like
                                                kuma.branching.compile_branching)
"""

from __future__ import annotations

import io
import warnings as py_warnings
from typing import Any

import numpy as np
import onnx
import onnx.shape_inference
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from kuma.branch_assembly import (
    KumaBranch,
    assemble_kuma_branching_package,
    prune_unused_placeholders,
    collect_node_refs,
    validate_branching_args,
)
from kuma.export import validate_model_and_inputs
from kuma.golden import namespace_golden, value_stats
from kuma.kernels import load_kernels
from kuma.manifest import build_playback_meta
from kuma.onnx_ops import (
    CONVERTERS,
    attr,
    init_list,
    matmul_rhs_is_2d_initializer,
    matmul_transposed_weight_name,
    ref,
    safe_name,
)
from kuma.pack_weights import pack_f32_arrays
from kuma.package_iph import Package


# ── ONNX type helpers ─────────────────────────────────────────────────────────

_ONNX_CAST_DTYPES: dict[int, Any] = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
}

_ONNX_ELEM_TYPE: dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
}


def _get_shape(type_proto) -> list[int] | None:
    """Static shape of a value_info, with unknown/symbolic dims encoded as -1."""
    t = type_proto.tensor_type
    if not t.HasField("shape"):
        return None
    return [int(d.dim_value) if d.HasField("dim_value") else -1 for d in t.shape.dim]


def _get_dtype(type_proto) -> str:
    return _ONNX_ELEM_TYPE.get(type_proto.tensor_type.elem_type, "unknown")


def _constant_array_from_node(node) -> np.ndarray | None:
    for a in node.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
        if a.name == "value_int":
            return np.asarray(a.i, dtype=np.int64)
        if a.name == "value_ints":
            return np.asarray(list(a.ints), dtype=np.int64)
        if a.name == "value_float":
            return np.asarray(a.f, dtype=np.float32)
        if a.name == "value_floats":
            return np.asarray(list(a.floats), dtype=np.float32)
    return None


def _shape_array_for(name: str, shape_map: dict[str, dict]) -> np.ndarray | None:
    shape = shape_map.get(name, {}).get("shape")
    # Unknown dims are encoded as -1 by _get_shape -- not a foldable static shape.
    if shape is None or any(d is None or d < 0 for d in shape):
        return None
    return np.asarray(shape, dtype=np.int64)


# ── Step 1: torch → ONNX ──────────────────────────────────────────────────────

def _torch_to_onnx(model: nn.Module, example_inputs: tuple, opset_version: int) -> onnx.ModelProto:
    buf = io.BytesIO()
    input_names = [f"input_{i}" for i in range(len(example_inputs))]
    with torch.no_grad(), py_warnings.catch_warnings():
        py_warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        py_warnings.filterwarnings("ignore", message="To copy construct from a tensor.*", category=UserWarning)
        torch.onnx.export(
            model,
            example_inputs,
            buf,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
        )
    # Shape inference so meta.shape can be populated everywhere.
    return onnx.shape_inference.infer_shapes(onnx.load_model_from_string(buf.getvalue()))


# ── Step 2: static values + shapes ────────────────────────────────────────────

def _collect_static_values(graph) -> dict[str, np.ndarray]:
    """Initializers plus Constant-node outputs, by tensor name."""
    init_values = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            value = _constant_array_from_node(node)
            if value is not None:
                init_values[node.output[0]] = value
    return init_values


def _build_shape_map(graph) -> dict[str, dict]:
    shape_map: dict[str, dict] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        s = _get_shape(vi.type)
        if s is not None:
            shape_map[vi.name] = {"shape": s, "dtype": _get_dtype(vi.type)}
    return shape_map


def _fold_one(node, init_values: dict[str, np.ndarray], shape_map: dict[str, dict]) -> np.ndarray | None:
    inp = list(node.input)
    op = node.op_type

    if op == "Shape" and inp:
        shape = _shape_array_for(inp[0], shape_map)
        if shape is not None:
            start = int(attr(node, "start", 0))
            end_attr = attr(node, "end", None)
            end = len(shape) if end_attr is None else int(end_attr)
            return shape[start:end]

    elif op == "Cast" and inp and inp[0] in init_values:
        dtype = _ONNX_CAST_DTYPES.get(int(attr(node, "to", 0)))
        if dtype is not None:
            return init_values[inp[0]].astype(dtype, copy=False)

    elif op == "Unsqueeze" and inp and inp[0] in init_values:
        axes = init_list(init_values, inp[1] if len(inp) > 1 else None)
        if axes is None:
            axes = attr(node, "axes", None)
        if axes is not None:
            value = init_values[inp[0]]
            rank = value.ndim + len(axes)
            for axis in sorted((a + rank if a < 0 else a) for a in axes):
                value = np.expand_dims(value, axis)
            return value

    elif op == "Squeeze" and inp and inp[0] in init_values:
        axes = init_list(init_values, inp[1] if len(inp) > 1 else None)
        if axes is None:
            axes = attr(node, "axes", None)
        return np.squeeze(init_values[inp[0]], axis=None if axes is None else tuple(int(a) for a in axes))

    elif op == "Reshape" and len(inp) >= 2 and inp[0] in init_values and inp[1] in init_values:
        return np.reshape(init_values[inp[0]], [int(v) for v in init_values[inp[1]].reshape(-1)])

    elif op == "Concat" and inp and all(i in init_values for i in inp):
        return np.concatenate([init_values[i] for i in inp], axis=int(attr(node, "axis", 0)))

    elif op == "Gather" and len(inp) >= 2 and inp[0] in init_values and inp[1] in init_values:
        axis = int(attr(node, "axis", 0))
        return np.take(init_values[inp[0]], init_values[inp[1]].astype(np.int64), axis=axis)

    elif op in ("Add", "Sub", "Mul", "Div") and len(inp) >= 2 and inp[0] in init_values and inp[1] in init_values:
        fn = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}[op]
        return fn(init_values[inp[0]], init_values[inp[1]])

    elif op == "Floor" and inp and inp[0] in init_values:
        return np.floor(init_values[inp[0]])

    elif op == "Slice" and len(inp) >= 3 and inp[0] in init_values:
        starts = init_list(init_values, inp[1])
        ends = init_list(init_values, inp[2])
        axes = init_list(init_values, inp[3] if len(inp) > 3 else None)
        steps = init_list(init_values, inp[4] if len(inp) > 4 else None)
        if starts is not None and ends is not None:
            arr = init_values[inp[0]]
            axes = axes if axes is not None else list(range(len(starts)))
            steps = steps if steps is not None else [1] * len(starts)
            if len(starts) == len(ends) == len(axes) == len(steps):
                slices = [slice(None)] * arr.ndim
                for axis, start, end, step in zip(axes, starts, ends, steps):
                    slices[int(axis)] = slice(int(start), int(end), int(step))
                return arr[tuple(slices)]

    elif op == "ConstantOfShape" and inp and inp[0] in init_values:
        shape = tuple(int(v) for v in init_values[inp[0]].reshape(-1))
        fill = attr(node, "value", None)
        scalar = np.asarray(fill).reshape(-1)[0] if fill is not None else np.float32(0)
        if int(np.prod(shape, dtype=np.int64)) <= 1024:
            return np.full(shape, scalar)

    return None


def _fold_static_onnx_values(graph, init_values: dict[str, np.ndarray], shape_map: dict[str, dict]) -> None:
    """Evaluate small constant ONNX shape/index subgraphs into init_values.

    ONNX commonly represents static slice bounds and reshape sizes as little graphs
    made from Shape/Gather/Slice/Cast/Unsqueeze/Concat. Kuma converters want those as
    Python attrs; leaving them as executable graph nodes creates reachable fallbacks.

    A node whose static evaluation hits a numpy shape/index/type error (ValueError,
    IndexError, TypeError -- e.g. an out-of-range Gather index or incompatible Reshape)
    is simply left unfolded: it stays a real graph op, and if it's reachable and has no
    converter, export_via_onnx fails loudly on it later. Any other exception is a bug
    and propagates.
    """
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if not node.output or node.output[0] in init_values:
                continue
            try:
                value = _fold_one(node, init_values, shape_map)
            except (ValueError, IndexError, TypeError):
                value = None
            if value is not None:
                init_values[node.output[0]] = np.asarray(value)
                changed = True


def _add_matmul_transposed_weights(graph, init_values: dict[str, np.ndarray], shape_map: dict[str, dict]) -> None:
    """Synthesize the [N, K] transpose of every 2D MatMul rhs initializer (see
    onnx_ops._matmul for why kuma-bart needs that layout)."""
    for node in graph.node:
        if not matmul_rhs_is_2d_initializer(node, init_values):
            continue
        transposed_name = matmul_transposed_weight_name(node.input[1])
        if transposed_name not in init_values:
            init_values[transposed_name] = np.ascontiguousarray(init_values[node.input[1]].T)
            shape_map[transposed_name] = {"shape": list(init_values[transposed_name].shape), "dtype": "float32"}


def _check_safe_name_collisions(graph, init_values: dict[str, np.ndarray]) -> None:
    """safe_name() isn't injective; two distinct ONNX tensor names mapping to the same
    manifest node name would silently alias two different tensors. Refuse instead."""
    names: set[str] = set(init_values)
    names.update(vi.name for vi in graph.input)
    names.update(vi.name for vi in graph.output)
    for node in graph.node:
        names.update(node.output)
    names.discard("")

    by_safe: dict[str, str] = {}
    collisions: list[str] = []
    for name in sorted(names):
        s = safe_name(name)
        other = by_safe.setdefault(s, name)
        if other != name:
            collisions.append(f"{other!r} and {name!r} -> {s!r}")
    if collisions:
        raise ValueError(
            "ONNX backend: distinct ONNX tensor names collide after sanitisation to manifest node names: "
            + "; ".join(collisions[:8])
            + (f" ... ({len(collisions) - 8} more)" if len(collisions) > 8 else "")
        )


# ── Step 3: graph conversion ──────────────────────────────────────────────────

def _node_meta(name: str, shape_map: dict[str, dict]) -> dict[str, Any]:
    info = shape_map.get(name, {})
    if info.get("shape") is None:
        return {}
    return {"shape": info["shape"], "dtype": info.get("dtype", "float32")}


def _build_graph(
    graph,
    input_specs: list,
    init_values: dict[str, np.ndarray],
    shape_map: dict[str, dict],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Walk an ONNX GraphProto and emit kuma manifest nodes.

    Returns (nodes, warnings, unsupported ONNX op types)."""
    nodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    unsupported: list[str] = []
    init_names = set(init_values)

    # Initializers actually referenced by ops (MatMul rhs -> its transposed copy)
    used_inits: set[str] = set()
    for n in graph.node:
        for idx, inp in enumerate(n.input):
            if idx == 1 and matmul_rhs_is_2d_initializer(n, init_values):
                used_inits.add(matmul_transposed_weight_name(inp))
            elif inp in init_names:
                used_inits.add(inp)

    # User-input placeholders
    for spec in input_specs:
        nodes.append({
            "id": len(nodes),
            "name": safe_name(spec.name),
            "op": "placeholder",
            "target": safe_name(spec.name),
            "args": [], "kwargs": {}, "meta": _node_meta(spec.name, shape_map),
            "kind": "user_input",
        })

    # Weight placeholders (sorted for deterministic ordering)
    for name in sorted(used_inits):
        nodes.append({
            "id": len(nodes),
            "name": safe_name(name),
            "op": "placeholder",
            "target": safe_name(name),
            "args": [], "kwargs": {},
            "meta": {"shape": list(init_values[name].shape), "dtype": "float32"},
            "kind": "parameter",
            "weight_name": name,
        })

    # Op nodes
    for onnx_node in graph.node:
        op_type = onnx_node.op_type
        # Constants and anything _fold_static_onnx_values evaluated are already
        # represented by init_values (a weight placeholder of the same safe name, if
        # referenced at all) -- emitting the op too would duplicate that node name.
        if onnx_node.output and onnx_node.output[0] in init_values:
            continue
        out_name = onnx_node.output[0] if onnx_node.output else f"_node_{len(nodes)}"
        meta = _node_meta(out_name, shape_map)

        converter = CONVERTERS.get(op_type)
        result = converter(onnx_node, init_values, shape_map) if converter is not None else None
        if result is None:
            if op_type not in unsupported:
                unsupported.append(op_type)
            if converter is None:
                warnings.append(f"Unsupported ONNX op: {op_type} (→ {out_name})")
            else:
                known_inputs = [inp for inp in onnx_node.input if inp in init_values]
                warnings.append(
                    f"Cannot statically convert {op_type} (→ {out_name}); "
                    f"inputs={list(onnx_node.input)} static_inputs={known_inputs}"
                )
            _emit_fallback(nodes, out_name, onnx_node, meta)
            continue

        nodes.append({
            "id": len(nodes),
            "name": safe_name(out_name),
            "op": "call_function",
            "target": result["target"],
            "args": result["args"],
            "kwargs": result.get("kwargs", {}),
            "meta": meta,
        })

    nodes.append({
        "id": len(nodes), "name": "__output__", "op": "output",
        "target": "output", "args": [[ref(out.name) for out in graph.output]], "kwargs": {}, "meta": {},
    })
    return nodes, warnings, unsupported


def _emit_fallback(nodes, out_name, onnx_node, meta):
    """Emit an alias passthrough for an unconvertible op so downstream refs still
    resolve. Marked with meta.onnx_fallback_op; _check_no_reachable_fallbacks refuses
    to ship a package where any such node is reachable from the output."""
    if onnx_node.input:
        fallback_meta = dict(meta)
        fallback_meta["onnx_fallback_op"] = onnx_node.op_type
        fallback_meta["onnx_inputs"] = list(onnx_node.input)
        nodes.append({
            "id": len(nodes),
            "name": safe_name(out_name),
            "op": "call_function",
            "target": "aten.alias.default",
            "args": [ref(onnx_node.input[0])],
            "kwargs": {}, "meta": fallback_meta,
        })
    # If no inputs (e.g. a Constant op), skip — a reachable reference to it fails loudly.


def _prune_unreachable_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only nodes needed to produce the manifest output.

    ONNX exports often include shape-construction side graphs. Several converters
    consume those shapes as Python metadata, leaving the original shape nodes dead.
    Dropping unreachable nodes before weight packing prevents dead int64 constants
    from becoming required Kuma weights.
    """
    by_name = {node["name"]: node for node in nodes}
    reachable: set[str] = set()
    stack = [node["name"] for node in nodes if node["op"] == "output"]
    while stack:
        name = stack.pop()
        if name in reachable or name not in by_name:
            continue
        reachable.add(name)
        refs: set[str] = set()
        collect_node_refs(by_name[name].get("args", []), refs)
        collect_node_refs(by_name[name].get("kwargs", {}), refs)
        stack.extend(refs)

    return [
        node for node in nodes
        if node["name"] in reachable or (node["op"] == "placeholder" and node.get("kind") == "user_input")
    ]


def _check_no_reachable_fallbacks(graph_nodes: list[dict[str, Any]], warnings: list[str]) -> None:
    if not any(n["op"] == "call_function" for n in graph_nodes):
        raise ValueError("ONNX backend produced an empty Kuma graph after pruning; graph output was not connected")
    fallbacks = [n for n in graph_nodes if n.get("meta", {}).get("onnx_fallback_op")]
    if not fallbacks:
        return
    ops = sorted({n["meta"]["onnx_fallback_op"] for n in fallbacks})
    details = [
        f"{n['name']}:{n['meta']['onnx_fallback_op']}"
        + (f" inputs={n['meta']['onnx_inputs']}" if n["meta"].get("onnx_inputs") else "")
        for n in fallbacks
    ]
    raise ValueError(
        "ONNX backend cannot convert reachable op(s): "
        + ", ".join(ops)
        + ". Refusing to write a passthrough-corrupted Kuma package. "
        + "Reachable fallback nodes: "
        + ", ".join(details[:8])
        + (f" ... ({len(details) - 8} more)" if len(details) > 8 else "")
        + ". Conversion warnings: "
        + " | ".join(warnings[-8:])
    )


# ── Step 4: weight packing ────────────────────────────────────────────────────

def _pack_used_weights(
    graph_nodes: list[dict[str, Any]], init_values: dict[str, np.ndarray]
) -> tuple[bytes, list[dict[str, Any]], list[str]]:
    """Pack every initializer a surviving placeholder references (sorted by name) into
    the standard f32 blob. Integer/bool initializers that survive pruning are
    data/index tensors, not shape attrs; runtime buffers are f32 and the gather kernels
    intentionally read them as f32-encoded integers. Anything else is unpackable and
    raises."""
    used = sorted({
        n["weight_name"] for n in graph_nodes
        if n.get("kind") in ("parameter", "buffer") and "weight_name" in n
    })
    to_pack: list[tuple[str, np.ndarray]] = []
    skipped: list[str] = []
    for name in used:
        arr = init_values[name]
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
            to_pack.append((name, arr))
        else:
            skipped.append(f"{name} ({arr.dtype})")
    if skipped:
        raise ValueError(
            "ONNX backend cannot pack referenced non-numeric initializer(s): "
            + ", ".join(skipped[:10])
            + (f" ... ({len(skipped) - 10} more)" if len(skipped) > 10 else "")
        )
    blob, entries = pack_f32_arrays(to_pack)
    return blob, entries, skipped


# ── Step 5: manifest, reference run, golden, report ───────────────────────────

def _io_entries(value_infos, shape_map: dict[str, dict], kind: str | None) -> list[dict[str, Any]]:
    entries = []
    for vi in value_infos:
        entry: dict[str, Any] = {"name": vi.name}
        if kind:
            entry["kind"] = kind
        info = shape_map.get(vi.name, {})
        if info.get("shape"):
            entry["shape"] = info["shape"]
            entry["dtype"] = info.get("dtype", "float32")
        entries.append(entry)
    return entries


def _op_counts(graph_nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in graph_nodes:
        if n["op"] == "call_function":
            counts[n["target"]] = counts.get(n["target"], 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def _example_input_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _run_onnx_reference_for_manifest_nodes(
    onnx_model,
    input_specs: list,
    example_inputs: tuple,
    graph_nodes: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run onnx.reference over the model with every manifest call_function node's ONNX
    tensor exposed as a graph output. Returns ({safe_name: value}, {input name: feed})."""
    # Sorted: iteration order here determines golden.json's node order.
    call_node_names = sorted(
        node["name"] for node in graph_nodes
        if (
            node["op"] == "call_function"
            and node.get("target") != "aten.alias.default"
            and not node.get("meta", {}).get("onnx_fallback_op")
        )
    )
    output_name_by_safe = {
        safe_name(out_name): out_name
        for node in onnx_model.graph.node
        for out_name in node.output
        if out_name
    }
    capture_output_names = [output_name_by_safe[name] for name in call_node_names if name in output_name_by_safe]

    model_for_golden = onnx.ModelProto()
    model_for_golden.CopyFrom(onnx_model)
    existing_outputs = {out.name for out in model_for_golden.graph.output}
    value_infos = {
        vi.name: vi
        for vi in list(model_for_golden.graph.input)
        + list(model_for_golden.graph.output)
        + list(model_for_golden.graph.value_info)
    }
    for name in capture_output_names:
        if name in existing_outputs:
            continue
        if name in value_infos:
            model_for_golden.graph.output.append(value_infos[name])
        else:
            model_for_golden.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
        existing_outputs.add(name)

    feeds = {spec.name: _example_input_to_numpy(value) for spec, value in zip(input_specs, example_inputs)}
    outputs = ReferenceEvaluator(model_for_golden).run(capture_output_names, feeds) if capture_output_names else []
    values_by_safe = {safe_name(name): np.asarray(value) for name, value in zip(capture_output_names, outputs)}
    return values_by_safe, feeds


def _concretize_manifest_shapes_from_reference(
    manifest: dict[str, Any],
    graph_nodes: list[dict[str, Any]],
    values_by_safe: dict[str, np.ndarray],
    feeds: dict[str, np.ndarray],
) -> None:
    def concrete_shape(name: str, shape: list[int] | None) -> list[int] | None:
        arr = values_by_safe.get(name)
        if arr is None:
            arr = feeds.get(name)
        if arr is None:
            return shape
        actual = list(arr.shape)
        if shape is None or len(shape) != len(actual) or any(d == -1 for d in shape):
            return actual
        return shape

    for node in graph_nodes:
        meta = node.get("meta", {})
        if "shape" in meta:
            meta["shape"] = concrete_shape(node["name"], meta.get("shape"))

    for entry in manifest.get("inputs", []) + manifest.get("outputs", []):
        if "shape" in entry:
            entry["shape"] = concrete_shape(safe_name(entry["name"]), entry.get("shape"))


def _capture_onnx_golden_from_reference(values_by_safe: dict[str, np.ndarray], feeds: dict[str, np.ndarray]) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for name in sorted(values_by_safe):
        stats = value_stats(values_by_safe[name])
        if stats is not None:
            nodes[name] = stats
    inputs_json = {
        safe_name(name): value.astype(np.float32, copy=False).reshape(-1).tolist()
        for name, value in sorted(feeds.items())
    }
    return {"inputs": inputs_json, "nodes": nodes}


def _debug_report(
    graph_nodes: list[dict[str, Any]],
    weight_entries: list[dict[str, Any]],
    op_counts: dict[str, int],
    unsupported: list[str],
    warnings: list[str],
    opset_version: int,
) -> str:
    total_bytes = sum(w["byte_length"] for w in weight_entries)
    lines = [
        "# Kuma ONNX Backend Debug Report",
        "",
        "## Graph",
        f"- {len(graph_nodes)} nodes  ({len(weight_entries)} weight tensors, {total_bytes / 1024 / 1024:.1f} MB)",
        f"- opset {opset_version}",
        "",
        "## Op counts",
        *(f"- {target}: {count}" for target, count in op_counts.items()),
    ]
    if unsupported:
        lines += ["", "## Unsupported ONNX ops", *(f"- {op}" for op in unsupported)]
    if warnings:
        lines += ["", "## Warnings", *(f"- {w}" for w in warnings)]
    return "\n".join(lines) + "\n"


# ── Public entry points ───────────────────────────────────────────────────────

def export_via_onnx(
    model: nn.Module,
    example_inputs: tuple,
    *,
    opset_version: int = 17,
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> Package:
    """Export a PyTorch model to a kuma Package via ONNX as an intermediate step.

    Uses torch.onnx.export (classic exporter, opset 17 by default) then walks the ONNX
    graph, maps each op to the corresponding aten target in kuma-bart's opRegistry
    (kuma.onnx_ops), and packs weight initializers into the standard f32 blob. Raises
    ValueError if any op reachable from the output can't be converted faithfully.
    """
    model.eval()
    validate_model_and_inputs(model, example_inputs)

    onnx_model = _torch_to_onnx(model, example_inputs, opset_version)
    graph = onnx_model.graph

    init_values = _collect_static_values(graph)
    shape_map = _build_shape_map(graph)
    _fold_static_onnx_values(graph, init_values, shape_map)
    input_specs = [vi for vi in graph.input if vi.name not in init_values]
    _add_matmul_transposed_weights(graph, init_values, shape_map)
    _check_safe_name_collisions(graph, init_values)

    graph_nodes, warnings, unsupported = _build_graph(graph, input_specs, init_values, shape_map)
    graph_nodes = prune_unused_placeholders(_prune_unreachable_nodes(graph_nodes))
    _check_no_reachable_fallbacks(graph_nodes, warnings)

    weights_blob, weight_entries, skipped = _pack_used_weights(graph_nodes, init_values)

    op_counts = _op_counts(graph_nodes)
    manifest: dict[str, Any] = {
        "format": "kuma",
        "format_version": 0,
        "weight_file": "weights.f32.bin",
        "endianness": "little",
        "inputs": _io_entries(input_specs, shape_map, "user_input"),
        "outputs": _io_entries(graph.output, shape_map, None),
        "weights": weight_entries,
        "graph": {
            "node_count": len(graph_nodes),
            "op_counts": op_counts,
            "nodes": graph_nodes,
        },
        "warnings": warnings,
        "unsupported_ops": unsupported,
    }
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    reference_values, reference_feeds = _run_onnx_reference_for_manifest_nodes(
        onnx_model, input_specs, example_inputs, graph_nodes
    )
    _concretize_manifest_shapes_from_reference(manifest, graph_nodes, reference_values, reference_feeds)

    return Package(
        manifest=manifest,
        weights_blob=weights_blob,
        graph_data=manifest["graph"],
        debug_report=_debug_report(graph_nodes, weight_entries, op_counts, unsupported, warnings, opset_version),
        kernels=load_kernels(),
        skipped=skipped,
        golden={
            "format_version": 0,
            "branches": [_capture_onnx_golden_from_reference(reference_values, reference_feeds)],
        },
    )


def compile_branching_onnx_kuma(
    router_snippet_name: str,
    router_snippet_source: str,
    router_input_names: list[str],
    router_output_specs: list[dict[str, Any]],
    selector_output_index: int,
    branch_input_output_index: int,
    branch_wrappers: list[nn.Module],
    branch_example_inputs: list[tuple],
    *,
    fps: float | None = None,
    duration_seconds: float | None = None,
    opset_version: int = 17,
) -> Package:
    """Export each branch wrapper via ONNX → kuma graph and assemble a branching .iph.

    Drop-in replacement for kuma.onnx_compiler.compile_branching_onnx:
    same router/branch interface, but the output package has format: "kuma" (native
    WGSL graphs) rather than format: "onnx-branching" (ort-web segments).

    router_snippet_name   — JS identifier for the router function (embedded in package)
    router_snippet_source — full JS source of the router (kuma-bart evaluates it)
    router_input_names    — names of top-level inputs the snippet reads
    router_output_specs   — [{"shape": [...], "dtype": "..."}] one per router output
    selector_output_index       — which router output picks the branch (int index)
    branch_input_output_index   — which router output is the routed scalar fed to each branch
    branch_wrappers       — one nn.Module per branch (must accept exactly 1 user input;
                            all must produce the same output shape)
    branch_example_inputs — one example-inputs tuple per branch (used for ONNX tracing)
    opset_version         — ONNX opset for torch.onnx.export (default 17)
    """
    validate_branching_args(
        router_output_specs=router_output_specs,
        selector_output_index=selector_output_index,
        branch_input_output_index=branch_input_output_index,
        n_branches=len(branch_wrappers),
        n_example_inputs=len(branch_example_inputs),
        branches_arg="branch_wrappers",
    )

    branches: list[KumaBranch] = []
    warnings: list[str] = []
    unsupported: list[str] = []
    golden_branches: list[dict[str, Any]] = []

    for i, (wrapper, example_inputs) in enumerate(zip(branch_wrappers, branch_example_inputs)):
        pkg = export_via_onnx(wrapper, example_inputs, opset_version=opset_version)
        golden_branches.append(namespace_golden(pkg.golden["branches"][0], f"branch{i}__"))
        warnings.extend(f"branch {i}: {w}" for w in pkg.manifest["warnings"])
        for op in pkg.manifest["unsupported_ops"]:
            if op not in unsupported:
                unsupported.append(op)
        branches.append(
            KumaBranch(nodes=pkg.graph_data["nodes"], weights_blob=pkg.weights_blob, weight_entries=pkg.manifest["weights"])
        )

    return assemble_kuma_branching_package(
        router_snippet_name=router_snippet_name,
        router_snippet_source=router_snippet_source,
        router_input_names=router_input_names,
        router_output_specs=router_output_specs,
        selector_output_index=selector_output_index,
        branch_input_output_index=branch_input_output_index,
        branches=branches,
        warnings=warnings,
        unsupported_ops=unsupported,
        golden_branches=golden_branches,
        fps=fps,
        duration_seconds=duration_seconds,
        report_title="Kuma ONNX-Kuma Branching Export Debug Report",
        report_extra_lines=[f"- ONNX opset: {opset_version}"],
    )
