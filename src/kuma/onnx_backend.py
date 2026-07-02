"""ONNX backend: torch.nn.Module → ONNX → kuma Package.

Provides an alternative compilation path where the model is first exported to
ONNX (via torch.onnx.export), then converted to kuma's native manifest+WGSL
format.  The ONNX op set is stable and well-documented, making the kuma kernel
mapping easier to maintain and straightforward to port to TypeScript later.

Entry point: export_via_onnx(model, example_inputs, ...) → Package
"""

from __future__ import annotations

import io
import re
import warnings as py_warnings
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from kuma.kernels import load_kernels
from kuma.manifest import build_playback_meta
from kuma.package_iph import Package


_FIRST_SAMPLES = 8
_SPREAD_SAMPLES = 8
_SPREAD_SEED = 0x6B756D61


# ── ONNX attribute helpers ────────────────────────────────────────────────────

def _attr(node, name: str, default=None):
    """Read a named attribute from an ONNX NodeProto."""
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == 1:    return a.f           # FLOAT
        if a.type == 2:    return int(a.i)      # INT
        if a.type == 3:    return a.s            # STRING
        if a.type == 6:    return list(a.floats) # FLOATS
        if a.type == 7:    return [int(v) for v in a.ints]  # INTS
        if a.type == 4:    # TENSOR
            from onnx import numpy_helper
            return numpy_helper.to_array(a.t)
    return default


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _init_list(init_values: dict[str, np.ndarray], name: str | None) -> list[int] | None:
    if not name or name not in init_values:
        return None
    return [int(v) for v in init_values[name].reshape(-1)]


def _init_scalar(init_values: dict[str, np.ndarray], name: str | None) -> int | float | None:
    if not name or name not in init_values:
        return None
    arr = init_values[name].reshape(-1)
    if arr.size != 1:
        return None
    value = arr[0]
    return int(value) if np.issubdtype(arr.dtype, np.integer) else float(value)


_ONNX_CAST_DTYPES: dict[int, Any] = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
}


def _get_shape(type_proto) -> list[int] | None:
    t = type_proto.tensor_type
    if not t.HasField("shape"):
        return None
    dims = []
    for d in t.shape.dim:
        dims.append(int(d.dim_value) if d.HasField("dim_value") else -1)
    return dims


_ONNX_ELEM_TYPE: dict[int, str] = {
    1: "float32", 2: "uint8", 3: "int8", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
}


def _get_dtype(type_proto) -> str:
    return _ONNX_ELEM_TYPE.get(type_proto.tensor_type.elem_type, "unknown")


def _constant_array_from_node(node) -> np.ndarray | None:
    for attr in node.attribute:
        if attr.name == "value":
            from onnx import numpy_helper
            return numpy_helper.to_array(attr.t)
        if attr.name == "value_int":
            return np.asarray(attr.i, dtype=np.int64)
        if attr.name == "value_ints":
            return np.asarray(list(attr.ints), dtype=np.int64)
        if attr.name == "value_float":
            return np.asarray(attr.f, dtype=np.float32)
        if attr.name == "value_floats":
            return np.asarray(list(attr.floats), dtype=np.float32)
    return None


def _shape_array_for(name: str, shape_map: dict[str, dict]) -> np.ndarray | None:
    shape = shape_map.get(name, {}).get("shape")
    if shape is None or any(d is None for d in shape):
        return None
    return np.asarray(shape, dtype=np.int64)


def _fold_static_onnx_values(graph, init_values: dict[str, np.ndarray], shape_map: dict[str, dict]) -> None:
    """Evaluate small constant ONNX shape/index subgraphs.

    ONNX commonly represents static slice bounds and reshape sizes as little graphs
    made from Shape/Gather/Slice/Cast/Unsqueeze/Concat. Kuma converters want those as
    Python attrs; leaving them as executable graph nodes creates reachable fallbacks.
    """
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if not node.output:
                continue
            dst = node.output[0]
            if dst in init_values:
                continue
            inp = list(node.input)
            value: np.ndarray | None = None

            try:
                if node.op_type == "Shape" and inp:
                    shape = _shape_array_for(inp[0], shape_map)
                    if shape is not None:
                        start = int(_attr(node, "start", 0))
                        end_attr = _attr(node, "end", None)
                        end = len(shape) if end_attr is None else int(end_attr)
                        value = shape[start:end]

                elif node.op_type == "Cast" and inp and inp[0] in init_values:
                    dtype = _ONNX_CAST_DTYPES.get(int(_attr(node, "to", 0)))
                    if dtype is not None:
                        value = init_values[inp[0]].astype(dtype, copy=False)

                elif node.op_type == "Unsqueeze" and inp and inp[0] in init_values:
                    axes = _init_list(init_values, inp[1] if len(inp) > 1 else None)
                    if axes is None:
                        axes = _attr(node, "axes", None)
                    if axes is not None:
                        value = init_values[inp[0]]
                        rank = value.ndim + len(axes)
                        for axis in sorted((a + rank if a < 0 else a) for a in axes):
                            value = np.expand_dims(value, axis)

                elif node.op_type == "Squeeze" and inp and inp[0] in init_values:
                    axes = _init_list(init_values, inp[1] if len(inp) > 1 else None)
                    if axes is None:
                        axes = _attr(node, "axes", None)
                    value = np.squeeze(init_values[inp[0]], axis=None if axes is None else tuple(int(a) for a in axes))

                elif node.op_type == "Reshape" and len(inp) >= 2 and inp[0] in init_values and inp[1] in init_values:
                    shape = [int(v) for v in init_values[inp[1]].reshape(-1)]
                    value = np.reshape(init_values[inp[0]], shape)

                elif node.op_type == "Concat" and inp and all(i in init_values for i in inp):
                    axis = int(_attr(node, "axis", 0))
                    value = np.concatenate([init_values[i] for i in inp], axis=axis)

                elif node.op_type == "Gather" and len(inp) >= 2 and inp[0] in init_values and inp[1] in init_values:
                    axis = int(_attr(node, "axis", 0))
                    value = np.take(init_values[inp[0]], init_values[inp[1]].astype(np.int64), axis=axis)

                elif node.op_type in ("Add", "Sub", "Mul", "Div") and len(inp) >= 2 and inp[0] in init_values and inp[1] in init_values:
                    a = init_values[inp[0]]
                    b = init_values[inp[1]]
                    if node.op_type == "Add":
                        value = np.add(a, b)
                    elif node.op_type == "Sub":
                        value = np.subtract(a, b)
                    elif node.op_type == "Mul":
                        value = np.multiply(a, b)
                    else:
                        value = np.divide(a, b)

                elif node.op_type == "Floor" and inp and inp[0] in init_values:
                    value = np.floor(init_values[inp[0]])

                elif node.op_type == "Slice" and len(inp) >= 3 and inp[0] in init_values:
                    starts = _init_list(init_values, inp[1])
                    ends = _init_list(init_values, inp[2])
                    axes = _init_list(init_values, inp[3] if len(inp) > 3 else None)
                    steps = _init_list(init_values, inp[4] if len(inp) > 4 else None)
                    if starts is not None and ends is not None:
                        arr = init_values[inp[0]]
                        axes = axes if axes is not None else list(range(len(starts)))
                        steps = steps if steps is not None else [1] * len(starts)
                        if len(starts) == len(ends) == len(axes) == len(steps):
                            slices = [slice(None)] * arr.ndim
                            for axis, start, end, step in zip(axes, starts, ends, steps):
                                slices[int(axis)] = slice(int(start), int(end), int(step))
                            value = arr[tuple(slices)]

                elif node.op_type == "ConstantOfShape" and inp and inp[0] in init_values:
                    shape = tuple(int(v) for v in init_values[inp[0]].reshape(-1))
                    fill = _attr(node, "value", None)
                    scalar = np.asarray(fill).reshape(-1)[0] if fill is not None else np.float32(0)
                    if int(np.prod(shape, dtype=np.int64)) <= 1024:
                        value = np.full(shape, scalar)

            except Exception:
                value = None

            if value is not None:
                init_values[dst] = np.asarray(value)
                changed = True


# ── Name sanitisation ─────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    """Convert an ONNX tensor name (may contain /.:) to a safe node identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
    return s or "tensor"


def _ref(name: str) -> dict[str, str]:
    return {"node_ref": _safe(name)}


def _ref_or_scalar(init_values: dict[str, np.ndarray], name: str) -> dict[str, str] | int | float:
    value = _init_scalar(init_values, name)
    if value is not None:
        return value
    return _ref(name)


def _matmul_transposed_weight_name(name: str) -> str:
    return f"{name}__kuma_transposed"


def _spread_indices(n: int, count: int = _SPREAD_SAMPLES) -> list[int]:
    if n == 0:
        return []
    state = _SPREAD_SEED
    indices = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        indices.append(state % n)
    return indices


def _numpy_tensor_stats(value: Any) -> dict[str, Any] | None:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
        return None
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    finite_mask = np.isfinite(flat)
    finite_vals = flat[finite_mask]
    n = int(flat.size)
    nfin = int(finite_mask.sum())
    spread_idx = _spread_indices(n)
    return {
        "shape": list(arr.shape),
        "n": n,
        "finite": nfin,
        "mean": float(finite_vals.mean()) if nfin else float("nan"),
        "min": float(finite_vals.min()) if nfin else float("nan"),
        "max": float(finite_vals.max()) if nfin else float("nan"),
        "first": flat[: min(_FIRST_SAMPLES, n)].tolist(),
        "spread_indices": spread_idx,
        "spread": [float(flat[i]) for i in spread_idx],
    }


def _numpy_value_stats(value: Any) -> dict[str, Any] | None:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        re = _numpy_tensor_stats(arr.real)
        im = _numpy_tensor_stats(arr.imag)
        if re is None or im is None:
            return None
        return {"re": re, "im": im}
    stats = _numpy_tensor_stats(arr)
    return {"re": stats} if stats is not None else None


def _example_input_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _run_onnx_reference_for_manifest_nodes(
    onnx_model,
    input_specs: list,
    example_inputs: tuple,
    graph_nodes: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import onnx
        from onnx import TensorProto, helper
        from onnx.reference import ReferenceEvaluator
    except ImportError as e:
        raise ImportError("ONNX reference capture requires onnx.reference from the `onnx` package.") from e

    call_node_names = {
        node["name"] for node in graph_nodes
        if (
            node["op"] == "call_function"
            and node.get("target") != "aten.alias.default"
            and not node.get("meta", {}).get("onnx_fallback_op")
        )
    }
    output_name_by_safe = {
        _safe(out_name): out_name
        for node in onnx_model.graph.node
        for out_name in node.output
        if _safe(out_name) in call_node_names
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

    feeds = {
        spec.name: _example_input_to_numpy(value)
        for spec, value in zip(input_specs, example_inputs)
    }
    ref = ReferenceEvaluator(model_for_golden)
    outputs = ref.run(capture_output_names, feeds) if capture_output_names else []
    values_by_safe = {_safe(name): np.asarray(value) for name, value in zip(capture_output_names, outputs)}
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

    for entry in manifest.get("inputs", []):
        if "shape" in entry:
            entry["shape"] = concrete_shape(_safe(entry["name"]), entry.get("shape"))

    for entry in manifest.get("outputs", []):
        if "shape" in entry:
            entry["shape"] = concrete_shape(_safe(entry["name"]), entry.get("shape"))


def _capture_onnx_golden_from_reference(values_by_safe: dict[str, np.ndarray], feeds: dict[str, np.ndarray]) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for safe_name, value in values_by_safe.items():
        stats = _numpy_value_stats(value)
        if stats is not None:
            nodes[safe_name] = stats

    inputs_json = {
        _safe(name): value.astype(np.float32, copy=False).reshape(-1).tolist()
        for name, value in feeds.items()
    }
    return {"inputs": inputs_json, "nodes": nodes}


# ── Op converters ─────────────────────────────────────────────────────────────
#
# Each converter(node, init_values, shape_map) returns either:
#   {"target": str, "args": list, "kwargs": dict}   — success
#   None                                              — can't convert (warn + alias)

_CONVERTERS: dict[str, Callable] = {}


def _op(*op_types: str):
    def decorator(fn: Callable) -> Callable:
        for t in op_types:
            _CONVERTERS[t] = fn
        return fn
    return decorator


@_op("Conv")
def _conv(node, init_values, shape_map):
    inp = list(node.input)
    pads      = _attr(node, "pads", [0, 0, 0, 0])
    strides   = _attr(node, "strides", [1, 1])
    dilations = _attr(node, "dilations", [1, 1])
    groups    = _attr(node, "group", 1)
    padding   = pads[: len(pads) // 2]  # [top, left] from [top, left, bottom, right]
    bias      = _ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.convolution.default",
        "args": [_ref(inp[0]), _ref(inp[1]), bias, strides, padding, dilations, False, [0, 0], groups],
        "kwargs": {},
    }


@_op("ConvTranspose")
def _conv_transpose(node, init_values, shape_map):
    inp            = list(node.input)
    pads           = _attr(node, "pads", [0, 0, 0, 0])
    strides        = _attr(node, "strides", [1, 1])
    dilations      = _attr(node, "dilations", [1, 1])
    output_padding = _attr(node, "output_padding", [0, 0])
    groups         = _attr(node, "group", 1)
    padding        = pads[: len(pads) // 2]
    bias           = _ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.convolution.default",
        "args": [_ref(inp[0]), _ref(inp[1]), bias, strides, padding, dilations, True, output_padding, groups],
        "kwargs": {},
    }


@_op("Relu")
def _relu(node, init_values, shape_map):
    return {"target": "aten.relu.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Gelu")
def _gelu(node, init_values, shape_map):
    return {"target": "aten.gelu.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Sigmoid")
def _sigmoid(node, init_values, shape_map):
    return {"target": "aten.sigmoid.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Tanh")
def _tanh(node, init_values, shape_map):
    return {"target": "aten.tanh.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Silu")
def _silu(node, init_values, shape_map):
    return {"target": "aten.silu.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Sqrt")
def _sqrt(node, init_values, shape_map):
    return {"target": "aten.sqrt.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Cos")
def _cos(node, init_values, shape_map):
    return {"target": "aten.cos.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Sin")
def _sin(node, init_values, shape_map):
    return {"target": "aten.sin.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Floor")
def _floor(node, init_values, shape_map):
    return {"target": "aten.floor.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Erf")
def _erf(node, init_values, shape_map):
    return {"target": "aten.erf.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Add")
def _add(node, init_values, shape_map):
    return {
        "target": "aten.add.Tensor",
        "args": [_ref_or_scalar(init_values, node.input[0]), _ref_or_scalar(init_values, node.input[1])],
        "kwargs": {},
    }


@_op("Sub")
def _sub(node, init_values, shape_map):
    return {
        "target": "aten.sub.Tensor",
        "args": [_ref_or_scalar(init_values, node.input[0]), _ref_or_scalar(init_values, node.input[1])],
        "kwargs": {},
    }


@_op("Mul")
def _mul(node, init_values, shape_map):
    return {
        "target": "aten.mul.Tensor",
        "args": [_ref_or_scalar(init_values, node.input[0]), _ref_or_scalar(init_values, node.input[1])],
        "kwargs": {},
    }


@_op("Div")
def _div(node, init_values, shape_map):
    return {
        "target": "aten.div.Tensor",
        "args": [_ref_or_scalar(init_values, node.input[0]), _ref_or_scalar(init_values, node.input[1])],
        "kwargs": {},
    }


@_op("Pow")
def _pow(node, init_values, shape_map):
    exp_name = node.input[1]
    if exp_name in init_values:
        scalar = float(init_values[exp_name].flat[0])
        return {"target": "aten.pow.Tensor_Scalar", "args": [_ref(node.input[0]), scalar], "kwargs": {}}
    return None  # dynamic exponent — unsupported


@_op("Gemm")
def _gemm(node, init_values, shape_map):
    inp = list(node.input)
    # Standard nn.Linear: transA=0, transB=1.  linear.wgsl expects weight [N, K]
    # which is exactly what the ONNX Gemm weight initializer has when transB=1.
    # aten.linear.default(input, weight, bias) dispatches through linearHandler.
    bias = _ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {"target": "aten.linear.default", "args": [_ref(inp[0]), _ref(inp[1]), bias], "kwargs": {}}


@_op("MatMul")
def _matmul(node, init_values, shape_map):
    rhs = node.input[1]
    if rhs in init_values and init_values[rhs].ndim == 2:
        rhs = _matmul_transposed_weight_name(rhs)
    return {"target": "aten.mm.default", "args": [_ref(node.input[0]), _ref(rhs)], "kwargs": {}}


@_op("LayerNormalization")
def _layernorm(node, init_values, shape_map):
    inp = list(node.input)
    eps = _attr(node, "epsilon", 1e-5)
    weight_name = inp[1] if len(inp) > 1 and inp[1] else None
    # normalized_shape comes from the scale initializer's shape
    normalized_shape = list(init_values[weight_name].shape) if weight_name and weight_name in init_values else []
    bias = _ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.layer_norm.default",
        "args": [_ref(inp[0]), normalized_shape, _ref(weight_name) if weight_name else None, bias, eps],
        "kwargs": {},
    }


@_op("GroupNormalization")
def _groupnorm(node, init_values, shape_map):
    inp       = list(node.input)
    num_groups = _attr(node, "num_groups", 1)
    eps       = _attr(node, "epsilon", 1e-5)
    weight    = _ref(inp[1]) if len(inp) > 1 and inp[1] else None
    bias      = _ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.group_norm.default",
        "args": [_ref(inp[0]), num_groups, weight, bias, eps],
        "kwargs": {},
    }


@_op("InstanceNormalization")
def _instancenorm(node, init_values, shape_map):
    inp = list(node.input)
    in_shape = shape_map.get(inp[0], {}).get("shape", [])
    if len(in_shape) < 2 or in_shape[1] in (-1, None):
        return None
    channels = int(in_shape[1])
    eps = _attr(node, "epsilon", 1e-5)
    weight = _ref(inp[1]) if len(inp) > 1 and inp[1] else None
    bias = _ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.group_norm.default",
        "args": [_ref(inp[0]), channels, weight, bias, eps],
        "kwargs": {},
    }


@_op("Reshape")
def _reshape(node, init_values, shape_map):
    inp = list(node.input)
    shape_src = inp[1] if len(inp) > 1 else None
    if shape_src and shape_src in init_values:
        shape = [int(v) for v in init_values[shape_src].flat]
        return {"target": "aten.reshape.default", "args": [_ref(inp[0]), shape], "kwargs": {}}
    out_shape = shape_map.get(node.output[0], {}).get("shape") if node.output else None
    if out_shape:
        return {"target": "aten.reshape.default", "args": [_ref(inp[0]), out_shape], "kwargs": {}}
    # Dynamic shape — not statically representable
    return None


@_op("Flatten")
def _flatten(node, init_values, shape_map):
    axis    = _attr(node, "axis", 1)
    in_info = shape_map.get(node.input[0], {})
    in_shape = in_info.get("shape")
    if in_shape:
        out_shape = list(in_shape[:axis]) + [int(np.prod(in_shape[axis:]))]
        return {"target": "aten.reshape.default", "args": [_ref(node.input[0]), out_shape], "kwargs": {}}
    return None


@_op("Transpose")
def _transpose(node, init_values, shape_map):
    perm = _attr(node, "perm", None)
    if perm is None:
        return None
    return {"target": "aten.permute.default", "args": [_ref(node.input[0]), perm], "kwargs": {}}


@_op("Concat")
def _concat(node, init_values, shape_map):
    axis = _attr(node, "axis", 0)
    return {"target": "aten.cat.default", "args": [[_ref(i) for i in node.input], axis], "kwargs": {}}


@_op("Einsum")
def _einsum(node, init_values, shape_map):
    equation = _text(_attr(node, "equation", ""))
    if not equation:
        return None
    return {"target": "aten.einsum.default", "args": [equation, [_ref(i) for i in node.input]], "kwargs": {}}


@_op("Expand")
def _expand(node, init_values, shape_map):
    inp = list(node.input)
    size = _init_list(init_values, inp[1] if len(inp) > 1 else None)
    if size is None:
        size = shape_map.get(node.output[0], {}).get("shape") if node.output else None
    if not size:
        return None
    return {"target": "aten.expand.default", "args": [_ref(inp[0]), size], "kwargs": {}}


@_op("ReduceSum")
def _reducesum(node, init_values, shape_map):
    inp = list(node.input)
    axes = _init_list(init_values, inp[1] if len(inp) > 1 else None)
    if axes is None:
        axes = _attr(node, "axes", None)
    if axes is None:
        in_shape = shape_map.get(inp[0], {}).get("shape")
        if not in_shape:
            return None
        axes = list(range(len(in_shape)))
    return {"target": "aten.sum.dim_IntList", "args": [_ref(inp[0]), [int(a) for a in axes]], "kwargs": {}}


@_op("Gather")
def _gather(node, init_values, shape_map):
    inp = list(node.input)
    axis = int(_attr(node, "axis", 0))
    index_scalar = _init_scalar(init_values, inp[1] if len(inp) > 1 else None)
    if isinstance(index_scalar, int):
        return {"target": "aten.select.int", "args": [_ref(inp[0]), axis, index_scalar], "kwargs": {}}
    if axis == 0:
        return {"target": "aten.index.Tensor", "args": [_ref(inp[0]), [_ref(inp[1])]], "kwargs": {}}
    return {"target": "aten.gather.default", "args": [_ref(inp[0]), axis, _ref(inp[1])], "kwargs": {}}


@_op("Slice")
def _slice(node, init_values, shape_map):
    inp = list(node.input)
    starts = _init_list(init_values, inp[1] if len(inp) > 1 else None)
    ends = _init_list(init_values, inp[2] if len(inp) > 2 else None)
    axes = _init_list(init_values, inp[3] if len(inp) > 3 else None)
    steps = _init_list(init_values, inp[4] if len(inp) > 4 else None)
    starts = starts if starts is not None else _attr(node, "starts", None)
    ends = ends if ends is not None else _attr(node, "ends", None)
    axes = axes if axes is not None else _attr(node, "axes", None)
    steps = steps if steps is not None else _attr(node, "steps", None)
    if starts is None or ends is None:
        return None
    axes = axes if axes is not None else list(range(len(starts)))
    steps = steps if steps is not None else [1] * len(starts)
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        return None
    if len(starts) != 1:
        return {
            "target": "aten.slice_multi.default",
            "args": [_ref(inp[0]), [int(v) for v in axes], [int(v) for v in starts], [int(v) for v in ends], [int(v) for v in steps]],
            "kwargs": {},
        }
    return {
        "target": "aten.slice.Tensor",
        "args": [_ref(inp[0]), int(axes[0]), int(starts[0]), int(ends[0]), int(steps[0])],
        "kwargs": {},
    }


@_op("DepthToSpace")
def _depth_to_space(node, init_values, shape_map):
    blocksize = int(_attr(node, "blocksize", 0))
    mode = _text(_attr(node, "mode", "DCR"))
    if blocksize <= 0:
        return None
    # PyTorch PixelShuffle exports as CRD. DCR has a different channel ordering.
    if mode.upper() != "CRD":
        return None
    return {"target": "aten.pixel_shuffle.default", "args": [_ref(node.input[0]), blocksize], "kwargs": {}}


@_op("Clip")
def _clip(node, init_values, shape_map):
    inp = list(node.input)
    def _scalar(name):
        if name and name in init_values:
            return float(init_values[name].flat[0])
        return None
    min_val = _scalar(inp[1]) if len(inp) > 1 else _attr(node, "min", None)
    max_val = _scalar(inp[2]) if len(inp) > 2 else _attr(node, "max", None)
    return {"target": "aten.clamp.default", "args": [_ref(inp[0]), min_val, max_val], "kwargs": {}}


@_op("Identity", "Dropout")
def _identity(node, init_values, shape_map):
    # Dropout is inference-only here — treated as identity
    return {"target": "aten.alias.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Cast")
def _cast(node, init_values, shape_map):
    # Kuma stores runtime tensors as f32 buffers. For model data this is an identity
    # in the observed ONNX graphs; for index tensors, kernels read f32-encoded ints.
    # Constant casts are folded into init_values before conversion, so shape/index
    # side casts are normally pruned away entirely.
    return {"target": "aten.alias.default", "args": [_ref(node.input[0])], "kwargs": {}}


@_op("Squeeze", "Unsqueeze")
def _squeeze_unsqueeze(node, init_values, shape_map):
    # Shape change only — passthrough in kuma (meta.shape carries the updated shape)
    return {"target": "aten.alias.default", "args": [_ref(node.input[0])], "kwargs": {}}


# ── Weight packing ────────────────────────────────────────────────────────────

def _pack_weights(
    init_values: dict[str, np.ndarray],
    used: set[str],
) -> tuple[bytes, list[dict[str, Any]], list[str]]:
    """Pack used float32 initializers into a contiguous blob (same layout as pack_weights.py)."""
    blob     = bytearray()
    entries: list[dict[str, Any]] = []
    skipped: list[str] = []

    for name in sorted(used):
        arr_orig = init_values.get(name)
        if arr_orig is None:
            continue
        if not (
            np.issubdtype(arr_orig.dtype, np.floating)
            or np.issubdtype(arr_orig.dtype, np.integer)
            or np.issubdtype(arr_orig.dtype, np.bool_)
        ):
            skipped.append(f"{name} ({arr_orig.dtype})")
            continue
        # Runtime buffers are f32. Integer ONNX initializers that survive pruning are
        # data/index tensors, not shape attrs; gather kernels intentionally read them
        # as f32-encoded integer values.
        arr = arr_orig.astype("<f4", copy=False).flatten()
        pad = (-len(blob)) % 4
        if pad:
            blob.extend(b"\x00" * pad)
        raw = arr.tobytes()
        entries.append({
            "name":        name,
            "shape":       list(arr_orig.shape),
            "dtype":       "float32",
            "byte_offset": len(blob),
            "byte_length": len(raw),
            "n_elements":  int(arr.size),
        })
        blob.extend(raw)

    return bytes(blob), entries, skipped


# ── Graph conversion ──────────────────────────────────────────────────────────

def _build_graph(
    graph,
    input_specs: list,
    init_values: dict[str, np.ndarray],
    shape_map: dict[str, dict],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Walk an ONNX GraphProto and emit kuma manifest nodes."""
    nodes:       list[dict[str, Any]] = []
    warnings:    list[str] = []
    unsupported: list[str] = []
    node_id = 0

    for n in graph.node:
        if n.op_type != "MatMul" or len(n.input) < 2:
            continue
        rhs = n.input[1]
        arr = init_values.get(rhs)
        if arr is None or arr.ndim != 2:
            continue
        transposed_name = _matmul_transposed_weight_name(rhs)
        if transposed_name not in init_values:
            init_values[transposed_name] = np.ascontiguousarray(arr.T)
            shape_map[transposed_name] = {"shape": list(init_values[transposed_name].shape), "dtype": "float32"}

    init_names = set(init_values)

    # Initializers actually referenced by ops
    used_inits: set[str] = set()
    for n in graph.node:
        for idx, inp in enumerate(n.input):
            if n.op_type == "MatMul" and idx == 1:
                arr = init_values.get(inp)
                if arr is not None and arr.ndim == 2:
                    used_inits.add(_matmul_transposed_weight_name(inp))
                    continue
            if inp in init_names:
                used_inits.add(inp)

    # User-input placeholders
    for spec in input_specs:
        info = shape_map.get(spec.name, {})
        meta: dict[str, Any] = {}
        if info.get("shape") is not None:
            meta["shape"] = info["shape"]
            meta["dtype"] = info.get("dtype", "float32")
        nodes.append({
            "id":     node_id,
            "name":   _safe(spec.name),
            "op":     "placeholder",
            "target": _safe(spec.name),
            "args":   [], "kwargs": {}, "meta": meta,
            "kind":   "user_input",
        })
        node_id += 1

    # Weight placeholders (sorted for deterministic ordering)
    for name in sorted(used_inits):
        arr = init_values[name]
        nodes.append({
            "id":          node_id,
            "name":        _safe(name),
            "op":          "placeholder",
            "target":      _safe(name),
            "args":        [], "kwargs": {},
            "meta":        {"shape": list(arr.shape), "dtype": "float32"},
            "kind":        "parameter",
            "weight_name": name,
        })
        node_id += 1

    # Op nodes
    for onnx_node in graph.node:
        op_type  = onnx_node.op_type
        if op_type in ("Constant", "Shape") and onnx_node.output and onnx_node.output[0] in init_values:
            continue
        out_name = onnx_node.output[0] if onnx_node.output else f"_node_{node_id}"
        info     = shape_map.get(out_name, {})
        meta     = {}
        if info.get("shape") is not None:
            meta["shape"] = info["shape"]
            meta["dtype"] = info.get("dtype", "float32")

        converter = _CONVERTERS.get(op_type)
        if converter is None:
            if op_type not in unsupported:
                unsupported.append(op_type)
            warnings.append(f"Unsupported ONNX op: {op_type} (→ {out_name})")
            _emit_fallback(nodes, node_id, out_name, onnx_node, meta)
            node_id += 1
            continue

        result = converter(onnx_node, init_values, shape_map)
        if result is None:
            if op_type not in unsupported:
                unsupported.append(op_type)
            known_inputs = [inp for inp in onnx_node.input if inp in init_values]
            warnings.append(
                f"Cannot statically convert {op_type} (→ {out_name}); "
                f"inputs={list(onnx_node.input)} static_inputs={known_inputs}"
            )
            _emit_fallback(nodes, node_id, out_name, onnx_node, meta)
            node_id += 1
            continue

        nodes.append({
            "id":     node_id,
            "name":   _safe(out_name),
            "op":     "call_function",
            "target": result["target"],
            "args":   result["args"],
            "kwargs": result.get("kwargs", {}),
            "meta":   meta,
        })
        node_id += 1

    # Output node
    output_refs = [_ref(out.name) for out in graph.output]
    nodes.append({
        "id": node_id, "name": "__output__", "op": "output",
        "target": "output", "args": [output_refs], "kwargs": {}, "meta": {},
    })

    return nodes, warnings, unsupported


def _emit_fallback(nodes, node_id, out_name, onnx_node, meta):
    """Emit an alias passthrough for an unsupported op so downstream refs still resolve."""
    if onnx_node.input:
        fallback_meta = dict(meta)
        fallback_meta["onnx_fallback_op"] = onnx_node.op_type
        fallback_meta["onnx_inputs"] = list(onnx_node.input)
        nodes.append({
            "id":     node_id,
            "name":   _safe(out_name),
            "op":     "call_function",
            "target": "aten.alias.default",
            "args":   [_ref(onnx_node.input[0])],
            "kwargs": {}, "meta": fallback_meta,
        })
    # If no inputs (e.g. a Constant op), skip — downstream will fail loudly at runtime.


def _collect_node_refs(value: Any, refs: set[str]) -> None:
    if isinstance(value, dict):
        node_ref = value.get("node_ref")
        if isinstance(node_ref, str):
            refs.add(node_ref)
        for v in value.values():
            _collect_node_refs(v, refs)
    elif isinstance(value, list):
        for v in value:
            _collect_node_refs(v, refs)


def _prune_unused_placeholders(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop initializer placeholders whose converted graph no longer references them.

    ONNX shape/index constants often become plain Python attrs in converters. Keeping
    their original placeholders in the manifest is both noisy and, when non-float32,
    can create dangling weight references.
    """
    refs: set[str] = set()
    for node in nodes:
        if node["op"] == "placeholder":
            continue
        _collect_node_refs(node.get("args", []), refs)
        _collect_node_refs(node.get("kwargs", {}), refs)

    return [
        node for node in nodes
        if node["op"] != "placeholder" or node.get("kind") == "user_input" or node["name"] in refs
    ]


def _prune_unreachable_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only nodes needed to produce the manifest output.

    ONNX exports often include shape-construction side graphs. Several converters
    consume those shapes as Python metadata, leaving the original shape nodes dead.
    Dropping unreachable nodes before weight packing prevents dead int64 constants
    from becoming required Kuma weights.
    """
    by_name = {node["name"]: node for node in nodes}
    reachable: set[str] = set()

    def visit_name(name: str) -> None:
        if name in reachable:
            return
        node = by_name.get(name)
        if node is None:
            return
        reachable.add(name)
        refs: set[str] = set()
        _collect_node_refs(node.get("args", []), refs)
        _collect_node_refs(node.get("kwargs", {}), refs)
        for ref in refs:
            visit_name(ref)

    for node in nodes:
        if node["op"] == "output":
            visit_name(node["name"])

    return [
        node for node in nodes
        if node["name"] in reachable or (node["op"] == "placeholder" and node.get("kind") == "user_input")
    ]


# ── Public entry point ────────────────────────────────────────────────────────

def export_via_onnx(
    model: nn.Module,
    example_inputs: tuple,
    *,
    opset_version: int = 17,
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> Package:
    """Export a PyTorch model to a kuma Package via ONNX as an intermediate step.

    Uses torch.onnx.export (classic exporter, opset 17 by default) then walks
    the ONNX graph, maps each op to the corresponding aten target in kuma-bart's
    opRegistry, and packs weight initializers into the standard f32 blob.
    """
    try:
        import onnx
        import onnx.shape_inference
        from onnx import numpy_helper
    except ImportError as e:
        raise ImportError(
            "The ONNX backend requires the `onnx` package.  "
            "Install it with: pip install onnx"
        ) from e

    model.eval()

    # ── Step 1: torch → ONNX ──────────────────────────────────────────────────
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
    buf.seek(0)
    onnx_model = onnx.load_model_from_string(buf.read())

    # ── Step 2: shape inference so we can populate meta.shape everywhere ──────
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    graph = onnx_model.graph

    # ── Step 3: collect initializers (weights + constant nodes) ──────────────
    init_values: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        init_values[init.name] = numpy_helper.to_array(init)
    # Constant nodes (shape tensors, bias constants, etc.) also become init_values
    for onnx_node in graph.node:
        if onnx_node.op_type == "Constant" and onnx_node.output:
            value = _constant_array_from_node(onnx_node)
            if value is not None:
                init_values[onnx_node.output[0]] = value

    # ── Step 4: build a shape/dtype map for all named tensors ────────────────
    shape_map: dict[str, dict] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        s = _get_shape(vi.type)
        if s is not None:
            shape_map[vi.name] = {"shape": s, "dtype": _get_dtype(vi.type)}

    _fold_static_onnx_values(graph, init_values, shape_map)

    # ── Step 5: real model inputs (exclude initializers from graph.input) ────
    init_names = set(init_values)
    input_specs = [vi for vi in graph.input if vi.name not in init_names]

    # ── Step 6: build kuma graph nodes ────────────────────────────────────────
    graph_nodes, warnings, unsupported = _build_graph(graph, input_specs, init_values, shape_map)

    graph_nodes = _prune_unused_placeholders(_prune_unreachable_nodes(graph_nodes))
    real_nodes = [n for n in graph_nodes if n["op"] == "call_function"]
    if not real_nodes:
        raise ValueError("ONNX backend produced an empty Kuma graph after pruning; graph output was not connected")
    reachable_fallbacks = sorted({
        n.get("meta", {}).get("onnx_fallback_op")
        for n in graph_nodes
        if n.get("meta", {}).get("onnx_fallback_op")
    })
    if reachable_fallbacks:
        fallback_details = [
            (
                f"{n['name']}:{n.get('meta', {}).get('onnx_fallback_op')}"
                + (
                    f" inputs={n.get('meta', {}).get('onnx_inputs')}"
                    if n.get("meta", {}).get("onnx_inputs") else ""
                )
            )
            for n in graph_nodes
            if n.get("meta", {}).get("onnx_fallback_op")
        ]
        raise ValueError(
            "ONNX backend cannot convert reachable op(s): "
            + ", ".join(reachable_fallbacks)
            + ". Refusing to write a passthrough-corrupted Kuma package. "
            + "Reachable fallback nodes: "
            + ", ".join(fallback_details[:8])
            + (f" ... ({len(fallback_details) - 8} more)" if len(fallback_details) > 8 else "")
            + ". Conversion warnings: "
            + " | ".join(warnings[-8:])
        )

    # ── Step 7: pack weights ──────────────────────────────────────────────────
    used_inits: set[str] = {
        n["weight_name"] for n in graph_nodes
        if n.get("kind") in ("parameter", "buffer") and "weight_name" in n
    }
    weights_blob, weight_entries, skipped = _pack_weights(init_values, used_inits)
    warnings += [f"skipped non-float32 initializer: {s}" for s in skipped]
    packed_names = {entry["name"] for entry in weight_entries}
    missing_packed = sorted(used_inits - packed_names)
    if missing_packed:
        raise ValueError(
            "ONNX backend cannot pack referenced non-float32 initializer(s): "
            + ", ".join(missing_packed[:10])
            + (f" ... ({len(missing_packed) - 10} more)" if len(missing_packed) > 10 else "")
        )

    # ── Step 8: manifest ──────────────────────────────────────────────────────
    inputs = []
    for spec in input_specs:
        info = shape_map.get(spec.name, {})
        entry: dict[str, Any] = {"name": spec.name, "kind": "user_input"}
        if info.get("shape"):
            entry["shape"] = info["shape"]
            entry["dtype"] = info.get("dtype", "float32")
        inputs.append(entry)

    outputs = []
    for out in graph.output:
        info = shape_map.get(out.name, {})
        entry = {"name": out.name}
        if info.get("shape"):
            entry["shape"] = info["shape"]
            entry["dtype"] = info.get("dtype", "float32")
        outputs.append(entry)

    op_counts: dict[str, int] = {}
    for n in graph_nodes:
        if n["op"] == "call_function":
            op_counts[n["target"]] = op_counts.get(n["target"], 0) + 1

    manifest: dict[str, Any] = {
        "format":         "kuma",
        "format_version": 0,
        "weight_file":    "weights.f32.bin",
        "endianness":     "little",
        "inputs":         inputs,
        "outputs":        outputs,
        "weights":        weight_entries,
        "graph": {
            "node_count": len(graph_nodes),
            "op_counts":  {k: op_counts[k] for k in sorted(op_counts)},
            "nodes":      graph_nodes,
        },
        "warnings":       warnings,
        "unsupported_ops": unsupported,
    }

    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    reference_values, reference_feeds = _run_onnx_reference_for_manifest_nodes(
        onnx_model, input_specs, example_inputs, graph_nodes
    )
    _concretize_manifest_shapes_from_reference(manifest, graph_nodes, reference_values, reference_feeds)

    # ── Step 9: debug report ──────────────────────────────────────────────────
    total_bytes = sum(w["byte_length"] for w in weight_entries)
    report_lines = [
        "# Kuma ONNX Backend Debug Report",
        "",
        "## Graph",
        f"- {len(graph_nodes)} nodes  ({len(weight_entries)} weight tensors, "
        f"{total_bytes / 1024 / 1024:.1f} MB)",
        f"- opset {opset_version}",
        "",
        "## Op counts",
    ]
    for target, count in sorted(op_counts.items()):
        report_lines.append(f"- {target}: {count}")
    if unsupported:
        report_lines += ["", "## Unsupported ONNX ops"]
        for op in unsupported:
            report_lines.append(f"- {op}")
    if warnings:
        report_lines += ["", "## Warnings"]
        for w in warnings:
            report_lines.append(f"- {w}")
    debug_report = "\n".join(report_lines) + "\n"
    golden = {
        "format_version": 0,
        "branches": [_capture_onnx_golden_from_reference(reference_values, reference_feeds)],
    }

    return Package(
        manifest=manifest,
        weights_blob=weights_blob,
        graph_data=manifest["graph"],
        debug_report=debug_report,
        kernels=load_kernels(),
        skipped=skipped,
        golden=golden,
    )


# ── Branching entry point ──────────────────────────────────────────────────────

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
    branch_wrappers       — one nn.Module per branch (must accept exactly 1 user input)
    branch_example_inputs — one example-inputs tuple per branch (used for ONNX tracing)
    opset_version         — ONNX opset for torch.onnx.export (default 17)
    """
    from kuma.branching import (  # reuse namespace utilities from the torch path
        _filter_and_repack_weights,
        _namespace_nodes,
        _namespace_ref,
        _namespace_weight_entries,
    )
    from kuma.golden import namespace_golden

    if not branch_wrappers:
        raise ValueError("compile_branching_onnx_kuma requires at least one branch wrapper")
    if len(branch_wrappers) != len(branch_example_inputs):
        raise ValueError("branch_wrappers and branch_example_inputs must have the same length")

    router_node_name = "router"
    switch_name = "switch_0"

    router_node = {
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
    for i in range(len(router_output_specs)):
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

    branches: list[dict[str, Any]] = []
    all_weight_entries: list[dict[str, Any]] = []
    blob_parts: list[bytes] = []
    all_warnings: list[str] = []
    all_unsupported: list[str] = []
    golden_branches: list[dict[str, Any]] = []
    running_offset = 0
    switch_output_shape: list[int] | None = None

    for i, (wrapper, example_inputs) in enumerate(zip(branch_wrappers, branch_example_inputs)):
        node_prefix   = f"branch{i}__"
        weight_prefix = f"branch{i}."

        pkg = export_via_onnx(wrapper, example_inputs, opset_version=opset_version)
        if pkg.golden is None or not pkg.golden.get("branches"):
            raise ValueError(f"branch {i}: ONNX export did not produce golden data")
        golden_branches.append(namespace_golden(pkg.golden["branches"][0], node_prefix))
        all_warnings.extend(f"branch {i}: {w}" for w in pkg.manifest.get("warnings", []))
        for op in pkg.manifest.get("unsupported_ops", []):
            if op not in all_unsupported:
                all_unsupported.append(op)

        raw_nodes     = pkg.graph_data["nodes"]
        weight_entries = pkg.manifest["weights"]
        weights_blob   = pkg.weights_blob

        output_node   = next(n for n in raw_nodes if n["op"] == "output")
        body_nodes    = [n for n in raw_nodes if n["op"] != "output"]

        used_weight_names = {
            n["weight_name"] for n in body_nodes
            if n["op"] == "placeholder" and "weight_name" in n
        }
        tight_blob, kept_entries = _filter_and_repack_weights(weights_blob, weight_entries, used_weight_names)

        pad = (-running_offset) % 4
        if pad:
            blob_parts.append(b"\x00" * pad)
            running_offset += pad

        all_weight_entries.extend(_namespace_weight_entries(kept_entries, weight_prefix, running_offset))
        blob_parts.append(tight_blob)
        running_offset += len(tight_blob)

        namespaced_body = _namespace_nodes(body_nodes, node_prefix, weight_prefix)

        branch_output_ref = _namespace_ref(output_node["args"][0][0], node_prefix)
        output_producer   = next(n for n in namespaced_body if n["name"] == branch_output_ref["node_ref"])
        if i == 0:
            switch_output_shape = output_producer.get("meta", {}).get("shape")

        user_inputs = [n for n in namespaced_body if n["op"] == "placeholder" and n.get("kind") == "user_input"]
        if len(user_inputs) != 1:
            raise ValueError(
                f"branch {i}: expected exactly 1 user_input placeholder (the routed scalar), "
                f"got {len(user_inputs)}: {[n['name'] for n in user_inputs]}"
            )

        branches.append({
            "nodes":  namespaced_body,
            "inputs": [{"node_ref": user_inputs[0]["name"]}],
            "output": branch_output_ref,
        })

    weights_blob_combined = b"".join(blob_parts)

    switch_node = {
        "id": 1_000_000,
        "name": switch_name,
        "op": "switch",
        "target": "switch",
        "args": [{"node_ref": getitem_names[branch_input_output_index]}],
        "kwargs": {},
        "meta": {"shape": switch_output_shape},
        "selector": {"node_ref": getitem_names[selector_output_index]},
        "branches": branches,
    }
    output_node_final = {
        "id": 1_000_001,
        "name": "output",
        "op": "output",
        "target": "output",
        "args": [[{"node_ref": switch_name}]],
        "kwargs": {},
        "meta": {},
    }

    all_nodes = [router_node, *getitem_nodes, switch_node, output_node_final]

    manifest: dict[str, Any] = {
        "format":         "kuma",
        "format_version": 0,
        "weight_file":    "weights.f32.bin",
        "endianness":     "little",
        "inputs":         [{"name": n, "kind": "user_input"} for n in router_input_names],
        "outputs":        [{"name": "output", "shape": switch_output_shape}],
        "weights":        all_weight_entries,
        "graph": {
            "node_count": len(all_nodes),
            "op_counts":  {},
            "nodes":      all_nodes,
        },
        "warnings":       all_warnings,
        "unsupported_ops": all_unsupported,
    }
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    debug_report = (
        "# Kuma ONNX-Kuma Branching Export Debug Report\n\n"
        f"- Branches: {len(branch_wrappers)}\n"
        f"- Router snippet: {router_snippet_name}\n"
        f"- Total weight bytes: {len(weights_blob_combined):,}\n"
        f"- ONNX opset: {opset_version}\n"
        + ("\n## Unsupported ONNX ops\n"
           + "\n".join(f"- {op}" for op in all_unsupported) + "\n" if all_unsupported else "")
        + ("\n## Warnings\n"
           + "\n".join(f"- {w}" for w in all_warnings) + "\n" if all_warnings else "")
    )

    return Package(
        manifest=manifest,
        weights_blob=weights_blob_combined,
        graph_data={"format_version": 0, "nodes": all_nodes},
        debug_report=debug_report,
        kernels=load_kernels(),
        snippets={router_snippet_name: router_snippet_source.encode("utf-8")},
        golden={"format_version": 0, "branches": golden_branches},
    )
