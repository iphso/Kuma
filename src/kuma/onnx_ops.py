"""ONNX op -> Kuma manifest node converters (used by kuma.onnx_backend).

Each converter(node, init_values, shape_map) returns either:
  {"target": str, "args": list, "kwargs": dict}   — success
  None                                              — can't convert faithfully

A None result becomes an aliased fallback node in onnx_backend._build_graph; if that
fallback is reachable from the graph output, export_via_onnx raises (loud compile
error) rather than writing a silently-wrong package. So a converter must return None
for every attribute combination whose emitted aten target would NOT compute the same
thing in kuma-bart -- "close enough" conversions are bugs.

`init_values` maps tensor name -> statically-known numpy value (initializers, Constant
nodes, and folded shape/index subgraphs); `shape_map` maps tensor name ->
{"shape": [...], "dtype": str} from ONNX shape inference, with unknown dims as -1.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
from onnx import AttributeProto, numpy_helper

# ── Attribute / value helpers ─────────────────────────────────────────────────


def attr(node, name: str, default=None):
    """Read a named attribute from an ONNX NodeProto."""
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == AttributeProto.FLOAT:
            return a.f
        if a.type == AttributeProto.INT:
            return int(a.i)
        if a.type == AttributeProto.STRING:
            return a.s
        if a.type == AttributeProto.FLOATS:
            return list(a.floats)
        if a.type == AttributeProto.INTS:
            return [int(v) for v in a.ints]
        if a.type == AttributeProto.TENSOR:
            return numpy_helper.to_array(a.t)
    return default


def text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def init_list(init_values: dict[str, np.ndarray], name: str | None) -> list[int] | None:
    if not name or name not in init_values:
        return None
    return [int(v) for v in init_values[name].reshape(-1)]


def init_scalar(init_values: dict[str, np.ndarray], name: str | None) -> int | float | None:
    if not name or name not in init_values:
        return None
    arr = init_values[name].reshape(-1)
    if arr.size != 1:
        return None
    value = arr[0]
    return int(value) if np.issubdtype(arr.dtype, np.integer) else float(value)


# ── Name sanitisation ─────────────────────────────────────────────────────────


def safe_name(name: str) -> str:
    """Convert an ONNX tensor name (may contain /.:) to a safe node identifier.

    Not injective ("a/b" and "a.b" both map to "a_b") -- onnx_backend checks every
    tensor name in a graph for collisions up front and raises rather than letting two
    tensors silently share one manifest node name."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
    return s or "tensor"


def ref(name: str) -> dict[str, str]:
    return {"node_ref": safe_name(name)}


def ref_or_scalar(init_values: dict[str, np.ndarray], name: str) -> dict[str, str] | int | float:
    value = init_scalar(init_values, name)
    if value is not None:
        return value
    return ref(name)


def matmul_transposed_weight_name(name: str) -> str:
    """Name of the synthesized [N, K] transpose of a 2D MatMul rhs initializer [K, N]."""
    return f"{name}__kuma_transposed"


def matmul_rhs_is_2d_initializer(node, init_values: dict[str, np.ndarray]) -> bool:
    if node.op_type != "MatMul" or len(node.input) < 2:
        return False
    arr = init_values.get(node.input[1])
    return arr is not None and arr.ndim == 2


# ── Converter registry ────────────────────────────────────────────────────────

CONVERTERS: dict[str, Callable] = {}


def _op(*op_types: str):
    def decorator(fn: Callable) -> Callable:
        for t in op_types:
            CONVERTERS[t] = fn
        return fn
    return decorator


def _unary(op_type: str, target: str) -> None:
    def convert(node, init_values, shape_map):
        return {"target": target, "args": [ref(node.input[0])], "kwargs": {}}
    convert.__name__ = f"_convert_{op_type.lower()}"
    CONVERTERS[op_type] = convert


for _onnx_op, _target in (
    ("Relu", "aten.relu.default"),
    ("Sigmoid", "aten.sigmoid.default"),
    ("Tanh", "aten.tanh.default"),
    ("Silu", "aten.silu.default"),
    ("Sqrt", "aten.sqrt.default"),
    ("Cos", "aten.cos.default"),
    ("Sin", "aten.sin.default"),
    ("Floor", "aten.floor.default"),
    ("Erf", "aten.erf.default"),
):
    _unary(_onnx_op, _target)


def _symmetric_2d_padding(node) -> list[int] | None:
    """[pad_h, pad_w] for a 2D conv, or None when the padding can't be expressed as
    aten.convolution's symmetric `padding` (auto_pad set, or begin != end pads)."""
    auto_pad = text(attr(node, "auto_pad", b"NOTSET"))
    if auto_pad != "NOTSET":
        return None
    pads = attr(node, "pads", [0, 0, 0, 0])
    half = len(pads) // 2
    begin, end = pads[:half], pads[half:]
    if begin != end:
        return None
    return begin


@_op("Conv")
def _conv(node, init_values, shape_map):
    inp = list(node.input)
    padding = _symmetric_2d_padding(node)
    if padding is None:
        return None
    strides = attr(node, "strides", [1, 1])
    dilations = attr(node, "dilations", [1, 1])
    groups = attr(node, "group", 1)
    bias = ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.convolution.default",
        "args": [ref(inp[0]), ref(inp[1]), bias, strides, padding, dilations, False, [0, 0], groups],
        "kwargs": {},
    }


@_op("ConvTranspose")
def _conv_transpose(node, init_values, shape_map):
    # kuma-bart's conv2d handler (kuma-bart/src/ops/conv2d.ts) rejects
    # aten.convolution with transposed=True -- conv_transpose2d.wgsl ships in the
    # package but nothing dispatches it. Emitting it would only defer the failure to
    # model-load time in the browser, so refuse here and fail the compile loudly.
    return None


@_op("Gelu")
def _gelu(node, init_values, shape_map):
    # Opset 20+ only (torch's opset-17 export decomposes GELU into Erf/Mul/Add). Mirror
    # torch.export: the `approximate` kwarg is present only for the tanh variant; its
    # absence means exact erf GELU (kuma-bart picks gelu_erf.wgsl vs gelu.wgsl on it).
    approximate = text(attr(node, "approximate", b"none"))
    kwargs = {"approximate": "tanh"} if approximate == "tanh" else {}
    return {"target": "aten.gelu.default", "args": [ref(node.input[0])], "kwargs": kwargs}


def _binary(target: str):
    def convert(node, init_values, shape_map):
        return {
            "target": target,
            "args": [ref_or_scalar(init_values, node.input[0]), ref_or_scalar(init_values, node.input[1])],
            "kwargs": {},
        }
    return convert


CONVERTERS["Add"] = _binary("aten.add.Tensor")
CONVERTERS["Sub"] = _binary("aten.sub.Tensor")
CONVERTERS["Mul"] = _binary("aten.mul.Tensor")
CONVERTERS["Div"] = _binary("aten.div.Tensor")


@_op("Pow")
def _pow(node, init_values, shape_map):
    exp_name = node.input[1]
    if exp_name in init_values:
        scalar = float(init_values[exp_name].flat[0])
        return {"target": "aten.pow.Tensor_Scalar", "args": [ref(node.input[0]), scalar], "kwargs": {}}
    return None  # dynamic exponent — unsupported


@_op("Gemm")
def _gemm(node, init_values, shape_map):
    # kuma-bart's aten.linear computes x @ W^T + bias with W:[N, K], bias:[N]. That is
    # exactly ONNX Gemm with transA=0, transB=1, alpha=1, beta=1 (nn.Linear's export)
    # and a per-output-feature bias. Anything else would need extra ops -- refuse.
    inp = list(node.input)
    if attr(node, "transA", 0) != 0 or attr(node, "transB", 0) != 1:
        return None
    if attr(node, "alpha", 1.0) != 1.0 or attr(node, "beta", 1.0) != 1.0:
        return None
    weight = init_values.get(inp[1])
    if weight is None or weight.ndim != 2:
        return None
    bias = None
    if len(inp) > 2 and inp[2]:
        c = init_values.get(inp[2])
        # C may broadcast as [M, N] / [1, N] / scalar in ONNX; linear's bias is exactly [N].
        if c is None or c.size != weight.shape[0] or (c.ndim > 1 and c.shape[-1] != weight.shape[0]):
            return None
        bias = ref(inp[2])
    return {"target": "aten.linear.default", "args": [ref(inp[0]), ref(inp[1]), bias], "kwargs": {}}


@_op("MatMul")
def _matmul(node, init_values, shape_map):
    # kuma-bart's aten.mm handler (kuma-bart/src/ops/linear.ts) computes
    # x(..., K) @ weight(N, K)^T -- i.e. it expects its rhs in nn.Linear's [N, K]
    # layout, not ONNX MatMul's [K, N]. For a 2D initializer rhs, onnx_backend
    # synthesizes a pre-transposed [N, K] copy and we bind that. A dynamic (or >2D)
    # rhs can't be pre-transposed, and binding it directly would compute x @ rhs^T
    # (silently wrong when K == N), so refuse.
    if not matmul_rhs_is_2d_initializer(node, init_values):
        return None
    rhs = matmul_transposed_weight_name(node.input[1])
    return {"target": "aten.mm.default", "args": [ref(node.input[0]), ref(rhs)], "kwargs": {}}


@_op("LayerNormalization")
def _layernorm(node, init_values, shape_map):
    inp = list(node.input)
    eps = attr(node, "epsilon", 1e-5)
    weight_name = inp[1] if len(inp) > 1 and inp[1] else None
    # normalized_shape comes from the scale initializer's shape
    normalized_shape = list(init_values[weight_name].shape) if weight_name and weight_name in init_values else []
    bias = ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.layer_norm.default",
        "args": [ref(inp[0]), normalized_shape, ref(weight_name) if weight_name else None, bias, eps],
        "kwargs": {},
    }


@_op("GroupNormalization")
def _groupnorm(node, init_values, shape_map):
    inp = list(node.input)
    num_groups = attr(node, "num_groups", 1)
    eps = attr(node, "epsilon", 1e-5)
    weight = ref(inp[1]) if len(inp) > 1 and inp[1] else None
    bias = ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.group_norm.default",
        "args": [ref(inp[0]), num_groups, weight, bias, eps],
        "kwargs": {},
    }


@_op("InstanceNormalization")
def _instancenorm(node, init_values, shape_map):
    inp = list(node.input)
    in_shape = shape_map.get(inp[0], {}).get("shape", [])
    if len(in_shape) < 2 or in_shape[1] is None or in_shape[1] < 0:
        return None
    channels = int(in_shape[1])
    eps = attr(node, "epsilon", 1e-5)
    weight = ref(inp[1]) if len(inp) > 1 and inp[1] else None
    bias = ref(inp[2]) if len(inp) > 2 and inp[2] else None
    return {
        "target": "aten.group_norm.default",
        "args": [ref(inp[0]), channels, weight, bias, eps],
        "kwargs": {},
    }


@_op("Reshape")
def _reshape(node, init_values, shape_map):
    inp = list(node.input)
    shape_src = inp[1] if len(inp) > 1 else None
    if shape_src and shape_src in init_values:
        shape = [int(v) for v in init_values[shape_src].flat]
        return {"target": "aten.reshape.default", "args": [ref(inp[0]), shape], "kwargs": {}}
    out_shape = shape_map.get(node.output[0], {}).get("shape") if node.output else None
    # Unknown dims are -1 in shape_map; more than one can't be a valid reshape target.
    if out_shape and sum(1 for d in out_shape if d < 0) <= 1:
        return {"target": "aten.reshape.default", "args": [ref(inp[0]), out_shape], "kwargs": {}}
    # Dynamic shape — not statically representable
    return None


@_op("Flatten")
def _flatten(node, init_values, shape_map):
    axis = attr(node, "axis", 1)
    in_shape = shape_map.get(node.input[0], {}).get("shape")
    # Unknown dims are encoded as -1 (see onnx_backend._get_shape): np.prod over them
    # would produce a bogus (negative) size, so require a fully static input shape.
    if not in_shape or any(d is None or d < 0 for d in in_shape):
        return None
    if axis < 0:
        axis += len(in_shape)
    out_shape = [int(np.prod(in_shape[:axis], dtype=np.int64)), int(np.prod(in_shape[axis:], dtype=np.int64))]
    return {"target": "aten.reshape.default", "args": [ref(node.input[0]), out_shape], "kwargs": {}}


@_op("Transpose")
def _transpose(node, init_values, shape_map):
    perm = attr(node, "perm", None)
    if perm is None:
        return None
    return {"target": "aten.permute.default", "args": [ref(node.input[0]), perm], "kwargs": {}}


@_op("Concat")
def _concat(node, init_values, shape_map):
    axis = attr(node, "axis", 0)
    return {"target": "aten.cat.default", "args": [[ref(i) for i in node.input], axis], "kwargs": {}}


@_op("Einsum")
def _einsum(node, init_values, shape_map):
    equation = text(attr(node, "equation", ""))
    if not equation:
        return None
    return {"target": "aten.einsum.default", "args": [equation, [ref(i) for i in node.input]], "kwargs": {}}


@_op("Expand")
def _expand(node, init_values, shape_map):
    inp = list(node.input)
    size = init_list(init_values, inp[1] if len(inp) > 1 else None)
    if size is None:
        size = shape_map.get(node.output[0], {}).get("shape") if node.output else None
    if not size:
        return None
    return {"target": "aten.expand.default", "args": [ref(inp[0]), size], "kwargs": {}}


@_op("ReduceSum")
def _reducesum(node, init_values, shape_map):
    inp = list(node.input)
    axes = init_list(init_values, inp[1] if len(inp) > 1 else None)
    if axes is None:
        axes = attr(node, "axes", None)
    if axes is None:
        in_shape = shape_map.get(inp[0], {}).get("shape")
        if not in_shape:
            return None
        axes = list(range(len(in_shape)))
    return {"target": "aten.sum.dim_IntList", "args": [ref(inp[0]), [int(a) for a in axes]], "kwargs": {}}


@_op("Gather")
def _gather(node, init_values, shape_map):
    inp = list(node.input)
    axis = int(attr(node, "axis", 0))
    index_scalar = init_scalar(init_values, inp[1] if len(inp) > 1 else None)
    if isinstance(index_scalar, int):
        return {"target": "aten.select.int", "args": [ref(inp[0]), axis, index_scalar], "kwargs": {}}
    if axis == 0:
        return {"target": "aten.index.Tensor", "args": [ref(inp[0]), [ref(inp[1])]], "kwargs": {}}
    # ONNX Gather along axis != 0 with a tensor index is index_select-style (output
    # rank = data rank - 1 + index rank). aten.gather has GatherElements semantics
    # (index same rank as data, elementwise lookup) -- not the same op. No kuma-bart
    # target matches, so refuse.
    return None


@_op("Slice")
def _slice(node, init_values, shape_map):
    inp = list(node.input)
    starts = init_list(init_values, inp[1] if len(inp) > 1 else None)
    ends = init_list(init_values, inp[2] if len(inp) > 2 else None)
    axes = init_list(init_values, inp[3] if len(inp) > 3 else None)
    steps = init_list(init_values, inp[4] if len(inp) > 4 else None)
    starts = starts if starts is not None else attr(node, "starts", None)
    ends = ends if ends is not None else attr(node, "ends", None)
    axes = axes if axes is not None else attr(node, "axes", None)
    steps = steps if steps is not None else attr(node, "steps", None)
    if starts is None or ends is None:
        return None
    axes = axes if axes is not None else list(range(len(starts)))
    steps = steps if steps is not None else [1] * len(starts)
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        return None
    if len(starts) != 1:
        return {
            "target": "aten.slice_multi.default",
            "args": [ref(inp[0]), [int(v) for v in axes], [int(v) for v in starts], [int(v) for v in ends], [int(v) for v in steps]],
            "kwargs": {},
        }
    return {
        "target": "aten.slice.Tensor",
        "args": [ref(inp[0]), int(axes[0]), int(starts[0]), int(ends[0]), int(steps[0])],
        "kwargs": {},
    }


@_op("DepthToSpace")
def _depth_to_space(node, init_values, shape_map):
    blocksize = int(attr(node, "blocksize", 0))
    mode = text(attr(node, "mode", "DCR"))
    if blocksize <= 0:
        return None
    # PyTorch PixelShuffle exports as CRD. DCR has a different channel ordering.
    if mode.upper() != "CRD":
        return None
    return {"target": "aten.pixel_shuffle.default", "args": [ref(node.input[0]), blocksize], "kwargs": {}}


@_op("Clip")
def _clip(node, init_values, shape_map):
    inp = list(node.input)

    def _scalar(name):
        if name and name in init_values:
            return float(init_values[name].flat[0])
        return None

    min_val = _scalar(inp[1]) if len(inp) > 1 else attr(node, "min", None)
    max_val = _scalar(inp[2]) if len(inp) > 2 else attr(node, "max", None)
    return {"target": "aten.clamp.default", "args": [ref(inp[0]), min_val, max_val], "kwargs": {}}


@_op("Identity", "Dropout")
def _identity(node, init_values, shape_map):
    # Dropout is inference-only here — treated as identity
    return {"target": "aten.alias.default", "args": [ref(node.input[0])], "kwargs": {}}


@_op("Cast")
def _cast(node, init_values, shape_map):
    # Kuma stores runtime tensors as f32 buffers. For model data this is an identity
    # in the observed ONNX graphs; for index tensors, kernels read f32-encoded ints.
    # Constant casts are folded into init_values before conversion, so shape/index
    # side casts are normally pruned away entirely.
    return {"target": "aten.alias.default", "args": [ref(node.input[0])], "kwargs": {}}


@_op("Squeeze", "Unsqueeze")
def _squeeze_unsqueeze(node, init_values, shape_map):
    # Shape change only — passthrough in kuma (meta.shape carries the updated shape)
    return {"target": "aten.alias.default", "args": [ref(node.input[0])], "kwargs": {}}
