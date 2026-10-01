"""Regression tests for ONNX -> Kuma converter correctness (kuma.onnx_ops) and the
ONNX backend's graph-walking helpers, plus GELU variant plumbing."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

import kuma
from kuma import onnx_backend
from kuma.export import export_program
from kuma.compiler import compile as kuma_compile
from kuma.kernels import load_kernels
from kuma.onnx_ops import CONVERTERS


def _convert(node, init_values=None, shape_map=None):
    return CONVERTERS[node.op_type](node, init_values or {}, shape_map or {})


_W = {"w": np.zeros((8, 16), dtype=np.float32), "b": np.zeros((8,), dtype=np.float32)}


# ── Bug: ConvTranspose must not emit convolution(transposed=True) ─────────────

def test_conv_transpose_refused():
    node = helper.make_node("ConvTranspose", ["x", "w"], ["y"], strides=[2, 2])
    assert _convert(node) is None


def test_conv_transpose_model_fails_compile_loudly():
    model = nn.Sequential(nn.ConvTranspose2d(4, 2, 2, stride=2))
    with pytest.raises(ValueError, match="ConvTranspose"):
        kuma.export_via_onnx(model, (torch.randn(1, 4, 4, 4),))


# ── Bug: MatMul only with a 2D initializer rhs (pre-transposed to [N, K]) ─────

def test_matmul_dynamic_rhs_refused():
    assert _convert(helper.make_node("MatMul", ["x", "y"], ["z"])) is None


def test_matmul_initializer_rhs_uses_transposed_copy():
    out = _convert(helper.make_node("MatMul", ["x", "w"], ["z"]), {"w": np.zeros((16, 8), np.float32)})
    assert out["target"] == "aten.mm.default"
    assert out["args"][1] == {"node_ref": "w__kuma_transposed"}


def test_matmul_dynamic_rhs_model_fails_compile_loudly():
    class M(nn.Module):
        def forward(self, x, y):
            return x @ y

    with pytest.raises(ValueError, match="MatMul"):
        kuma.export_via_onnx(M(), (torch.randn(4, 4), torch.randn(4, 4)))


# ── Bug: Gather axis != 0 with a tensor index is not aten.gather ──────────────

def test_gather_tensor_index_nonzero_axis_refused():
    node = helper.make_node("Gather", ["x", "idx"], ["y"], axis=1)
    assert _convert(node, {"idx": np.array([0, 2], dtype=np.int64)}) is None


def test_gather_scalar_index_and_axis0_still_convert():
    sel = _convert(helper.make_node("Gather", ["x", "i"], ["y"], axis=1), {"i": np.array(1, dtype=np.int64)})
    assert sel["target"] == "aten.select.int"
    idx = _convert(helper.make_node("Gather", ["x", "idx"], ["y"], axis=0))
    assert idx["target"] == "aten.index.Tensor"


# ── Bug: Gemm transA/transB/alpha/beta ────────────────────────────────────────

def test_gemm_linear_form_converts():
    out = _convert(helper.make_node("Gemm", ["x", "w", "b"], ["y"], transB=1), _W)
    assert out["target"] == "aten.linear.default"


@pytest.mark.parametrize(
    "attrs",
    [{"transB": 0}, {"transB": 1, "transA": 1}, {"transB": 1, "alpha": 2.0}, {"transB": 1, "beta": 0.5}],
)
def test_gemm_other_forms_refused(attrs):
    assert _convert(helper.make_node("Gemm", ["x", "w", "b"], ["y"], **attrs), _W) is None


def test_gemm_non_initializer_weight_refused():
    assert _convert(helper.make_node("Gemm", ["x", "dyn", "b"], ["y"], transB=1), _W) is None


# ── Bug: Conv padding must be symmetric and auto_pad NOTSET ───────────────────

def test_conv_symmetric_pads_convert():
    out = _convert(helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 2, 1, 2]))
    assert out["args"][4] == [1, 2]


def test_conv_asymmetric_pads_refused():
    assert _convert(helper.make_node("Conv", ["x", "w"], ["y"], pads=[0, 0, 1, 1])) is None


def test_conv_auto_pad_refused():
    assert _convert(helper.make_node("Conv", ["x", "w"], ["y"], auto_pad="SAME_UPPER")) is None


# ── Bug: unknown dims are -1, not None ────────────────────────────────────────

def test_shape_array_for_rejects_unknown_dims():
    assert onnx_backend._shape_array_for("x", {"x": {"shape": [-1, 3]}}) is None
    np.testing.assert_array_equal(onnx_backend._shape_array_for("x", {"x": {"shape": [2, 3]}}), [2, 3])


def test_flatten_rejects_unknown_dims():
    node = helper.make_node("Flatten", ["x"], ["y"], axis=1)
    assert _convert(node, shape_map={"x": {"shape": [-1, 3, 4]}}) is None
    out = _convert(node, shape_map={"x": {"shape": [2, 3, 4]}})
    assert out["args"][1] == [2, 12]


# ── Bug: folded constants must not duplicate node names ───────────────────────

def _graph(nodes, inputs, outputs, initializers=()):
    return helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))


def test_folded_constant_does_not_duplicate_node_name():
    c = helper.make_node("Constant", [], ["c"], value=numpy_helper.from_array(np.array([1, 2], dtype=np.int64)))
    cast = helper.make_node("Cast", ["c"], ["c_f"], to=TensorProto.FLOAT)
    add = helper.make_node("Add", ["x", "c_f"], ["y"])
    graph = _graph(
        [c, cast, add],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    init_values = onnx_backend._collect_static_values(graph)
    shape_map = onnx_backend._build_shape_map(graph)
    onnx_backend._fold_static_onnx_values(graph, init_values, shape_map)
    assert "c_f" in init_values
    nodes, _, _ = onnx_backend._build_graph(graph, list(graph.input), init_values, shape_map)
    names = [n["name"] for n in nodes]
    assert len(names) == len(set(names)), names
    assert next(n for n in nodes if n["name"] == "c_f")["op"] == "placeholder"


# ── Bug: safe-name collisions ─────────────────────────────────────────────────

def test_safe_name_collision_raises():
    graph = _graph(
        [helper.make_node("Relu", ["x"], ["a/b"]), helper.make_node("Relu", ["a/b"], ["a.b"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("a.b", TensorProto.FLOAT, [2])],
    )
    with pytest.raises(ValueError, match="collide"):
        onnx_backend._check_safe_name_collisions(graph, {})


# ── Bug: deterministic golden.json ────────────────────────────────────────────

class _Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        y = torch.relu(self.conv(x))
        return torch.sigmoid(y) * y + 1.0


def test_onnx_golden_node_order_is_sorted_and_stable(tmp_path):
    x = torch.randn(1, 3, 6, 6)
    a = kuma.export_via_onnx(_Chain(), (x,))
    keys = list(a.golden["branches"][0]["nodes"])
    assert keys == sorted(keys) and keys
    path = a.save(tmp_path / "a.iph")
    with zipfile.ZipFile(path) as zf:
        text = zf.read("golden.json").decode()
    assert text == json.dumps(json.loads(text), sort_keys=True)


# ── export_model backend validation ───────────────────────────────────────────

def test_export_model_unknown_backend_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        kuma.export_model(_Chain(), (torch.randn(1, 3, 6, 6),), tmp_path / "x.iph", backend="tflite")


def test_export_via_onnx_validates_inputs():
    with pytest.raises(ValueError, match="float32"):
        kuma.export_via_onnx(_Chain(), (torch.randn(1, 3, 6, 6, dtype=torch.float64),))


# ── GELU variants ─────────────────────────────────────────────────────────────

def test_gelu_erf_kernel_bundled():
    kernels = load_kernels()
    assert "gelu_erf.wgsl" in kernels and "gelu.wgsl" in kernels


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_torch_gelu_approximate_preserved_in_manifest(approximate):
    model = nn.Sequential(nn.GELU(approximate=approximate))
    pkg = kuma_compile(export_program(model, (torch.randn(2, 3),)))
    (gelu,) = [n for n in pkg.manifest["graph"]["nodes"] if n["target"] == "aten.gelu.default"]
    # torch.export only records the kwarg when it differs from the default ("none").
    assert gelu["kwargs"].get("approximate", "none") == approximate


@pytest.mark.parametrize("approximate,expected", [("none", {}), ("tanh", {"approximate": "tanh"})])
def test_onnx_gelu_converter_preserves_approximate(approximate, expected):
    out = _convert(helper.make_node("Gelu", ["x"], ["y"], approximate=approximate))
    assert out["target"] == "aten.gelu.default"
    assert out["kwargs"] == expected
