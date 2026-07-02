"""Smoke tests for the ONNX compilation backend.

Each test runs the same model through both the direct torch.export path and the
ONNX-intermediary path and asserts that the Package contract holds either way:
correct .iph zip, consistent weights blob, valid manifest schema.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import (
    assert_manifest_schema,
    assert_weight_sizes_match,
    run_pipeline,
)

BACKENDS = ["torch", "onnx"]


class _Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x)


class _GroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.norm = nn.GroupNorm(2, 8)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _OnnxMatMulRhs(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 4))

    def forward(self, x):
        return x @ self.weight


@pytest.mark.parametrize("backend", BACKENDS)
def test_conv_relu(tmp_path, backend):
    r = run_pipeline(_Conv(), (torch.randn(1, 3, 16, 16),), tmp_path / backend, backend=backend)
    assert_manifest_schema(r["manifest"])
    assert_weight_sizes_match(r["weight_entries"], r["blob"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_gelu(tmp_path, backend):
    r = run_pipeline(_Residual(), (torch.randn(1, 8, 16, 16),), tmp_path / backend, backend=backend)
    assert_manifest_schema(r["manifest"])
    assert_weight_sizes_match(r["weight_entries"], r["blob"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear(tmp_path, backend):
    r = run_pipeline(_Linear(), (torch.randn(4, 16),), tmp_path / backend, backend=backend)
    assert_manifest_schema(r["manifest"])
    assert_weight_sizes_match(r["weight_entries"], r["blob"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_groupnorm(tmp_path, backend):
    r = run_pipeline(_GroupNorm(), (torch.randn(1, 8, 16, 16),), tmp_path / backend, backend=backend)
    assert_manifest_schema(r["manifest"])
    assert_weight_sizes_match(r["weight_entries"], r["blob"])


def test_onnx_matmul_rhs_initializer_transposed_for_linear_kernel(tmp_path):
    r = run_pipeline(_OnnxMatMulRhs(), (torch.randn(1, 270, 480, 2),), tmp_path / "onnx_matmul", backend="onnx")
    assert_manifest_schema(r["manifest"])
    assert_weight_sizes_match(r["weight_entries"], r["blob"])

    matmul_nodes = [
        n for n in r["manifest"]["graph"]["nodes"]
        if n.get("target") == "aten.mm.default"
    ]
    assert matmul_nodes
    weight_ref = matmul_nodes[0]["args"][1]["node_ref"]
    weight_node = next(n for n in r["manifest"]["graph"]["nodes"] if n["name"] == weight_ref)
    assert weight_node["meta"]["shape"] == [4, 2]
