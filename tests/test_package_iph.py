"""Tests for the .iph package contract and the top-level kuma public API."""

from __future__ import annotations

import json
import zipfile

import torch
import torch.nn as nn

import kuma
from kuma.kernels import _KERNEL_NAMES


class _Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


_EXAMPLE_INPUT = (torch.randn(1, 3, 32, 32),)

_REQUIRED_KERNELS = set(_KERNEL_NAMES)


def _zip_names(path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def test_export_model_writes_iph(tmp_path):
    out = tmp_path / "model.iph"
    result_path = kuma.export_model(_Simple(), _EXAMPLE_INPUT, out=out)
    assert result_path == out
    assert out.exists()


def test_iph_contains_required_files(tmp_path):
    out = tmp_path / "model.iph"
    kuma.export_model(_Simple(), _EXAMPLE_INPUT, out=out)
    names = _zip_names(out)
    assert "manifest.json" in names
    assert "weights.f32.bin" in names
    assert "debug_report.md" in names


def test_iph_contains_all_kernels(tmp_path):
    out = tmp_path / "model.iph"
    kuma.export_model(_Simple(), _EXAMPLE_INPUT, out=out)
    names = _zip_names(out)
    kernel_names = {n[len("kernels/"):] for n in names if n.startswith("kernels/")}
    assert kernel_names == _REQUIRED_KERNELS


def test_iph_does_not_contain_exported_graph_json(tmp_path):
    """exported_graph.json is a write_dir()-only debug artifact, not part of the .iph contract."""
    out = tmp_path / "model.iph"
    kuma.export_model(_Simple(), _EXAMPLE_INPUT, out=out)
    assert "exported_graph.json" not in _zip_names(out)


def test_iph_manifest_matches_package_manifest(tmp_path):
    model = _Simple()
    model.eval()
    with torch.no_grad():
        ep = torch.export.export(model, _EXAMPLE_INPUT)
    package = kuma.compile(ep)

    out = tmp_path / "model.iph"
    package.save(out)

    with zipfile.ZipFile(out) as zf:
        zipped_manifest = json.loads(zf.read("manifest.json"))
    assert zipped_manifest == package.manifest


def test_export_exported_program(tmp_path):
    model = _Simple()
    model.eval()
    with torch.no_grad():
        ep = torch.export.export(model, _EXAMPLE_INPUT)

    out = tmp_path / "model.iph"
    kuma.export_exported_program(ep, out=out)
    names = _zip_names(out)
    assert "manifest.json" in names
    assert "weights.f32.bin" in names


def test_compile_then_write_dir(tmp_path):
    model = _Simple()
    model.eval()
    with torch.no_grad():
        ep = torch.export.export(model, _EXAMPLE_INPUT)

    package = kuma.compile(ep)
    out_dir = tmp_path / "debug"
    package.write_dir(out_dir)

    for fname in ["exported_graph.json", "weights.f32.bin", "manifest.json", "debug_report.md"]:
        assert (out_dir / fname).exists()
    for kernel_name in _REQUIRED_KERNELS:
        assert (out_dir / "kernels" / kernel_name).exists()
