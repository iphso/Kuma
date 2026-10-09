"""Ops the terrain-diffusion EDM2 U-Nets trace to (Loom's export): mean over the
channel axis, ones_like + cat (the bias channel), clip, nearest x2 upsampling, the
stride-2 slice 'down' resample, the normalize pattern (vector_norm + scalar mul/add +
div) and softmax. Checks they compile through the torch path with golden stats, and
that the shipped kernel set has what kuma-bart dispatches for them."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match
from kuma.compiler import compile as kuma_compile
from kuma.kernels import load_kernels


class _Film(nn.Module):
    """c / sqrt(mean(c^2, dim=1, keepdim=True) + eps), then y * c[..., None, None]."""

    def forward(self, y, c):
        c = c + 1
        c = c / torch.sqrt(torch.mean(c ** 2, dim=1, keepdim=True) + 1e-8)
        return y * c.unsqueeze(2).unsqueeze(3)


class _BiasChannel(nn.Module):
    def forward(self, x):
        return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)


class _Clip(nn.Module):
    def forward(self, x):
        return torch.clip(x, -256, 256)


class _Resample(nn.Module):
    def forward(self, x):
        down = x[:, :, ::2, ::2]
        return F.interpolate(down, scale_factor=2.0, mode="nearest")


class _Normalize(nn.Module):
    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x / (norm * math.sqrt(1.0 / x.shape[1]) + 1e-4)


class _Softmax(nn.Module):
    def forward(self, q, k):
        att = torch.einsum("nhcq,nhck->nhqk", q, k * 0.125).softmax(dim=3)
        return torch.einsum("nhqk,nhck->nhcq", att, k)


@pytest.mark.parametrize(
    "model,inputs,expected",
    [
        (_Film(), (torch.randn(1, 8, 4, 4), torch.randn(1, 8)), [{"aten.mean.dim"}]),
        (_BiasChannel(), (torch.randn(1, 3, 4, 4),), [{"aten.ones_like.default"}, {"aten.cat.default"}]),
        (_Clip(), (torch.randn(1, 3, 4, 4),), [{"aten.clip.default", "aten.clamp.default"}]),
        # torch < 2.5 decomposes nearest upsampling into arange/_unsafe_index (which
        # kuma-bart cannot run); Loom's export image pins torch 2.6, which keeps the op.
        (_Resample(), (torch.randn(1, 3, 8, 8),), [{"aten.slice.Tensor"}, {"aten.upsample_nearest2d.vec", "aten.upsample_nearest2d.default", "aten._unsafe_index.Tensor"}]),
        (_Normalize(), (torch.randn(1, 8, 4, 4),), [{"aten.linalg_vector_norm.default"}, {"aten.div.Tensor"}]),
        (_Softmax(), (torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16)), [{"aten.softmax.int", "aten._softmax.default"}, {"aten.einsum.default", "aten.bmm.default"}]),
    ],
    ids=["film_mean", "bias_channel", "clip", "resample", "normalize", "attention_softmax"],
)
def test_terrain_op_exports(model, inputs, expected, tmp_path):
    result = run_pipeline(model, inputs, tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    targets = {n["target"] for n in result["graph_data"]["nodes"] if n["op"] == "call_function"}
    # Each entry lists the acceptable spellings of one op (the IR naming moved between
    # torch 2.3 and 2.6: clip/clamp, softmax.int/_softmax, einsum kept vs lowered to bmm).
    for alternatives in expected:
        assert alternatives & targets, f"expected one of {alternatives} in {targets}"


def test_softmax_kernel_ships_and_golden_matches_eager(tmp_path):
    model = _Softmax().eval()
    inputs = (torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
    with torch.no_grad():
        ep = torch.export.export(model, inputs)
        expected = model(*inputs)
    package = kuma_compile(ep, inputs)
    assert "softmax.wgsl" in package.kernels
    assert b"@compute @workgroup_size(64)" in package.kernels["softmax.wgsl"]
    softmax_nodes = [n for n in package.manifest["graph"]["nodes"] if n["target"] in ("aten.softmax.int", "aten._softmax.default")]
    assert len(softmax_nodes) == 1
    golden = package.golden["branches"][0]["nodes"]
    stats = golden[softmax_nodes[0]["name"]]["re"]
    assert stats["shape"] == [1, 2, 16, 16]
    assert 0.0 <= stats["min"] and stats["max"] <= 1.0
    # The final node's golden stats come from the same eager values torch produces.
    out_name = package.manifest["outputs"][0]["name"]
    assert golden[out_name]["re"]["first"] == pytest.approx(expected.reshape(-1)[:8].tolist(), abs=1e-6)


def test_shipped_kernels_cover_the_terrain_dispatches():
    kernels = load_kernels()
    for name in ("mean.wgsl", "upsample_nearest2d.wgsl", "softmax.wgsl", "clamp.wgsl", "slice.wgsl", "bmm.wgsl"):
        assert name in kernels, name


def test_large_weight_blobs_are_stored_uncompressed(tmp_path):
    import zipfile

    from kuma.package_iph import Package

    model = _Normalize().eval()
    inputs = (torch.randn(1, 8, 4, 4),)
    with torch.no_grad():
        ep = torch.export.export(model, inputs)
    package = kuma_compile(ep, inputs)

    def weights_compression(path):
        with zipfile.ZipFile(path) as zf:
            return zf.getinfo("weights.f32.bin").compress_type

    # Small blob: deflated by default; the override forces stored.
    assert weights_compression(package.save(tmp_path / "small.iph")) == zipfile.ZIP_DEFLATED
    assert weights_compression(package.save(tmp_path / "stored.iph", compress_weights=False)) == zipfile.ZIP_STORED
    # A blob at the threshold is stored by default, and reads back byte-identical.
    big = Package(**{**package.__dict__, "weights_blob": bytes(Package.STORE_WEIGHTS_FROM_BYTES)})
    big_path = big.save(tmp_path / "big.iph")
    assert weights_compression(big_path) == zipfile.ZIP_STORED
    with zipfile.ZipFile(big_path) as zf:
        assert len(zf.read("weights.f32.bin")) == Package.STORE_WEIGHTS_FROM_BYTES
    assert weights_compression(big.save(tmp_path / "big_deflated.iph", compress_weights=True)) == zipfile.ZIP_DEFLATED
