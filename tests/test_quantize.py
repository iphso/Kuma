"""Int8 weight packages (kuma.quantize): format, conversion, and parity with the
fake-quantization measured in Loom (py/quant/fakequant.py)."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn

import kuma
from kuma.kernels import _QUANT_KERNEL_NAMES
from kuma.quantize import (
    DEFAULT_MIN_ELEMENTS,
    QUANT_WEIGHT_FILE,
    dequantize_int8,
    quantize_iph,
    quantize_int8,
    read_package,
)


def _fakequant_int8(w: np.ndarray, shape) -> np.ndarray:
    """Loom/py/quant/fakequant.py's quant(w, shape, "int8", 0), verbatim: the reference
    the int8 format must reproduce bit for bit."""
    qmax = 127
    rows = w.reshape(shape[0], -1) if len(shape) > 1 else w.reshape(1, -1)
    n = rows.shape[1]
    g = n
    r = rows.reshape(rows.shape[0], n // g, g)
    scale = np.abs(r).max(axis=2, keepdims=True) / qmax
    scale[scale == 0] = 1
    return (np.clip(np.round(r / scale), -qmax, qmax) * scale).astype(np.float32).reshape(w.shape)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 64, 3, padding=1)  # 4608 weights: quantized
        self.conv2 = nn.Conv2d(64, 8, 1)  # 512: stays float32
        self.fc = nn.Linear(8, 640)  # 5120: quantized

    def forward(self, x):
        y = self.conv2(torch.relu(self.conv1(x)))
        return self.fc(y.mean(dim=(2, 3)))


_INPUT = (torch.randn(1, 8, 8, 8),)


def _f32_weights(path) -> dict[str, np.ndarray]:
    """Every weight of a package as float32 (int8 ones dequantized)."""
    with zipfile.ZipFile(path) as zf:
        m = json.loads(zf.read("manifest.json"))
        f32 = zf.read(m["weight_file"])
        q8 = zf.read(m["quant_weight_file"]) if m.get("quant_weight_file") else b""
    out = {}
    for w in m["weights"]:
        if w["dtype"] == "float32":
            out[w["name"]] = np.frombuffer(f32, "<f4", w["n_elements"], w["byte_offset"])
        else:
            q = np.frombuffer(q8, np.int8, w["n_elements"], w["byte_offset"])
            qq = w["quant"]
            s = np.frombuffer(f32, "<f4", qq["scale_count"], qq["scale_offset"])
            out[w["name"]] = dequantize_int8(q, s, qq["group_size"])
    return out


@pytest.mark.parametrize("shape", [(64, 8, 3, 3), (640, 8), (5000,), (3, 4096), (7, 5, 1, 1)])
def test_quantize_int8_matches_fakequant_bit_for_bit(shape):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(shape) * rng.uniform(0.001, 3)).astype(np.float32)
    w.reshape(shape[0] if len(shape) > 1 else 1, -1)[0] = 0  # an all-zero row: scale 1
    flat = w.reshape(-1)
    flat[1::7] = flat[1::7].round(1)  # some exact .5 ties after scaling are likely
    q, scales, group = quantize_int8(w, shape)
    assert q.dtype == np.int8 and scales.dtype == np.float32
    assert np.abs(q.astype(np.int32)).max() <= 127
    assert group * scales.size == w.size
    ours = dequantize_int8(q, scales, group)
    ref = _fakequant_int8(w.copy(), list(shape)).reshape(-1)
    # Same values (fakequant may keep -0.0 where int8 can only say 0: equal as floats).
    assert np.array_equal(ours, ref)


def test_compile_int8_package_layout(tmp_path):
    out = tmp_path / "net.iph"
    kuma.export_model(_Net(), _INPUT, out=out, quantize="int8")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        m = json.loads(zf.read("manifest.json"))
        f32 = zf.read("weights.f32.bin")
        q8 = zf.read(QUANT_WEIGHT_FILE)
    assert m["format_version"] == 1
    assert m["quant_weight_file"] == QUANT_WEIGHT_FILE
    assert {f"kernels/{k}" for k in _QUANT_KERNEL_NAMES} <= names
    assert "golden.json" not in names
    assert any("golden.json dropped" in w for w in m["warnings"])

    by_name = {w["name"]: w for w in m["weights"]}
    assert by_name["conv1.weight"]["dtype"] == "int8"
    assert by_name["fc.weight"]["dtype"] == "int8"
    assert by_name["conv2.weight"]["dtype"] == "float32"  # 512 < 4096
    assert by_name["conv1.bias"]["dtype"] == "float32"
    for w in m["weights"]:
        if w["dtype"] == "float32":
            assert w["byte_offset"] % 4 == 0 and w["byte_offset"] + w["byte_length"] <= len(f32)
            continue
        qq = w["quant"]
        assert w["file"] == QUANT_WEIGHT_FILE
        assert w["byte_offset"] % 4 == 0 and w["byte_length"] == w["n_elements"]
        assert w["byte_offset"] + -(-w["byte_length"] // 4) * 4 <= len(q8)  # padded to 4
        assert qq["scheme"] == "int8_symmetric" and qq["scale_dtype"] == "float32"
        assert qq["scale_count"] == w["shape"][0]
        assert qq["group_size"] * qq["scale_count"] == w["n_elements"]
        assert qq["scale_offset"] % 4 == 0 and qq["scale_offset"] + 4 * qq["scale_count"] <= len(f32)
    assert m["quantization"]["quantized_elements"] == 4608 + 5120
    assert len(q8) == 4608 + 5120


def test_float32_export_is_unchanged(tmp_path):
    out = tmp_path / "net.iph"
    kuma.export_model(_Net(), _INPUT, out=out)
    with zipfile.ZipFile(out) as zf:
        m = json.loads(zf.read("manifest.json"))
        names = set(zf.namelist())
    assert m["format_version"] == 0
    assert "quant_weight_file" not in m
    assert QUANT_WEIGHT_FILE not in names
    assert not any(n.endswith("_q8.wgsl") for n in names)
    assert all(w["dtype"] == "float32" for w in m["weights"])


def test_convert_saved_package_matches_fakequant(tmp_path):
    src = tmp_path / "net.iph"
    dst = tmp_path / "net.q8.iph"
    kuma.export_model(_Net(), _INPUT, out=src)
    quantize_iph(src, dst)

    before = _f32_weights(src)
    after = _f32_weights(dst)
    assert before.keys() == after.keys()
    with zipfile.ZipFile(src) as zf:
        shapes = {w["name"]: w["shape"] for w in json.loads(zf.read("manifest.json"))["weights"]}
    for name, w in before.items():
        if w.size >= DEFAULT_MIN_ELEMENTS:
            assert np.array_equal(after[name], _fakequant_int8(w.copy(), shapes[name]).reshape(-1)), name
        else:
            assert np.array_equal(after[name], w), name

    # Everything but the weights, manifest, kernels and report carries over unchanged.
    a, b = read_package(src), read_package(dst)
    assert a.manifest["graph"] == b.manifest["graph"]
    assert a.manifest["inputs"] == b.manifest["inputs"]
    assert set(b.kernels) == set(a.kernels) | set(_QUANT_KERNEL_NAMES)
    assert all(b.kernels[k] == v for k, v in a.kernels.items())
    assert "Int8 weights" in b.debug_report


def test_convert_options(tmp_path):
    src = tmp_path / "net.iph"
    kuma.export_model(_Net(), _INPUT, out=src)
    only_fc = quantize_iph(src, tmp_path / "fc.iph", include=r"^fc\.")
    m = json.loads(zipfile.ZipFile(only_fc).read("manifest.json"))
    assert [w["name"] for w in m["weights"] if w["dtype"] == "int8"] == ["fc.weight"]

    everything = quantize_iph(src, tmp_path / "all.iph", min_elements=1, keep_golden=True)
    with zipfile.ZipFile(everything) as zf:
        m = json.loads(zf.read("manifest.json"))
        assert "golden.json" in zf.namelist()
    assert all(w["dtype"] == "int8" for w in m["weights"])

    with pytest.raises(ValueError, match="already quantized"):
        quantize_iph(everything, tmp_path / "again.iph")


def test_unknown_quantize_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="quantize"):
        kuma.export_model(_Net(), _INPUT, out=tmp_path / "x.iph", quantize="int4")
