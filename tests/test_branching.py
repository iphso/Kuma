"""compile_branching (torch), compile_onnx / compile_branching_onnx (ort-web formats),
and torch-path golden capture."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

import kuma
from conftest import assert_weight_sizes_match
from kuma.compiler import compile as kuma_compile
from kuma.export import export_program

_ROUTER_JS = "function main(inputs) { return [[0], inputs[0]]; }"


def _router_kwargs(**overrides):
    kw = dict(
        router_snippet_name="router.js",
        router_snippet_source=_ROUTER_JS,
        router_input_names=["t"],
        router_output_specs=[{"shape": [1], "dtype": "int64"}, {"shape": [1]}],
        selector_output_index=0,
        branch_input_output_index=1,
    )
    kw.update(overrides)
    return kw


class _Segment(nn.Module):
    """t:[1] -> [1, out]: a tiny Linear so each branch carries its own weights."""

    def __init__(self, out_features: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(1, out_features)

    def forward(self, t):
        return torch.relu(self.fc(t.reshape(1, 1)))


_T = (torch.tensor([0.25]),)


def _eps(*segments):
    return [export_program(s, _T) for s in segments]


# ── compile_branching (torch) ─────────────────────────────────────────────────

def test_compile_branching_structure_and_weights(tmp_path):
    segs = [_Segment(seed=0), _Segment(seed=1)]
    pkg = kuma.compile_branching(**_router_kwargs(), branch_eps=_eps(*segs), branch_example_inputs=[_T, _T])
    m = pkg.manifest
    assert m["format"] == "kuma"
    names = [n["name"] for n in m["graph"]["nodes"]]
    assert names == ["router", "router_out_0", "router_out_1", "switch_0", "output"]
    switch = m["graph"]["nodes"][3]
    assert switch["selector"] == {"node_ref": "router_out_0"}
    assert switch["args"] == [{"node_ref": "router_out_1"}]
    assert len(switch["branches"]) == 2
    assert switch["meta"]["shape"] == [1, 4]
    assert m["outputs"] == [{"name": "output", "shape": [1, 4]}]

    # Every branch node + weight is namespaced, and each branch only carries its own weights.
    for i, br in enumerate(switch["branches"]):
        assert all(n["name"].startswith(f"branch{i}__") for n in br["nodes"])
        assert br["output"]["node_ref"].startswith(f"branch{i}__")
    weight_names = sorted(w["name"] for w in m["weights"])
    assert weight_names == ["branch0.fc.bias", "branch0.fc.weight", "branch1.fc.bias", "branch1.fc.weight"]
    assert_weight_sizes_match(m["weights"], pkg.weights_blob)
    for w in m["weights"]:
        assert w["byte_offset"] % 4 == 0
    # Weights round-trip to the right branch's parameters.
    for i, seg in enumerate(segs):
        e = next(w for w in m["weights"] if w["name"] == f"branch{i}.fc.weight")
        raw = np.frombuffer(pkg.weights_blob[e["byte_offset"]: e["byte_offset"] + e["byte_length"]], dtype="<f4")
        np.testing.assert_array_equal(raw.reshape(e["shape"]), seg.fc.weight.detach().numpy())

    # Golden keys are namespaced exactly like the branch node names.
    assert len(pkg.golden["branches"]) == 2
    for i, (gb, br) in enumerate(zip(pkg.golden["branches"], switch["branches"])):
        branch_names = {n["name"] for n in br["nodes"]}
        assert gb["nodes"] and set(gb["nodes"]) <= branch_names
        assert all(k.startswith(f"branch{i}__") for k in gb["inputs"])

    path = pkg.save(tmp_path / "b.iph")
    with zipfile.ZipFile(path) as zf:
        assert "snippets/router.js" in zf.namelist()
        assert "golden.json" in zf.namelist()


def test_compile_branching_rejects_mismatched_output_shapes():
    with pytest.raises(ValueError, match="output shapes differ"):
        kuma.compile_branching(**_router_kwargs(), branch_eps=_eps(_Segment(4), _Segment(5)))


def test_compile_branching_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        kuma.compile_branching(**_router_kwargs(), branch_eps=_eps(_Segment(), _Segment()), branch_example_inputs=[_T])


def test_compile_branching_rejects_empty_and_bad_indices():
    with pytest.raises(ValueError, match="at least one branch"):
        kuma.compile_branching(**_router_kwargs(), branch_eps=[])
    with pytest.raises(ValueError, match="out of range"):
        kuma.compile_branching(**_router_kwargs(selector_output_index=2), branch_eps=_eps(_Segment()))


# ── ort-web ONNX formats ──────────────────────────────────────────────────────

def test_compile_onnx_package(tmp_path):
    pkg = kuma.compile_onnx(_Segment(), _T, input_names=["t"], output_names=["y"], fps=30.0)
    assert pkg.manifest["format"] == "onnx"
    assert pkg.manifest["playback"] == {"fps": 30.0}
    model = onnx.load_model_from_string(pkg.models["model.onnx"])
    assert [i.name for i in model.graph.input] == ["t"]
    path = pkg.save(tmp_path / "o.iph")
    with zipfile.ZipFile(path) as zf:
        assert "models/model.onnx" in zf.namelist()


def test_compile_onnx_validates_inputs():
    with pytest.raises(ValueError, match="float32"):
        kuma.compile_onnx(_Segment(), (torch.tensor([0.25], dtype=torch.float64),))
    with pytest.raises(TypeError, match="tuple"):
        kuma.compile_onnx(_Segment(), torch.tensor([0.25]))


def test_compile_branching_onnx_package():
    pkg = kuma.compile_branching_onnx(
        **_router_kwargs(), branch_wrappers=[_Segment(seed=0), _Segment(seed=1)], branch_example_inputs=[_T, _T]
    )
    m = pkg.manifest
    assert m["format"] == "onnx-branching"
    switch = next(n for n in m["graph"]["nodes"] if n["op"] == "switch")
    assert switch["branches"] == [{"model_file": "models/segment_0.onnx"}, {"model_file": "models/segment_1.onnx"}]
    assert set(pkg.models) == {"segment_0.onnx", "segment_1.onnx"}
    for blob in pkg.models.values():
        onnx.checker.check_model(onnx.load_model_from_string(blob))


def test_compile_branching_onnx_validation():
    with pytest.raises(ValueError, match="same length"):
        kuma.compile_branching_onnx(**_router_kwargs(), branch_wrappers=[_Segment(), _Segment()], branch_example_inputs=[_T])
    with pytest.raises(ValueError, match="output shapes differ"):
        kuma.compile_branching_onnx(**_router_kwargs(), branch_wrappers=[_Segment(4), _Segment(5)], branch_example_inputs=[_T, _T])


def test_compile_branching_onnx_kuma_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="output shapes differ"):
        kuma.compile_branching_onnx_kuma(**_router_kwargs(), branch_wrappers=[_Segment(4), _Segment(5)], branch_example_inputs=[_T, _T])


# ── torch golden capture ──────────────────────────────────────────────────────

class _Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x)) * 2.0


def test_torch_golden_capture_matches_eager(tmp_path):
    model = _Small().eval()
    x = torch.randn(1, 2, 5, 5)
    ep = export_program(model, (x,))
    pkg = kuma_compile(ep, (x,))
    golden = pkg.golden["branches"][0]

    call_names = {n["name"] for n in pkg.manifest["graph"]["nodes"] if n["op"] == "call_function"}
    assert set(golden["nodes"]) == call_names
    (input_name,) = golden["inputs"]
    np.testing.assert_allclose(golden["inputs"][input_name], x.reshape(-1).tolist(), rtol=0, atol=0)

    out_ref = pkg.manifest["outputs"][0]["name"]
    stats = golden["nodes"][out_ref]["re"]
    with torch.no_grad():
        expected = model(x)
    assert stats["shape"] == list(expected.shape)
    assert stats["n"] == expected.numel() == stats["finite"]
    assert stats["mean"] == pytest.approx(float(expected.mean()), rel=1e-5)
    flat = expected.reshape(-1)
    assert stats["spread"] == pytest.approx([float(flat[i]) for i in stats["spread_indices"]])

    # golden.json is written with sorted keys -> byte-stable across runs.
    path = pkg.save(tmp_path / "g.iph")
    with zipfile.ZipFile(path) as zf:
        text = zf.read("golden.json").decode()
    assert text == json.dumps(json.loads(text), sort_keys=True)
