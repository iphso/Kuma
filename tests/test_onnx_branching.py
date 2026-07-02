"""Tests for compile_branching_onnx_kuma: structural contract and golden correctness.

These run entirely on the Python/CPU side (no WebGPU needed) and verify the manifest
and golden.json produced by the ONNX branching path before kuma-bart ever sees them.
If a golden key doesn't match the corresponding manifest node name, verify() will
silently report nodesMissing rather than actual=0 -- but a key mismatch in the INPUT
section causes the branch's user-input buffer to not be found, which would cause a
KumaManifestError throw rather than zeros. The actual 'all zeros' symptom points to
correct keys but wrong GPU computation -- these tests validate the Python-side contract
so the search can be narrowed to the TypeScript runtime.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from kuma.onnx_backend import compile_branching_onnx_kuma, export_via_onnx


# ── Simple 1-branch models for structural contract testing ────────────────────

class _AddConstant(nn.Module):
    """y = x + 1.0 — the simplest non-trivial op, output value is always input + 1."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _ScaleAndShift(nn.Module):
    """y = x * 2.0 + 0.5 — two ops, second depends on first."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 0.5


# Minimal router: always routes to branch 0, passes the input straight through.
# The function must be named `main` — kuma-bart's snippet evaluator does
# `new Function(source + "\\nreturn main;")` regardless of router_snippet_name.
_ROUTER_JS = """
function main(inputs) {
    const t = inputs[0];
    return [[0], t];
}
"""

_ROUTER_SPECS = {
    "name": "routerFn",
    "source": _ROUTER_JS,
    "input_names": ["input_0"],
    "output_specs": [{"shape": [1]}, {"shape": [1]}],
    "selector_index": 0,
    "branch_input_index": 1,
}


def _make_1branch_pkg(branch_module: nn.Module, example_input: torch.Tensor):
    """Compile a single-branch branching package for the given module."""
    return compile_branching_onnx_kuma(
        router_snippet_name=_ROUTER_SPECS["name"],
        router_snippet_source=_ROUTER_SPECS["source"],
        router_input_names=_ROUTER_SPECS["input_names"],
        router_output_specs=_ROUTER_SPECS["output_specs"],
        selector_output_index=_ROUTER_SPECS["selector_index"],
        branch_input_output_index=_ROUTER_SPECS["branch_input_index"],
        branch_wrappers=[branch_module.eval()],
        branch_example_inputs=[(example_input,)],
    )


# ── Golden structure contract ─────────────────────────────────────────────────

class TestGoldenStructure:
    """The golden dict structure produced by compile_branching_onnx_kuma."""

    def test_golden_has_branches_list(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        golden = pkg.golden
        assert golden is not None
        assert "branches" in golden
        assert len(golden["branches"]) == 1

    def test_golden_branch_has_inputs_and_nodes(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        assert "inputs" in branch, "golden branch must have 'inputs' dict"
        assert "nodes" in branch, "golden branch must have 'nodes' dict"

    def test_golden_inputs_key_matches_user_input_placeholder_name(self):
        """The golden.inputs key must exactly match the user_input placeholder's `name`
        in the branch's node list, because verifyBranch feeds inputBuffers by that key.
        A mismatch here means the branch input buffer is never found → KumaManifestError
        (throw, not zeros), but this is still the contract to enforce.
        """
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        golden_branch = pkg.golden["branches"][0]
        golden_input_keys = list(golden_branch["inputs"].keys())

        # Find the switch node's branch 0 node list
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]
        user_input_names = [
            n["name"] for n in branch_nodes
            if n["op"] == "placeholder" and n.get("kind") == "user_input"
        ]

        assert len(user_input_names) == 1, (
            f"Expected exactly 1 user_input placeholder in branch 0, "
            f"got: {user_input_names}"
        )
        assert len(golden_input_keys) == 1, (
            f"Expected exactly 1 golden input key, got: {golden_input_keys}"
        )
        assert golden_input_keys[0] == user_input_names[0], (
            f"golden.inputs key {golden_input_keys[0]!r} does not match "
            f"the user_input placeholder name {user_input_names[0]!r}. "
            f"verifyBranch will not find the input buffer."
        )

    def test_golden_input_values_are_list_of_floats(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        for key, values in branch["inputs"].items():
            assert isinstance(values, list), f"golden.inputs[{key!r}] must be a list"
            assert all(isinstance(v, (int, float)) for v in values), (
                f"golden.inputs[{key!r}] must be a list of numbers"
            )

    def test_golden_nodes_are_namespaced_with_branch_prefix(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        for key in branch["nodes"]:
            assert key.startswith("branch0__"), (
                f"golden node key {key!r} should start with 'branch0__' "
                f"after namespace_golden"
            )

    def test_golden_inputs_keys_are_namespaced_with_branch_prefix(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        for key in branch["inputs"]:
            assert key.startswith("branch0__"), (
                f"golden input key {key!r} should start with 'branch0__' "
                f"after namespace_golden"
            )


# ── Golden value correctness ──────────────────────────────────────────────────

class TestGoldenValues:
    """Verify that golden captures the correct PyTorch eager values, not zeros.

    These tests use example inputs where the expected output is analytically known,
    so a bug in golden capture (wrong values, or values captured before ops run)
    would be caught directly.
    """

    def _find_node_golden(self, pkg, name_suffix: str) -> dict[str, Any] | None:
        """Find a golden node whose key ends with `name_suffix`."""
        branch = pkg.golden["branches"][0]
        for key, stats in branch["nodes"].items():
            if key.endswith(name_suffix) or name_suffix in key:
                return stats
        return None

    def test_add_node_has_nonzero_mean_when_input_is_zero(self):
        """x=0.0 → x+1.0 = 1.0; the add node must have golden mean≈1.0, not 0.0."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        nodes = branch["nodes"]

        # Find any node that corresponds to the add op.
        # The ONNX node name for "x + 1.0" typically contains "Add".
        add_stats = None
        for key, stats in nodes.items():
            re_stats = stats.get("re", stats) if "re" in stats else stats
            # Pick the first node whose mean is approximately 1.0
            if abs(re_stats.get("mean", 0.0) - 1.0) < 0.01:
                add_stats = re_stats
                break

        assert add_stats is not None, (
            f"Expected a golden node with mean≈1.0 (the output of x+1.0 with x=0.0), "
            f"but found none. All golden node means: "
            f"{[(k, v.get('re', v).get('mean')) for k, v in nodes.items()]}"
        )
        assert abs(add_stats["mean"] - 1.0) < 0.01, (
            f"Expected add node mean≈1.0, got {add_stats['mean']}"
        )

    def test_golden_nodes_are_not_all_zero(self):
        """With a non-trivial model, at least some golden nodes should have nonzero values."""
        pkg = _make_1branch_pkg(_ScaleAndShift(), torch.ones(1))
        # input=1.0 → 1.0*2.0+0.5=2.5; some intermediate/final values must be nonzero
        branch = pkg.golden["branches"][0]
        nodes = branch["nodes"]
        assert len(nodes) > 0, "Expected at least one captured node in golden"
        means = [
            (v.get("re", v) if "re" in v else v).get("mean", 0.0)
            for v in nodes.values()
        ]
        assert any(abs(m) > 0.001 for m in means), (
            f"All golden nodes have mean≈0.0, which suggests golden capture is broken. "
            f"Means: {means}"
        )

    def test_golden_node_n_matches_expected_element_count(self):
        """n (element count) in each node stats must match the shape product."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        for key, stats in branch["nodes"].items():
            re_stats = stats.get("re", stats) if "re" in stats else stats
            n = re_stats.get("n")
            shape = re_stats.get("shape")
            if n is not None and shape is not None:
                expected = 1
                for d in shape:
                    expected *= d
                assert n == expected, (
                    f"golden node {key!r}: n={n} but shape {shape} implies {expected} elements"
                )

    def test_golden_nodes_have_no_negative_one_in_shapes(self):
        """Shapes must be fully concrete after ONNX reference evaluation -- no -1 dims."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        branch = pkg.golden["branches"][0]
        for key, stats in branch["nodes"].items():
            re_stats = stats.get("re", stats) if "re" in stats else stats
            shape = re_stats.get("shape", [])
            assert -1 not in shape, (
                f"golden node {key!r} has -1 in shape {shape}: "
                f"ONNX shape inference failed to concretize this dimension"
            )


# ── Manifest structure contract ───────────────────────────────────────────────

class TestManifestStructure:
    """The manifest structure produced by compile_branching_onnx_kuma."""

    def test_manifest_graph_has_switch_node(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        nodes = pkg.manifest["graph"]["nodes"]
        switch_nodes = [n for n in nodes if n["op"] == "switch"]
        assert len(switch_nodes) == 1, (
            f"Expected exactly 1 switch node, got {len(switch_nodes)}"
        )

    def test_branch_nodes_start_with_user_input_placeholder(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]
        user_inputs = [n for n in branch_nodes if n.get("kind") == "user_input"]
        assert len(user_inputs) == 1, (
            f"Branch 0 should have exactly 1 user_input placeholder, "
            f"got {len(user_inputs)}: {[n['name'] for n in user_inputs]}"
        )

    def test_all_branch_node_names_have_branch_prefix(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]
        for node in branch_nodes:
            assert node["name"].startswith("branch0__"), (
                f"Branch 0 node name {node['name']!r} should start with 'branch0__'"
            )

    def test_all_weight_names_have_branch_prefix(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        for w in pkg.manifest["weights"]:
            assert w["name"].startswith("branch0."), (
                f"Weight name {w['name']!r} should start with 'branch0.'"
            )

    def test_weight_names_in_nodes_match_manifest_weights(self):
        """Every weight_name referenced by a branch node must appear in manifest.weights."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]

        weight_names_in_manifest = {w["name"] for w in pkg.manifest["weights"]}
        for node in branch_nodes:
            if "weight_name" in node:
                assert node["weight_name"] in weight_names_in_manifest, (
                    f"Node {node['name']!r} references weight_name "
                    f"{node['weight_name']!r} but it's not in manifest.weights. "
                    f"This would cause a missing-weight error at runtime."
                )

    def test_all_node_args_refer_to_nodes_that_exist(self):
        """Every node_ref in args/kwargs must point to a node earlier in the list."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]
        seen_names: set[str] = set()

        def check_refs(value: Any, node_name: str) -> None:
            if isinstance(value, dict) and "node_ref" in value:
                ref = value["node_ref"]
                assert ref in seen_names, (
                    f"Node {node_name!r} refers to {ref!r} which hasn't been defined yet. "
                    f"Topological order violation."
                )
            elif isinstance(value, list):
                for v in value:
                    check_refs(v, node_name)

        for node in branch_nodes:
            for arg in node.get("args", []):
                check_refs(arg, node["name"])
            for v in node.get("kwargs", {}).values():
                check_refs(v, node["name"])
            seen_names.add(node["name"])

    def test_no_minus_one_shapes_in_manifest_nodes(self):
        """After ONNX shape inference + concretization, no node should have -1 shapes."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]
        for node in branch_nodes:
            shape = node.get("meta", {}).get("shape")
            if shape is not None:
                assert -1 not in shape, (
                    f"Node {node['name']!r} has -1 in shape {shape}. "
                    f"kuma-bart will fail to create a buffer for this node."
                )

    def test_golden_node_names_are_subset_of_manifest_node_names(self):
        """Every key in golden.nodes must correspond to a node in the manifest.

        A mismatch here means the verifier can't find the computed tensor by name
        and will report nodesMissing -- these are tracked separately from value
        mismatches, so this test catches a different failure mode than actual=0.
        """
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_nodes = switch_node["branches"][0]["nodes"]
        manifest_names = {n["name"] for n in branch_nodes}

        golden_branch = pkg.golden["branches"][0]
        for key in golden_branch["nodes"]:
            assert key in manifest_names, (
                f"golden.nodes key {key!r} is not a node name in the manifest. "
                f"verifyBranch will report this as nodesMissing."
            )

    def test_branch_output_ref_points_to_existing_node(self):
        """The branch's declared output node_ref must resolve to a node in the branch."""
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        switch_node = next(
            n for n in pkg.manifest["graph"]["nodes"] if n["op"] == "switch"
        )
        branch_0 = switch_node["branches"][0]
        output_ref = branch_0["output"]["node_ref"]
        branch_node_names = {n["name"] for n in branch_0["nodes"]}
        assert output_ref in branch_node_names, (
            f"Branch 0 declares output node_ref={output_ref!r} but no such node exists. "
            f"Available: {sorted(branch_node_names)}"
        )


# ── Weight blob integrity ─────────────────────────────────────────────────────

class TestWeightBlob:
    """The weights_blob must be internally consistent with manifest.weights."""

    def test_weight_byte_offsets_are_within_blob(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        blob = pkg.weights_blob
        for w in pkg.manifest["weights"]:
            end = w["byte_offset"] + w["byte_length"]
            assert end <= len(blob), (
                f"Weight {w['name']!r}: byte_offset {w['byte_offset']} + "
                f"byte_length {w['byte_length']} = {end} exceeds blob size {len(blob)}"
            )

    def test_weight_byte_length_matches_n_elements(self):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        for w in pkg.manifest["weights"]:
            expected = w["n_elements"] * 4
            assert w["byte_length"] == expected, (
                f"Weight {w['name']!r}: byte_length {w['byte_length']} != "
                f"n_elements {w['n_elements']} * 4 = {expected}"
            )


# ── IPH package round-trip ────────────────────────────────────────────────────

class TestIphRoundtrip:
    """The .iph zip produced by save() must contain golden.json with the right keys."""

    def test_iph_contains_golden_json(self, tmp_path):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        iph_path = pkg.save(tmp_path / "test.iph")
        with zipfile.ZipFile(iph_path) as zf:
            names = set(zf.namelist())
        assert "golden.json" in names, (
            f".iph zip does not contain golden.json. Contents: {sorted(names)}"
        )

    def test_golden_json_roundtrip_preserves_branch_keys(self, tmp_path):
        pkg = _make_1branch_pkg(_AddConstant(), torch.zeros(1))
        iph_path = pkg.save(tmp_path / "test.iph")
        with zipfile.ZipFile(iph_path) as zf:
            golden = json.loads(zf.read("golden.json"))
        assert "branches" in golden
        assert len(golden["branches"]) == 1
        branch = golden["branches"][0]
        assert "inputs" in branch
        assert "nodes" in branch
        # All keys must still have the branch prefix
        for key in branch["inputs"]:
            assert key.startswith("branch0__")
        for key in branch["nodes"]:
            assert key.startswith("branch0__")


# ── export_via_onnx single-branch golden ──────────────────────────────────────

class TestExportViaOnnxGolden:
    """Sanity checks on the single-branch export path that compile_branching_onnx_kuma builds on."""

    def test_single_branch_export_has_golden(self):
        pkg = export_via_onnx(_AddConstant().eval(), (torch.zeros(1),))
        assert pkg.golden is not None
        assert "branches" in pkg.golden
        assert len(pkg.golden["branches"]) == 1

    def test_single_branch_golden_input_key_matches_placeholder(self):
        """golden.inputs key must match the user_input placeholder name (before namespacing)."""
        pkg = export_via_onnx(_AddConstant().eval(), (torch.zeros(1),))
        branch = pkg.golden["branches"][0]
        golden_input_keys = list(branch["inputs"].keys())

        graph_nodes = pkg.graph_data["nodes"]
        user_input_names = [
            n["name"] for n in graph_nodes
            if n["op"] == "placeholder" and n.get("kind") == "user_input"
        ]

        assert len(user_input_names) == 1
        assert len(golden_input_keys) == 1
        assert golden_input_keys[0] == user_input_names[0], (
            f"Single-branch export: golden.inputs key {golden_input_keys[0]!r} "
            f"!= user_input placeholder name {user_input_names[0]!r}"
        )

    def test_add_output_golden_value_matches_eager_pytorch(self):
        """The ONNX reference evaluator must produce the same value as eager PyTorch."""
        model = _AddConstant().eval()
        x = torch.zeros(1)
        expected_output = model(x).item()  # = 1.0

        pkg = export_via_onnx(model, (x,))
        branch = pkg.golden["branches"][0]
        nodes = branch["nodes"]

        # Find the node with the final/largest mean value (should be 1.0 for x+1 with x=0)
        final_node_stats = None
        for key, stats in nodes.items():
            re_stats = stats.get("re", stats) if "re" in stats else stats
            if abs(re_stats.get("mean", 0.0) - expected_output) < 0.01:
                final_node_stats = re_stats
                break

        assert final_node_stats is not None, (
            f"Expected a golden node with mean≈{expected_output} (the model's output), "
            f"but found none. Node means: "
            f"{[(k, (v.get('re', v) if 're' in v else v).get('mean')) for k, v in nodes.items()]}"
        )
