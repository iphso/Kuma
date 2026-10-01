"""Shared assembly for branching (router + switch) .iph packages.

All three branching entry points -- kuma.branching.compile_branching (torch.export
branches), kuma.onnx_backend.compile_branching_onnx_kuma (ONNX -> Kuma branches) and
kuma.onnx_compiler.compile_branching_onnx (raw ONNX branches for ort-web) -- emit the
same outer graph:

    router (js_snippet) -> router_out_<i> (getitem) -> switch_0 (switch) -> output

and the two Kuma-format ones additionally namespace and concatenate each branch's
nodes and weights identically. That shared logic lives here so the three paths can't
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from kuma.kernels import load_kernels
from kuma.manifest import build_playback_meta
from kuma.pack_weights import append_aligned
from kuma.package_iph import Package

ROUTER_NODE_NAME = "router"
SWITCH_NODE_NAME = "switch_0"
_SWITCH_NODE_ID = 1_000_000
_OUTPUT_NODE_ID = 1_000_001


# ── Validation ────────────────────────────────────────────────────────────────

def validate_branching_args(
    *,
    router_output_specs: Sequence[dict[str, Any]],
    selector_output_index: int,
    branch_input_output_index: int,
    n_branches: int,
    n_example_inputs: int | None,
    branches_arg: str,
) -> None:
    """Fail loudly on malformed branching arguments before any (slow) tracing runs."""
    if n_branches == 0:
        raise ValueError(f"branching export requires at least one branch ({branches_arg} is empty)")
    if n_example_inputs is not None and n_example_inputs != n_branches:
        raise ValueError(
            f"{branches_arg} has {n_branches} entries but branch_example_inputs has {n_example_inputs}; "
            "they must be the same length (one example-input tuple per branch)"
        )
    n_out = len(router_output_specs)
    for arg_name, idx in (
        ("selector_output_index", selector_output_index),
        ("branch_input_output_index", branch_input_output_index),
    ):
        if not 0 <= idx < n_out:
            raise ValueError(f"{arg_name}={idx} is out of range for {n_out} router output spec(s)")


def check_branch_output_shapes(shapes: Sequence[list[int] | None]) -> list[int] | None:
    """Every branch feeds the same switch output, so every branch must produce the same
    shape (the manifest records only one). Returns that shape; raises if they differ."""
    first = shapes[0] if shapes else None
    for i, shape in enumerate(shapes):
        if shape != first:
            raise ValueError(
                f"branch output shapes differ: branch 0 produces {first} but branch {i} produces {shape}; "
                "all branches of a switch must produce the same output shape"
            )
    return first


# ── Outer router/switch graph ─────────────────────────────────────────────────

def build_router_switch_nodes(
    *,
    router_snippet_name: str,
    router_input_names: Sequence[str],
    router_output_specs: Sequence[dict[str, Any]],
    selector_output_index: int,
    branch_input_output_index: int,
    branches: list[dict[str, Any]],
    switch_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """router -> getitems -> switch(branches) -> output, as manifest graph nodes."""
    router_node = {
        "id": 0,
        "name": ROUTER_NODE_NAME,
        "op": "js_snippet",
        "target": router_snippet_name,
        "args": [{"node_ref": name} for name in router_input_names],
        "kwargs": {},
        "meta": {
            "outputs": [
                {"shape": spec["shape"], "dtype": spec.get("dtype", "float32")} for spec in router_output_specs
            ]
        },
    }

    getitem_names = [f"router_out_{i}" for i in range(len(router_output_specs))]
    getitem_nodes = [
        {
            "id": i + 1,
            "name": name,
            "op": "call_function",
            "target": "getitem",
            "args": [{"node_ref": ROUTER_NODE_NAME}, i],
            "kwargs": {},
            "meta": {},
        }
        for i, name in enumerate(getitem_names)
    ]

    switch_node = {
        "id": _SWITCH_NODE_ID,
        "name": SWITCH_NODE_NAME,
        "op": "switch",
        "target": "switch",
        "args": [{"node_ref": getitem_names[branch_input_output_index]}],
        "kwargs": {},
        "meta": switch_meta,
        "selector": {"node_ref": getitem_names[selector_output_index]},
        "branches": branches,
    }

    output_node = {
        "id": _OUTPUT_NODE_ID,
        "name": "output",
        "op": "output",
        "target": "output",
        "args": [[{"node_ref": SWITCH_NODE_NAME}]],
        "kwargs": {},
        "meta": {},
    }

    return [router_node, *getitem_nodes, switch_node, output_node]


# ── Node-graph utilities ──────────────────────────────────────────────────────

def collect_node_refs(value: Any, refs: set[str]) -> None:
    """Add every {"node_ref": name} found anywhere in an args/kwargs value to `refs`."""
    if isinstance(value, dict):
        node_ref = value.get("node_ref")
        if isinstance(node_ref, str):
            refs.add(node_ref)
        for v in value.values():
            collect_node_refs(v, refs)
    elif isinstance(value, list):
        for v in value:
            collect_node_refs(v, refs)


def prune_unused_placeholders(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop weight placeholders no non-placeholder node references (user inputs stay)."""
    refs: set[str] = set()
    for node in nodes:
        if node["op"] == "placeholder":
            continue
        collect_node_refs(node.get("args", []), refs)
        collect_node_refs(node.get("kwargs", {}), refs)

    return [
        node for node in nodes
        if node["op"] != "placeholder" or node.get("kind") == "user_input" or node["name"] in refs
    ]


def namespace_ref(value: Any, prefix: str) -> Any:
    """Recursively prefix every {"node_ref": ...} found in an args/kwargs value."""
    if isinstance(value, dict) and "node_ref" in value:
        return {"node_ref": f"{prefix}{value['node_ref']}"}
    if isinstance(value, list):
        return [namespace_ref(v, prefix) for v in value]
    return value


def namespace_nodes(nodes: list[dict[str, Any]], node_prefix: str, weight_prefix: str) -> list[dict[str, Any]]:
    """Prefix every node's own name + every reference to it, so a branch's internal
    names can never collide with the outer scope or another branch."""
    out = []
    for node in nodes:
        n = dict(node)
        n["name"] = f"{node_prefix}{node['name']}"
        if node["op"] == "placeholder":
            # placeholders' target == their own (pre-namespacing) name by convention
            # (see graph.py) -- keep that convention consistent post-namespacing too.
            n["target"] = f"{node_prefix}{node['target']}"
        n["args"] = [namespace_ref(a, node_prefix) for a in node["args"]]
        n["kwargs"] = {k: namespace_ref(v, node_prefix) for k, v in node["kwargs"].items()}
        if "weight_name" in node:
            n["weight_name"] = f"{weight_prefix}{node['weight_name']}"
        out.append(n)
    return out


def namespace_weight_entries(
    entries: list[dict[str, Any]], weight_prefix: str, byte_offset_base: int
) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        ne = dict(e)
        ne["name"] = f"{weight_prefix}{e['name']}"
        ne["byte_offset"] = e["byte_offset"] + byte_offset_base
        out.append(ne)
    return out


def filter_and_repack_weights(
    blob: bytes, entries: list[dict[str, Any]], keep_names: set[str]
) -> tuple[bytes, list[dict[str, Any]]]:
    """pack_weights(ep) draws from ep.state_dict, which — unlike ep.graph_signature —
    isn't narrowed to parameters this specific branch's traced graph actually reads.
    For a multi-segment model that means every OTHER segment's weights leak into every
    branch's blob too. Keep only entries actually referenced by name, re-tightly-packed
    (aligned) into a fresh blob with byte_offsets recomputed relative to that new blob.
    """
    kept = [e for e in entries if e["name"] in keep_names]
    missing = keep_names - {e["name"] for e in kept}
    if missing:
        raise ValueError(f"weights referenced by the graph but missing from pack_weights output: {sorted(missing)}")

    out = bytearray()
    out_entries: list[dict[str, Any]] = []
    for e in kept:
        offset = append_aligned(out, blob[e["byte_offset"]: e["byte_offset"] + e["byte_length"]])
        out_entries.append({**e, "byte_offset": offset})
    return bytes(out), out_entries


# ── Kuma-format branch assembly ───────────────────────────────────────────────

@dataclass
class KumaBranch:
    """One branch's un-namespaced Kuma graph + its packed weights.

    `nodes` is a full serialized node list (placeholders, call_functions and exactly one
    "output" node whose args[0][0] is the branch result); `weight_entries` index into
    `weights_blob`."""

    nodes: list[dict[str, Any]]
    weights_blob: bytes
    weight_entries: list[dict[str, Any]]


def assemble_kuma_branching_package(
    *,
    router_snippet_name: str,
    router_snippet_source: str,
    router_input_names: Sequence[str],
    router_output_specs: Sequence[dict[str, Any]],
    selector_output_index: int,
    branch_input_output_index: int,
    branches: list[KumaBranch],
    warnings: list[str],
    unsupported_ops: list[str],
    golden_branches: list[dict[str, Any]] | None,
    fps: float | None,
    duration_seconds: float | None,
    report_title: str,
    report_extra_lines: Sequence[str] = (),
) -> Package:
    """Namespace + concatenate each branch, wrap them in the router/switch graph, and
    build the format "kuma" Package. golden_branches (already namespaced, one per
    branch) is optional."""
    switch_branches: list[dict[str, Any]] = []
    all_weight_entries: list[dict[str, Any]] = []
    blob = bytearray()
    output_shapes: list[list[int] | None] = []

    for i, branch in enumerate(branches):
        node_prefix = f"branch{i}__"
        weight_prefix = f"branch{i}."

        fx_output_node = next(n for n in branch.nodes if n["op"] == "output")
        body_nodes = [n for n in prune_unused_placeholders(branch.nodes) if n["op"] != "output"]

        # Only weights the branch's own graph reads, and only ones actually in the blob
        # (synthesized weights -- see pack_weights._is_synthesized_dft_basis -- are not).
        packed_names = {entry["name"] for entry in branch.weight_entries}
        used_weight_names = {
            n["weight_name"] for n in body_nodes
            if n["op"] == "placeholder" and n.get("weight_name") in packed_names
        }
        tight_blob, kept_entries = filter_and_repack_weights(
            branch.weights_blob, branch.weight_entries, used_weight_names
        )
        base_offset = append_aligned(blob, tight_blob)
        all_weight_entries.extend(namespace_weight_entries(kept_entries, weight_prefix, base_offset))

        namespaced_body = namespace_nodes(body_nodes, node_prefix, weight_prefix)
        branch_output_ref = namespace_ref(fx_output_node["args"][0][0], node_prefix)
        output_producer = next(n for n in namespaced_body if n["name"] == branch_output_ref["node_ref"])
        output_shapes.append(output_producer.get("meta", {}).get("shape"))

        user_inputs = [n for n in namespaced_body if n["op"] == "placeholder" and n.get("kind") == "user_input"]
        if len(user_inputs) != 1:
            raise ValueError(
                f"branch {i}: expected exactly 1 user_input placeholder (the routed value), "
                f"got {len(user_inputs)}: {[n['name'] for n in user_inputs]}"
            )

        switch_branches.append(
            {
                "nodes": namespaced_body,
                "inputs": [{"node_ref": user_inputs[0]["name"]}],
                "output": branch_output_ref,
            }
        )

    switch_output_shape = check_branch_output_shapes(output_shapes)
    weights_blob = bytes(blob)

    all_nodes = build_router_switch_nodes(
        router_snippet_name=router_snippet_name,
        router_input_names=router_input_names,
        router_output_specs=router_output_specs,
        selector_output_index=selector_output_index,
        branch_input_output_index=branch_input_output_index,
        branches=switch_branches,
        switch_meta={"shape": switch_output_shape},
    )

    manifest: dict[str, Any] = {
        "format": "kuma",
        "format_version": 0,
        "weight_file": "weights.f32.bin",
        "endianness": "little",
        "inputs": [{"name": name, "kind": "user_input"} for name in router_input_names],
        "outputs": [{"name": "output", "shape": switch_output_shape}],
        "weights": all_weight_entries,
        "graph": {
            "node_count": len(all_nodes),
            "op_counts": {},
            "nodes": all_nodes,
        },
        "warnings": warnings,
        "unsupported_ops": unsupported_ops,
    }
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    debug_report = (
        f"# {report_title}\n\n"
        f"- Branches: {len(branches)}\n"
        f"- Router snippet: {router_snippet_name}\n"
        f"- Total weight bytes: {len(weights_blob):,}\n"
        + "".join(f"{line}\n" for line in report_extra_lines)
        + ("\n## Unsupported ONNX ops\n" + "\n".join(f"- {op}" for op in unsupported_ops) + "\n" if unsupported_ops else "")
        + ("\n## Warnings\n" + "\n".join(f"- {w}" for w in warnings) + "\n" if warnings else "")
    )

    golden = {"format_version": 0, "branches": golden_branches} if golden_branches is not None else None

    return Package(
        manifest=manifest,
        weights_blob=weights_blob,
        graph_data={"format_version": 0, "nodes": all_nodes},
        debug_report=debug_report,
        kernels=load_kernels(),
        snippets={router_snippet_name: router_snippet_source.encode("utf-8")},
        golden=golden,
    )
