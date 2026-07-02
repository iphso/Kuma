"""Branching export: assemble N independently-traced ExportedPrograms plus a
caller-provided JS router snippet into one .iph Package with real conditional
execution at runtime (kuma-bart's scheduler resolves the router synchronously and
splices in only the chosen branch — the other branches are never dispatched).

This is a new, narrowly-scoped primitive (see kuma-bart's plan notes): the router
snippet's source is caller-provided (it's specific to whatever routing logic the
model needs, not a general Kuma kernel), and branch assembly is caller-driven (each
branch is independently torch.export'd by the caller — this module does not attempt
to detect branching inside a single traced graph).
"""

from __future__ import annotations

from typing import Any

import torch.export

from kuma.golden import capture_golden, namespace_golden
from kuma.graph import serialize_graph
from kuma.kernels import load_kernels
from kuma.manifest import build_playback_meta
from kuma.pack_weights import pack_weights


def _namespace_ref(value: Any, prefix: str) -> Any:
    """Recursively prefix every {"node_ref": ...} found in an args/kwargs value."""
    if isinstance(value, dict) and "node_ref" in value:
        return {"node_ref": f"{prefix}{value['node_ref']}"}
    if isinstance(value, list):
        return [_namespace_ref(v, prefix) for v in value]
    return value


def _namespace_nodes(nodes: list[dict[str, Any]], node_prefix: str, weight_prefix: str) -> list[dict[str, Any]]:
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
        n["args"] = [_namespace_ref(a, node_prefix) for a in node["args"]]
        n["kwargs"] = {k: _namespace_ref(v, node_prefix) for k, v in node["kwargs"].items()}
        if "weight_name" in node:
            n["weight_name"] = f"{weight_prefix}{node['weight_name']}"
        out.append(n)
    return out


def _namespace_weight_entries(
    entries: list[dict[str, Any]], weight_prefix: str, byte_offset_base: int
) -> list[dict[str, Any]]:
    out = []
    for e in entries:
        ne = dict(e)
        ne["name"] = f"{weight_prefix}{e['name']}"
        ne["byte_offset"] = e["byte_offset"] + byte_offset_base
        out.append(ne)
    return out


def _filter_and_repack_weights(
    blob: bytes, entries: list[dict[str, Any]], keep_names: set[str]
) -> tuple[bytes, list[dict[str, Any]]]:
    """pack_weights(ep) draws from ep.state_dict, which — unlike ep.graph_signature —
    isn't narrowed to parameters this specific branch's traced graph actually reads.
    For a multi-segment model that means every OTHER segment's weights leak into every
    branch's blob too (verified: a branch's own placeholder nodes only ever reference
    its own ~63 weights, but pack_weights(ep) for that same branch returns ~4x that).
    Keep only entries actually referenced by name, re-tightly-packed (4-byte aligned)
    into a fresh blob with byte_offsets recomputed relative to that new blob.
    """
    kept = [e for e in entries if e["name"] in keep_names]
    missing = keep_names - {e["name"] for e in kept}
    if missing:
        raise ValueError(f"weights referenced by the graph but missing from pack_weights output: {missing}")

    parts: list[bytes] = []
    offset = 0
    out_entries: list[dict[str, Any]] = []
    for e in kept:
        pad = (-offset) % 4
        if pad:
            parts.append(b"\x00" * pad)
            offset += pad
        parts.append(blob[e["byte_offset"]: e["byte_offset"] + e["byte_length"]])
        ne = dict(e)
        ne["byte_offset"] = offset
        out_entries.append(ne)
        offset += e["byte_length"]
    return b"".join(parts), out_entries


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


def compile_branching(
    router_snippet_name: str,
    router_snippet_source: str,
    router_input_names: list[str],
    router_output_specs: list[dict[str, Any]],
    selector_output_index: int,
    branch_input_output_index: int,
    branch_eps: list[torch.export.ExportedProgram],
    branch_example_inputs: list[tuple] | None = None,
    *,
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> Any:
    """
    Assemble a branching Package.

    fps/duration_seconds — optional playback metadata: how many real seconds a full
    sweep of the router's normalized time input (t in [0,1]) should take. Neither is
    derivable from the branches themselves (a model has no intrinsic notion of "real
    time" -- only the caller, who knows how this multi-segment export was actually
    authored/timed, does). A runtime falls back to its own default sweep speed when
    omitted, same as before this existed -- this is exactly the kind of model where it
    matters most, since different multi-segment exports are commonly authored at
    different speeds, and a runtime has no way to tell just from the graph.

    router_input_names  — top-level model input names the snippet reads (these become
                           the manifest's own `inputs`).
    router_output_specs — [{"shape": [...], "dtype": "..."}], one per value the
                           snippet's `main()` returns, in order.
    selector_output_index    — which router output picks the branch (must resolve to
                                exactly one value at runtime).
    branch_input_output_index — which router output feeds each branch's one
                                 (the routed value, e.g. local_norm_t) input.
    branch_eps — one independently torch.export'd ExportedProgram per branch, e.g.
                 via `torch.export.export(wrapper_for_segment_i, (example_local_t,))`.
    branch_example_inputs — optional, one example-input tuple per branch (same ones
                             each branch_eps[i] was traced with) -- when given, captures
                             golden.json (a real eager run's per-node value stats, for
                             verifying a runtime against), namespaced the same way as
                             each branch's own node names. Omit to skip golden capture
                             entirely, same as before this existed.
    """
    from kuma.package_iph import Package  # local import: avoids a package_iph<->branching cycle

    if not branch_eps:
        raise ValueError("compile_branching requires at least one branch ExportedProgram")

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
                {"shape": spec["shape"], "dtype": spec.get("dtype", "float32")} for spec in router_output_specs
            ]
        },
    }

    getitem_names: list[str] = []
    getitem_nodes: list[dict[str, Any]] = []
    for i in range(len(router_output_specs)):
        name = f"router_out_{i}"
        getitem_names.append(name)
        getitem_nodes.append(
            {
                "id": i + 1,
                "name": name,
                "op": "call_function",
                "target": "getitem",
                "args": [{"node_ref": router_node_name}, i],
                "kwargs": {},
                "meta": {},
            }
        )

    branches: list[dict[str, Any]] = []
    all_weight_entries: list[dict[str, Any]] = []
    blob_parts: list[bytes] = []
    warnings: list[str] = []
    running_offset = 0
    switch_output_shape: list[int] | None = None
    golden_branches: list[dict[str, Any]] | None = [] if branch_example_inputs is not None else None

    for i, ep in enumerate(branch_eps):
        node_prefix = f"branch{i}__"
        weight_prefix = f"branch{i}."

        if golden_branches is not None:
            golden_branches.append(namespace_golden(capture_golden(ep, branch_example_inputs[i]), node_prefix))

        graph_data = serialize_graph(ep)
        branch_weights_blob, weight_entries, skipped = pack_weights(ep)
        warnings.extend(f"branch {i}: skipped non-float32 tensor: {s}" for s in skipped)

        raw_nodes = graph_data["nodes"]
        fx_output_node = next(n for n in raw_nodes if n["op"] == "output")
        pruned_nodes = _prune_unused_placeholders(raw_nodes)
        body_nodes = [n for n in pruned_nodes if n["op"] != "output"]

        packed_weight_names = {entry["name"] for entry in weight_entries}
        used_weight_names = {
            n["weight_name"] for n in body_nodes
            if n["op"] == "placeholder" and n.get("weight_name") in packed_weight_names
        }
        tight_blob, kept_entries = _filter_and_repack_weights(branch_weights_blob, weight_entries, used_weight_names)

        pad = (-running_offset) % 4
        if pad:
            blob_parts.append(b"\x00" * pad)
            running_offset += pad

        all_weight_entries.extend(_namespace_weight_entries(kept_entries, weight_prefix, running_offset))
        blob_parts.append(tight_blob)
        running_offset += len(tight_blob)

        namespaced_body = _namespace_nodes(body_nodes, node_prefix, weight_prefix)

        branch_output_ref = _namespace_ref(fx_output_node["args"][0][0], node_prefix)
        output_producer = next(n for n in namespaced_body if n["name"] == branch_output_ref["node_ref"])
        branch_output_shape = output_producer.get("meta", {}).get("shape")
        if i == 0:
            switch_output_shape = branch_output_shape

        user_inputs = [n for n in namespaced_body if n["op"] == "placeholder" and n.get("kind") == "user_input"]
        if len(user_inputs) != 1:
            raise ValueError(
                f"branch {i}: expected exactly 1 user_input placeholder (the routed value), "
                f"got {len(user_inputs)}"
            )

        branches.append(
            {
                "nodes": namespaced_body,
                "inputs": [{"node_ref": user_inputs[0]["name"]}],
                "output": branch_output_ref,
            }
        )

    weights_blob = b"".join(blob_parts)

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

    output_node = {
        "id": 1_000_001,
        "name": "output",
        "op": "output",
        "target": "output",
        "args": [[{"node_ref": switch_name}]],
        "kwargs": {},
        "meta": {},
    }

    all_nodes = [router_node, *getitem_nodes, switch_node, output_node]

    manifest = {
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
        "unsupported_ops": [],
    }
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback

    debug_report = (
        "# Kuma — Branching Export Debug Report\n\n"
        f"- Branches: {len(branch_eps)}\n"
        f"- Router snippet: {router_snippet_name}\n"
        f"- Total weight bytes: {len(weights_blob):,}\n"
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
