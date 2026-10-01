"""Branching export: assemble N independently-traced ExportedPrograms plus a
caller-provided JS router snippet into one .iph Package with real conditional
execution at runtime (kuma-bart's scheduler resolves the router synchronously and
splices in only the chosen branch — the other branches are never dispatched).

This is a narrowly-scoped primitive: the router snippet's source is caller-provided
(it's specific to whatever routing logic the model needs, not a general Kuma kernel),
and branch assembly is caller-driven (each branch is independently torch.export'd by
the caller — this module does not attempt to detect branching inside a single traced
graph). The router/switch graph and branch namespacing are shared with the ONNX paths
via kuma.branch_assembly.
"""

from __future__ import annotations

from typing import Any

import torch.export

from kuma.branch_assembly import KumaBranch, assemble_kuma_branching_package, validate_branching_args
from kuma.golden import capture_golden, namespace_golden
from kuma.graph import serialize_graph
from kuma.pack_weights import pack_weights
from kuma.package_iph import Package


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
) -> Package:
    """
    Assemble a branching Package.

    fps/duration_seconds — optional playback metadata: how many real seconds a full
    sweep of the router's normalized time input (t in [0,1]) should take. Neither is
    derivable from the branches themselves (a model has no intrinsic notion of "real
    time" -- only the caller, who knows how this multi-segment export was actually
    authored/timed, does). A runtime falls back to its own default sweep speed when
    omitted -- this is exactly the kind of model where it matters most, since different
    multi-segment exports are commonly authored at different speeds, and a runtime has
    no way to tell just from the graph.

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
                 Every branch must produce the same output shape.
    branch_example_inputs — optional, one example-input tuple per branch (same ones
                             each branch_eps[i] was traced with) -- when given, captures
                             golden.json (a real eager run's per-node value stats, for
                             verifying a runtime against), namespaced the same way as
                             each branch's own node names. Omit to skip golden capture.
    """
    validate_branching_args(
        router_output_specs=router_output_specs,
        selector_output_index=selector_output_index,
        branch_input_output_index=branch_input_output_index,
        n_branches=len(branch_eps),
        n_example_inputs=None if branch_example_inputs is None else len(branch_example_inputs),
        branches_arg="branch_eps",
    )

    branches: list[KumaBranch] = []
    warnings: list[str] = []
    golden_branches: list[dict[str, Any]] | None = [] if branch_example_inputs is not None else None

    for i, ep in enumerate(branch_eps):
        if golden_branches is not None:
            golden_branches.append(namespace_golden(capture_golden(ep, branch_example_inputs[i]), f"branch{i}__"))

        graph_data = serialize_graph(ep)
        blob, weight_entries, skipped = pack_weights(ep)
        warnings.extend(f"branch {i}: skipped non-float32 tensor: {s}" for s in skipped)
        branches.append(KumaBranch(nodes=graph_data["nodes"], weights_blob=blob, weight_entries=weight_entries))

    return assemble_kuma_branching_package(
        router_snippet_name=router_snippet_name,
        router_snippet_source=router_snippet_source,
        router_input_names=router_input_names,
        router_output_specs=router_output_specs,
        selector_output_index=selector_output_index,
        branch_input_output_index=branch_input_output_index,
        branches=branches,
        warnings=warnings,
        unsupported_ops=[],
        golden_branches=golden_branches,
        fps=fps,
        duration_seconds=duration_seconds,
        report_title="Kuma — Branching Export Debug Report",
    )
