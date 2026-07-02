"""High-level compiler entry points: ExportedProgram -> Package -> .iph file."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.export

from kuma.debug import generate_debug_report
from kuma.export import export_program
from kuma.golden import capture_golden
from kuma.graph import serialize_graph
from kuma.kernels import load_kernels
from kuma.manifest import build_manifest, build_playback_meta
from kuma.pack_weights import pack_weights
from kuma.package_iph import Package


def compile(
    ep: torch.export.ExportedProgram,
    example_inputs: tuple | None = None,
    *,
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> Package:
    """Compile a captured ExportedProgram into an in-memory Package.

    `example_inputs` is optional and only used to capture golden.json (a real eager run's
    per-node value stats, for verifying a runtime against) -- when omitted (e.g. a caller
    only has the ExportedProgram, not the original inputs), the package simply has no
    golden data, same as before this existed.

    `fps`/`duration_seconds` are both optional, independent playback metadata -- for a
    model whose scalar input is a normalized time value (t in [0,1]) meant to be scrubbed
    or played back, `duration_seconds` is how many real seconds a full sweep should take;
    a runtime (e.g. kuma-bart's demo) uses it instead of guessing/hardcoding one fixed
    speed for every model. See manifest.json's optional top-level "playback" key.
    """
    graph_data = serialize_graph(ep)
    weights_blob, weight_entries, skipped = pack_weights(ep)
    warnings = [f"skipped non-float32 tensor: {s}" for s in skipped]
    manifest = build_manifest(ep, weight_entries, graph_data, warnings)
    playback = build_playback_meta(fps, duration_seconds)
    if playback is not None:
        manifest["playback"] = playback
    debug_report = generate_debug_report(ep, graph_data, weight_entries, manifest)
    # Wrapped in "branches" (a single trivial one, no namespacing needed) so this
    # matches kuma.branching.compile_branching's golden.json shape exactly -- one
    # uniform shape for a runtime's verifier to consume either way.
    golden = {"format_version": 0, "branches": [capture_golden(ep, example_inputs)]} if example_inputs is not None else None

    return Package(
        manifest=manifest,
        weights_blob=weights_blob,
        graph_data=graph_data,
        debug_report=debug_report,
        kernels=load_kernels(),
        skipped=skipped,
        golden=golden,
    )


def export_exported_program(
    ep: torch.export.ExportedProgram,
    out: str | Path,
    example_inputs: tuple | None = None,
    *,
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Compile an already-captured ExportedProgram and save it as a .iph package."""
    return compile(ep, example_inputs, fps=fps, duration_seconds=duration_seconds).save(out)


def export_model(
    model: torch.nn.Module,
    example_inputs: tuple,
    out: str | Path,
    *,
    backend: str = "torch",
    fps: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Run torch.export.export on `model` and save the result as a .iph package.

    backend="torch"  — direct torch.export path (default, existing behaviour)
    backend="onnx"   — export via ONNX as an intermediary (requires onnx package)
    """
    if backend == "onnx":
        from kuma.onnx_backend import export_via_onnx
        return export_via_onnx(
            model, example_inputs, fps=fps, duration_seconds=duration_seconds
        ).save(out)
    ep = export_program(model, example_inputs)
    return export_exported_program(ep, out, example_inputs, fps=fps, duration_seconds=duration_seconds)
