"""Package — the in-memory bundle that gets written out as a .iph file or a debug directory."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Package:
    """A compiled model, ready to be saved as a self-contained .iph package."""

    manifest: dict[str, Any]
    weights_blob: bytes
    graph_data: dict[str, Any]
    debug_report: str
    kernels: dict[str, bytes] = field(default_factory=dict)
    snippets: dict[str, bytes] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    # Optional companion to manifest.json: a handful of summary stats (shape/mean/min/
    # max/sample values) per graph node, captured from a real eager PyTorch run -- lets
    # a runtime (e.g. kuma-bart) verify it computes the same values, not just NaN-free
    # ones. None when the caller didn't have example inputs to capture it with.
    golden: dict[str, Any] | None = None
    # Optional model blobs (e.g. ONNX files) written to models/ in the zip. Populated
    # by the onnx compiler path; empty for standard Kuma/WGSL packages.
    models: dict[str, bytes] = field(default_factory=dict)

    def save(self, out: str | Path) -> Path:
        """Write the self-contained .iph package (a zip archive) to `out`."""
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(self.manifest, indent=2, sort_keys=True))
            zf.writestr("weights.f32.bin", self.weights_blob)
            for name, source in self.kernels.items():
                zf.writestr(f"kernels/{name}", source)
            for name, source in self.snippets.items():
                zf.writestr(f"snippets/{name}", source)
            for name, data in self.models.items():
                zf.writestr(f"models/{name}", data)
            zf.writestr("debug_report.md", self.debug_report)
            if self.golden is not None:
                zf.writestr("golden.json", json.dumps(self.golden))
        return out

    def write_dir(self, out_dir: str | Path) -> Path:
        """Write loose debug artifacts (manifest.json, weights.f32.bin, kernels/, snippets/,
        debug_report.md, exported_graph.json) to `out_dir` for manual inspection."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        (out_dir / "weights.f32.bin").write_bytes(self.weights_blob)
        (out_dir / "exported_graph.json").write_text(json.dumps(self.graph_data, indent=2))
        (out_dir / "debug_report.md").write_text(self.debug_report)
        if self.golden is not None:
            (out_dir / "golden.json").write_text(json.dumps(self.golden, indent=2))

        kernels_dir = out_dir / "kernels"
        kernels_dir.mkdir(exist_ok=True)
        for name, source in self.kernels.items():
            (kernels_dir / name).write_bytes(source)

        if self.snippets:
            snippets_dir = out_dir / "snippets"
            snippets_dir.mkdir(exist_ok=True)
            for name, source in self.snippets.items():
                (snippets_dir / name).write_bytes(source)

        if self.models:
            models_dir = out_dir / "models"
            models_dir.mkdir(exist_ok=True)
            for name, data in self.models.items():
                (models_dir / name).write_bytes(data)

        return out_dir
