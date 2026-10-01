"""Captures a compact, reproducible "golden" snapshot of every FX node's actual eager
runtime value -- a companion to manifest.json for verifying a runtime (e.g. kuma-bart)
computes the *same* values a real PyTorch run would, not just NaN-free ones. Deliberately
minimal: a handful of summary stats per node (shape, mean/min/max, a few sample values),
never full activations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.export
import torch.fx

try:
    from torch.export.graph_signature import InputKind
except ImportError:
    from torch._export.graph_signature import InputKind  # type: ignore[no-redef]


_FIRST_SAMPLES = 8
_SPREAD_SAMPLES = 8
# Fixed seed for the spread-index LCG -- arbitrary, just needs to be stable across runs
# so re-exporting the same model reproduces the same golden.json. The chosen indices are
# stored explicitly in the output, so the consuming side never needs to reproduce this
# LCG itself; it just reads `spread_indices` and samples its own tensor at those offsets.
_SPREAD_SEED = 0x6B756D61  # "kuma" as bytes, read as an int


def _spread_indices(n: int, count: int = _SPREAD_SAMPLES) -> list[int]:
    """`count` deterministic indices spread across [0, n) via a small LCG -- not
    torch's RNG, so this has no dependency on torch RNG state, version, or device."""
    if n == 0:
        return []
    state = _SPREAD_SEED
    indices = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        indices.append(state % n)
    return indices

def array_stats(value: Any) -> dict[str, Any] | None:
    """Summary stats for one real-valued array (numpy or anything np.asarray accepts).
    Shared by the torch golden path (below) and the ONNX backend's reference-evaluator
    golden capture, so both emit byte-for-byte the same golden.json schema. Returns
    None for non-numeric arrays."""
    arr = np.asarray(value)
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        return None
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    n = int(flat.size)
    finite_mask = np.isfinite(flat)
    finite_vals = flat[finite_mask]
    nfin = int(finite_mask.sum())
    spread_idx = _spread_indices(n)
    return {
        "shape": list(arr.shape),
        "n": n,
        "finite": nfin,
        "mean": float(finite_vals.mean()) if nfin else float("nan"),
        "min": float(finite_vals.min()) if nfin else float("nan"),
        "max": float(finite_vals.max()) if nfin else float("nan"),
        "first": flat[: min(_FIRST_SAMPLES, n)].tolist(),
        "spread_indices": spread_idx,
        "spread": [float(flat[i]) for i in spread_idx],
    }


def value_stats(value: Any) -> dict[str, Any] | None:
    """Stats for one node's runtime value (torch tensor or numpy array). Real part always
    under "re"; an "im" key is added only when the value is complex -- mirroring
    kuma-bart's ResolvedTensor.imag pairing (a complex value is never one interleaved
    buffer on either side). Returns None for non-tensor values."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        value = value.resolve_conj().numpy() if torch.is_complex(value) else value.to(torch.float32).numpy()
    elif not isinstance(value, np.ndarray):
        return None
    if np.iscomplexobj(value):
        re, im = array_stats(value.real), array_stats(value.imag)
        return {"re": re, "im": im} if re is not None and im is not None else None
    re = array_stats(value)
    return {"re": re} if re is not None else None


class _GoldenInterpreter(torch.fx.Interpreter):
    """Runs the graph exactly as torch.fx.Interpreter normally would, but records a
    stats snapshot for every (non-placeholder, non-output) node's actual value along
    the way. Placeholders are skipped because they're either the example input itself
    (captured separately) or a raw weight (already in weights.f32.bin -- no need to
    duplicate it here); the output node is skipped because whatever feeds it already
    has its own entry under its own name."""

    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__(graph_module)
        self.captured: dict[str, dict[str, Any]] = {}

    def run_node(self, n: torch.fx.Node) -> Any:
        result = super().run_node(n)
        if n.op in ("placeholder", "output"):
            return result
        stats = value_stats(result)
        if stats is not None:
            self.captured[n.name] = stats
        return result


def _build_full_args(ep: torch.export.ExportedProgram, example_inputs: tuple) -> tuple:
    """ep.graph_module's graph has a placeholder for every parameter/buffer the
    original model held, in *graph* order, ahead of the user-supplied inputs --
    torch.export lifts them into explicit graph inputs. ep.module() hides this by
    binding them itself, but in doing so rebuilds the graph, which risks node names no
    longer matching serialize_graph's (which reads ep.graph_module.graph directly).
    Running the Interpreter on ep.graph_module itself, with this full positional arg
    list reconstructed from ep.state_dict/ep.constants (the same sources pack_weights
    already draws from), keeps node names identical to what ends up in the manifest.
    """
    state_dict = dict(ep.state_dict)
    constants = dict(getattr(ep, "constants", None) or {})
    user_iter = iter(example_inputs)

    args = []
    constant_tensor_kind = getattr(InputKind, "CONSTANT_TENSOR", None)

    for spec in ep.graph_signature.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER) or (
            constant_tensor_kind is not None and spec.kind == constant_tensor_kind
        ):
            tensor = state_dict.get(spec.target, constants.get(spec.target))
            if tensor is None:
                raise ValueError(f"capture_golden: no value found for weight '{spec.target}'")
            args.append(tensor)
        elif spec.kind == InputKind.USER_INPUT:
            args.append(next(user_iter))
        else:
            raise ValueError(f"capture_golden: unsupported input kind {spec.kind!r} for '{spec.arg.name}'")
    return tuple(args)


def capture_golden(ep: torch.export.ExportedProgram, example_inputs: tuple) -> dict[str, Any]:
    """One eager pass over `ep`'s graph with `example_inputs`, returning
    {"inputs": {placeholder_name: [...flat values...]}, "nodes": {node_name: {...stats...}}}
    -- keyed by the exact FX node names that also appear in the exported manifest
    (pre-branch-namespacing; callers assembling a multi-branch package are responsible
    for prefixing these keys the same way they prefix the graph's own node names).
    """
    full_args = _build_full_args(ep, example_inputs)
    interp = _GoldenInterpreter(ep.graph_module)
    with torch.no_grad():
        interp.run(*full_args)

    input_names = [spec.arg.name for spec in ep.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT]
    inputs_json = {
        name: t.detach().to(torch.float32).reshape(-1).tolist()
        for name, t in zip(input_names, example_inputs)
        if torch.is_tensor(t)
    }

    return {"inputs": inputs_json, "nodes": interp.captured}


def namespace_golden(golden: dict[str, Any], node_prefix: str) -> dict[str, Any]:
    """Prefixes every node/input name in a captured golden snapshot, mirroring
    branching.py's _namespace_nodes -- so a branch's golden keys line up with its
    namespaced node names in the final manifest."""
    return {
        "inputs": {f"{node_prefix}{name}": values for name, values in golden["inputs"].items()},
        "nodes": {f"{node_prefix}{name}": stats for name, stats in golden["nodes"].items()},
    }
