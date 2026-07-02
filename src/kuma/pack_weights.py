"""Extract parameters/buffers from ExportedProgram and pack into a contiguous f32 blob."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.export


def _is_synthesized_dft_basis(name: str) -> bool:
    return ".complex_tucker.lifted_tensor_" in name


def pack_weights(
    ep: torch.export.ExportedProgram,
) -> tuple[bytes, list[dict[str, Any]], list[str]]:
    """
    Pack all float32 parameters (and any float32 constants/buffers) into one binary blob.

    Layout: tensors are sorted by name, each 4-byte aligned.

    Returns:
        blob      — raw bytes (little-endian float32)
        entries   — list of dicts suitable for the manifest "weights" array
        skipped   — names of non-float32 tensors that were dropped (e.g. "name (torch.int64)")
    """
    # ep.state_dict holds lifted parameters and buffers
    all_tensors: dict[str, torch.Tensor] = dict(ep.state_dict)

    # ep.constants holds non-tensor constants and sometimes non-persistent buffers
    constants = getattr(ep, "constants", None) or {}
    for name, val in constants.items():
        if isinstance(val, torch.Tensor) and name not in all_tensors:
            all_tensors[name] = val

    if not all_tensors:
        return b"", [], []

    blob = bytearray()
    entries: list[dict[str, Any]] = []
    skipped: list[str] = []

    for name in sorted(all_tensors):
        tensor = all_tensors[name]

        if _is_synthesized_dft_basis(name):
            continue

        if tensor.dtype != torch.float32:
            # Non-float32 tensors (e.g. BN's num_batches_tracked: int64) are
            # training-only bookkeeping and not needed for inference.  Skip them
            # and surface them as warnings in the manifest rather than hard-failing.
            skipped.append(f"{name} ({tensor.dtype})")
            continue

        # 4-byte alignment between tensors
        pad = (-len(blob)) % 4
        if pad:
            blob.extend(b"\x00" * pad)

        t_cpu = tensor.detach().cpu().contiguous()
        arr: np.ndarray = t_cpu.numpy().astype("<f4", copy=False).flatten()
        raw: bytes = arr.tobytes()

        byte_offset = len(blob)
        byte_length = len(raw)
        n_elements = int(arr.size)

        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "float32",
                "byte_offset": byte_offset,
                "byte_length": byte_length,
                "n_elements": n_elements,
            }
        )

        blob.extend(raw)

    return bytes(blob), entries, skipped
