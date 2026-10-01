"""Core export: validate model + inputs, run torch.export.export, return ExportedProgram."""

from __future__ import annotations

import torch
import torch.export


def _validate_inputs(example_inputs: tuple) -> None:
    for i, t in enumerate(example_inputs):
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype != torch.float32:
            raise ValueError(
                f"Input[{i}] has dtype {t.dtype}; only float32 is supported."
            )
        if t.device.type != "cpu":
            raise ValueError(
                f"Input[{i}] is on device '{t.device}'; only cpu is supported."
            )


def _validate_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            raise ValueError(
                f"Parameter '{name}' has dtype {param.dtype}; only float32 is supported."
            )
    for name, buf in model.named_buffers():
        if buf.dtype not in (torch.float32, torch.int64, torch.bool):
            raise ValueError(
                f"Buffer '{name}' has dtype {buf.dtype}; only float32/int64/bool buffers are supported."
            )


def validate_model_and_inputs(model: torch.nn.Module, example_inputs: tuple) -> None:
    """The input/parameter checks every capture path (torch.export and both ONNX
    paths) runs before tracing: example_inputs must be a tuple of cpu float32 tensors,
    parameters float32, buffers float32/int64/bool. Fails loudly otherwise."""
    if not isinstance(example_inputs, tuple):
        raise TypeError(f"example_inputs must be a tuple, got {type(example_inputs)}")
    _validate_inputs(example_inputs)
    _validate_parameters(model)


def export_program(
    model: torch.nn.Module,
    example_inputs: tuple,
) -> torch.export.ExportedProgram:
    """
    Validate a model + example inputs and run torch.export.export.

    Fails loudly on unsupported dtypes, non-CPU tensors, or dynamic-shape graphs.
    """
    model.eval()
    validate_model_and_inputs(model, example_inputs)

    with torch.no_grad():
        return torch.export.export(model, example_inputs)
