"""Kuma: PyTorch torch.export -> .iph compiler/exporter (Step 1 of a Torch->WebGPU compiler)."""

from kuma.branching import compile_branching
from kuma.compiler import compile, export_exported_program, export_model
from kuma.onnx_backend import compile_branching_onnx_kuma, export_via_onnx
from kuma.onnx_compiler import compile_branching_onnx, compile_onnx
from kuma.package_iph import Package

__version__ = "0.1.0"

__all__ = [
    "compile",
    "compile_branching",
    "compile_branching_onnx",
    "compile_branching_onnx_kuma",
    "compile_onnx",
    "export_exported_program",
    "export_model",
    "export_via_onnx",
    "Package",
]
