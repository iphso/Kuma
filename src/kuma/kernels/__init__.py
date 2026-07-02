"""Load the embedded WGSL kernels shipped with every .iph package."""

from __future__ import annotations

from importlib import resources

_KERNEL_NAMES: tuple[str, ...] = (
    # elementwise binary
    "add.wgsl",
    "sub.wgsl",
    "mul.wgsl",
    "div.wgsl",
    "minimum.wgsl",
    # elementwise unary / activations
    "relu.wgsl",
    "gelu.wgsl",
    "silu.wgsl",
    "hardswish.wgsl",
    "sigmoid.wgsl",
    "tanh.wgsl",
    "exp.wgsl",
    "log.wgsl",
    "sqrt.wgsl",
    "rsqrt.wgsl",
    "neg.wgsl",
    "abs.wgsl",
    "pow_scalar.wgsl",
    "clamp.wgsl",
    "cos.wgsl",
    "sin.wgsl",
    "floor.wgsl",
    "erf.wgsl",
    # matmul / conv
    "linear.wgsl",
    "bmm.wgsl",
    "conv2d.wgsl",
    "conv2d_depthwise.wgsl",
    "conv2d_pointwise.wgsl",
    "conv_transpose2d.wgsl",
    # reductions
    "mean.wgsl",
    "sum.wgsl",
    # gather / indexing
    "gather.wgsl",
    "gather_dim.wgsl",
    # normalization
    "layernorm.wgsl",
    "batchnorm.wgsl",
    "groupnorm.wgsl",
    # pooling
    "max_pool2d.wgsl",
    "avg_pool2d.wgsl",
    "adaptive_avg_pool2d.wgsl",
    # upsampling / shuffling
    "upsample_nearest2d.wgsl",
    "upsample_bilinear2d.wgsl",
    "pixel_shuffle.wgsl",
    # shape / layout (reshape.wgsl and permute.wgsl also cover several aliased
    # aten ops with no data-movement difference — see their file headers)
    "reshape.wgsl",
    "permute.wgsl",
    "concat.wgsl",
    "slice.wgsl",
)


def load_kernels() -> dict[str, bytes]:
    """Return {filename: wgsl_source_bytes} for every kernel bundled with Kuma."""
    kernels_dir = resources.files("kuma.kernels")
    return {name: kernels_dir.joinpath(name).read_bytes() for name in _KERNEL_NAMES}
