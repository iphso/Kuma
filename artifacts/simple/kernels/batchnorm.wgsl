// aten.native_batch_norm.default / aten._native_batch_norm_legit_no_training.default
// (inference only) — per-channel affine using running statistics.
// NCHW layout; channel = dim 1. One invocation per output element.

struct Params {
    channels: u32,
    spatial: u32,  // H * W
    eps: f32,
    n: u32,        // batch * channels * spatial
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> running_mean: array<f32>;
@group(0) @binding(4) var<storage, read> running_var: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= params.n) {
        return;
    }
    let c = (idx / params.spatial) % params.channels;
    let rstd = inverseSqrt(running_var[c] + params.eps);
    out[idx] = (input[idx] - running_mean[c]) * rstd * weight[c] + bias[c];
}
