// aten.native_layer_norm.default — normalize over the trailing `norm_size` elements per row.
// weight/bias are per-element affine params of length norm_size (broadcast across rows).
// One invocation per row (not per element) — a simple direct implementation, matching the
// rest of this Step-1 kernel set, with no cross-invocation reduction.

struct Params {
    rows: u32,
    norm_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let row = gid.x + gid.y * (num_wg.x * 64u);
    if (row >= params.rows) {
        return;
    }
    let base = row * params.norm_size;

    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.norm_size; i = i + 1u) {
        mean = mean + input[base + i];
    }
    mean = mean / f32(params.norm_size);

    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < params.norm_size; i = i + 1u) {
        let d = input[base + i] - mean;
        variance = variance + d * d;
    }
    variance = variance / f32(params.norm_size);

    let rstd = inverseSqrt(variance + params.eps);
    for (var i: u32 = 0u; i < params.norm_size; i = i + 1u) {
        let normed = (input[base + i] - mean) * rstd;
        out[base + i] = normed * weight[i] + bias[i];
    }
}
