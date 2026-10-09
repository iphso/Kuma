// aten.softmax.int / aten._softmax.default — softmax over one axis. The reduced axis
// is walked with stride `inner` (outer = product of dims before it, inner = product of
// dims after it, same factoring as sum.wgsl/mean.wgsl). One invocation per (outer,
// inner) row: three passes over the row (max, exp-sum, normalise), max-subtracted so
// large logits don't overflow, exactly as PyTorch computes it.

struct Params {
    outer: u32,
    reduce: u32,
    inner: u32,
    n: u32,  // outer * inner (row count = invocation count)
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= params.n) {
        return;
    }

    let inner_idx = idx % params.inner;
    let outer_idx = idx / params.inner;
    let base = outer_idx * params.reduce * params.inner + inner_idx;

    var m: f32 = input[base];
    for (var r: u32 = 1u; r < params.reduce; r = r + 1u) {
        m = max(m, input[base + r * params.inner]);
    }
    var s: f32 = 0.0;
    for (var r: u32 = 0u; r < params.reduce; r = r + 1u) {
        let e = exp(input[base + r * params.inner] - m);
        out[base + r * params.inner] = e;
        s = s + e;
    }
    let inv = 1.0 / s;
    for (var r: u32 = 0u; r < params.reduce; r = r + 1u) {
        out[base + r * params.inner] = out[base + r * params.inner] * inv;
    }
}
