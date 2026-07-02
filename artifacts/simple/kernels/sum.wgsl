// aten.sum.dim_IntList — sum over a contiguous block of `reduce` elements (the reduced
// axes must be contiguous in memory, e.g. trailing dims or H,W in NCHW).
// outer = product of dims before the reduced axes, inner = product of dims after them.

struct Params {
    outer: u32,
    reduce: u32,
    inner: u32,
    n: u32,  // outer * inner (output element count)
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

    var acc: f32 = 0.0;
    let base = outer_idx * params.reduce * params.inner + inner_idx;
    for (var r: u32 = 0u; r < params.reduce; r = r + 1u) {
        acc = acc + input[base + r * params.inner];
    }
    out[idx] = acc;
}
