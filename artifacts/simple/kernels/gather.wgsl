// aten.index.Tensor — gather rows along one axis using a 1D index tensor:
// out[outer, g, inner] = input[outer, indices[g], inner].
// Indices are stored as f32-encoded integers (WGSL has no i64; values here are small
// static indices, e.g. frame positions, safe to truncate to u32 at use).
// outer = product of dims before the gathered axis, inner = product of dims after it.

struct Params {
    outer: u32,
    inner: u32,
    in_extent: u32,
    num_indices: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

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
    let tmp = idx / params.inner;
    let gather_idx = tmp % params.num_indices;
    let outer_idx = tmp / params.num_indices;

    let src_row = u32(indices[gather_idx]);
    let in_offset = (outer_idx * params.in_extent + src_row) * params.inner + inner_idx;
    out[idx] = input[in_offset];
}
