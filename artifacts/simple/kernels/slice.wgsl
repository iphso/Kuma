// aten.slice.Tensor — extract [start : start + out_extent * step : step] along one axis.
// outer = product of dims before the sliced axis, inner = product of dims after it.
// Also covers aten.select.int (out_extent=1, step=1 — the selected dim is then dropped by
// a free reshape, since a length-1 axis never affects the contiguous memory layout).

struct Params {
    outer: u32,
    inner: u32,
    in_extent: u32,
    out_extent: u32,
    start: u32,
    step: u32,
    n: u32,
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
    let tmp = idx / params.inner;
    let slice_idx = tmp % params.out_extent;
    let outer_idx = tmp / params.out_extent;

    let in_slice_idx = params.start + slice_idx * params.step;
    let in_offset = (outer_idx * params.in_extent + in_slice_idx) * params.inner + inner_idx;
    out[idx] = input[in_offset];
}
