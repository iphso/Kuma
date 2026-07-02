// aten.cat.default — two-input concat along one axis (N-way concat lowers to a chain of these).
// outer = product of dims before the concat axis, inner = product of dims after it.

struct Params {
    outer: u32,
    inner: u32,
    size_a: u32,
    size_b: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
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
    let concat_size = params.size_a + params.size_b;
    let concat_idx = tmp % concat_size;
    let outer_idx = tmp / concat_size;

    if (concat_idx < params.size_a) {
        out[idx] = a[(outer_idx * params.size_a + concat_idx) * params.inner + inner_idx];
    } else {
        out[idx] = b[(outer_idx * params.size_b + (concat_idx - params.size_a)) * params.inner + inner_idx];
    }
}
