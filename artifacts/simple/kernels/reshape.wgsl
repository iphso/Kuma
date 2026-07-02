// aten.view.default / aten.reshape.default — contiguous reshape is a straight buffer copy.
// Also covers aten.flatten.using_ints, aten.squeeze.dim, aten.unsqueeze.default, and
// aten.clone.default — all no-op data movement for a static-shape, non-broadcast graph.

struct Params {
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
    let i = gid.x + gid.y * (num_wg.x * 64u);
    if (i >= params.n) {
        return;
    }
    out[i] = input[i];
}
