// aten.clamp.default — clamp(x, min_val, max_val). Pass -inf/+inf for an unbounded side
// (also covers aten.hardtanh.default / nn.ReLU6 via min_val=0, max_val=6).

struct Params {
    n: u32,
    min_val: f32,
    max_val: f32,
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
    out[i] = clamp(input[i], params.min_val, params.max_val);
}
