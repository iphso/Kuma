// aten.permute.default / aten.transpose.int — strided gather, rank <= 4 (pad unused dims with extent 1, stride 0).
// out_shape is the permuted tensor's shape; in_strides[d] is the source buffer stride for output axis d.
// Also covers aten.t.default (2D transpose) and aten.expand.default (broadcast — set the
// expanded axis's in_strides entry to 0 so every output position reads the same source element).

struct Params {
    out_shape: vec4<u32>,
    in_strides: vec4<u32>,
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

    var rem = i;
    var in_offset: u32 = 0u;
    for (var d: i32 = 3; d >= 0; d = d - 1) {
        let extent = params.out_shape[d];
        let coord = rem % extent;
        rem = rem / extent;
        in_offset = in_offset + coord * params.in_strides[d];
    }

    out[i] = input[in_offset];
}
