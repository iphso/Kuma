// aten.add/sub/mul/div.Tensor and aten.minimum.default with numpy broadcasting done by
// strided reads: `a` and `b` are read through per-axis strides (0 on a broadcast axis), so
// a scalar literal is a 1-element buffer and a [1,C,1,1] scale stays [1,C,1,1]. Replaces
// materializing each operand at the output shape first (a cached full-size constant per
// scalar literal, a full-size temporary per broadcast), which cost the terrain decoder
// 5.5 GB of cached constants at 512x512x128. Same f32 operation per element as
// add/sub/mul/div/minimum.wgsl, so results are bit-identical to them.
//
// op: 0 add, 1 sub, 2 mul, 3 div, 4 minimum. Rank <= 4, left-padded with size-1 axes.

struct Params {
    out_shape: vec4<u32>,
    a_strides: vec4<u32>,
    b_strides: vec4<u32>,
    n: u32,
    op: u32,
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
    let i = gid.x + gid.y * (num_wg.x * 64u);
    if (i >= params.n) {
        return;
    }

    var rem = i;
    var ai: u32 = 0u;
    var bi: u32 = 0u;
    for (var d: i32 = 3; d >= 0; d = d - 1) {
        let extent = params.out_shape[d];
        let coord = rem % extent;
        rem = rem / extent;
        ai = ai + coord * params.a_strides[d];
        bi = bi + coord * params.b_strides[d];
    }

    let x = a[ai];
    let y = b[bi];
    var r: f32;
    switch params.op {
        case 0u: { r = x + y; }
        case 1u: { r = x - y; }
        case 2u: { r = x * y; }
        case 3u: { r = x / y; }
        default: { r = min(x, y); }
    }
    out[i] = r;
}
