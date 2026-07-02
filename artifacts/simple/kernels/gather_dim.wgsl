// aten.gather.default — out[idx] = input[idx with the gathered axis replaced by
// index[idx]]. `index` has the same shape as `out`. Rank <= 8, split across two
// vec4<u32> pairs (WGSL vectors cap at 4 components, and a plain array<u32,N> in a
// uniform buffer would get padded to a 16-byte stride per element, breaking tight
// packing — two vec4s sidestep both issues while keeping dynamic-index support).
// The caller pre-zeroes in_strides[gather_axis] so that axis's contribution drops
// out of the loop below, with the looked-up index contributing via gather_stride.

struct Params {
    out_shape_lo: vec4<u32>,
    out_shape_hi: vec4<u32>,
    in_strides_lo: vec4<u32>,
    in_strides_hi: vec4<u32>,
    gather_stride: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> index: array<f32>;
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
    var in_offset: u32 = 0u;
    for (var d: i32 = 7; d >= 0; d = d - 1) {
        var extent: u32;
        var stride: u32;
        if (d < 4) {
            extent = params.out_shape_lo[d];
            stride = params.in_strides_lo[d];
        } else {
            extent = params.out_shape_hi[d - 4];
            stride = params.in_strides_hi[d - 4];
        }
        let coord = rem % extent;
        rem = rem / extent;
        in_offset = in_offset + coord * stride;
    }

    let gathered = u32(index[i]);
    in_offset = in_offset + gathered * params.gather_stride;
    out[i] = input[in_offset];
}
