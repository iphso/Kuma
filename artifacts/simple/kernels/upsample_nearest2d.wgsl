// aten.upsample_nearest2d.vec — nearest-neighbor upsample. in_idx = floor(out_idx * in_size / out_size).
// NCHW layout; one invocation per output element.

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let total = params.batch * params.channels * params.out_h * params.out_w;
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= total) {
        return;
    }

    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    let b = idx / (params.out_w * params.out_h * params.channels);

    let iy = (oh * params.in_h) / params.out_h;
    let ix = (ow * params.in_w) / params.out_w;
    let in_idx = ((b * params.channels + c) * params.in_h + iy) * params.in_w + ix;
    out[idx] = input[in_idx];
}
