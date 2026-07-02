// aten.pixel_shuffle.default — (B, C*r*r, H, W) -> (B, C, H*r, W*r).
// in_channel = c*r*r + (oh % r) * r + (ow % r); in_h = oh / r; in_w = ow / r.

struct Params {
    batch: u32,
    out_channels: u32,    // C
    in_h: u32,
    in_w: u32,
    upscale_factor: u32,  // r
    out_h: u32,            // in_h * r
    out_w: u32,            // in_w * r
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let total = params.batch * params.out_channels * params.out_h * params.out_w;
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= total) {
        return;
    }

    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.out_channels;
    let b = idx / (params.out_w * params.out_h * params.out_channels);

    let r = params.upscale_factor;
    let iy = oh / r;
    let ix = ow / r;
    let in_c = c * r * r + (oh % r) * r + (ow % r);
    let in_channels = params.out_channels * r * r;

    let in_idx = ((b * in_channels + in_c) * params.in_h + iy) * params.in_w + ix;
    out[idx] = input[in_idx];
}
