// aten._adaptive_avg_pool2d.default — adaptive average pooling; window bounds follow
// PyTorch's formula: start = floor(o * in / out), end = ceil((o + 1) * in / out).
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

    let h_start = (oh * params.in_h) / params.out_h;
    let h_end = ((oh + 1u) * params.in_h + params.out_h - 1u) / params.out_h;
    let w_start = (ow * params.in_w) / params.out_w;
    let w_end = ((ow + 1u) * params.in_w + params.out_w - 1u) / params.out_w;

    var acc: f32 = 0.0;
    var count: u32 = 0u;
    for (var iy: u32 = h_start; iy < h_end; iy = iy + 1u) {
        for (var ix: u32 = w_start; ix < w_end; ix = ix + 1u) {
            let in_idx = ((b * params.channels + c) * params.in_h + iy) * params.in_w + ix;
            acc = acc + input[in_idx];
            count = count + 1u;
        }
    }
    out[idx] = acc / f32(count);
}
