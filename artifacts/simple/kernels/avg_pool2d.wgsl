// aten.avg_pool2d.default — average pooling, count_include_pad=True (PyTorch's default):
// always divides by kh*kw regardless of how much of the window falls in padding.
// NCHW layout; one invocation per output element.

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
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

    var acc: f32 = 0.0;
    for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
        let iy_signed = i32(oh * params.stride_h + ky) - i32(params.pad_h);
        if (iy_signed < 0 || iy_signed >= i32(params.in_h)) {
            continue;
        }
        let iy = u32(iy_signed);
        for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
            let ix_signed = i32(ow * params.stride_w + kx) - i32(params.pad_w);
            if (ix_signed < 0 || ix_signed >= i32(params.in_w)) {
                continue;
            }
            let ix = u32(ix_signed);
            let in_idx = ((b * params.channels + c) * params.in_h + iy) * params.in_w + ix;
            acc = acc + input[in_idx];
        }
    }
    out[idx] = acc / f32(params.kh * params.kw);
}
