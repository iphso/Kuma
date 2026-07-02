// aten.convolution.default (transposed=True) — direct ConvTranspose2d via input-gather.
// Weight layout: (in_channels, out_channels / groups, kH, kW). Bias optional (zero-filled when absent).
// No output_padding, dilation=1 (matches every ConvTranspose2d usage in the target model families).

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    groups: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

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
    let oc = (idx / (params.out_w * params.out_h)) % params.out_channels;
    let b = idx / (params.out_w * params.out_h * params.out_channels);

    let in_channels_per_group = params.in_channels / params.groups;
    let out_channels_per_group = params.out_channels / params.groups;
    let group = oc / out_channels_per_group;
    let oc_in_group = oc % out_channels_per_group;
    let ic_base = group * in_channels_per_group;

    var acc: f32 = bias[oc];

    for (var icg: u32 = 0u; icg < in_channels_per_group; icg = icg + 1u) {
        let ic = ic_base + icg;
        for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
            let iy_num = i32(oh + params.pad_h) - i32(ky);
            if (iy_num < 0 || iy_num % i32(params.stride_h) != 0) {
                continue;
            }
            let iy = u32(iy_num) / params.stride_h;
            if (iy >= params.in_h) {
                continue;
            }
            for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
                let ix_num = i32(ow + params.pad_w) - i32(kx);
                if (ix_num < 0 || ix_num % i32(params.stride_w) != 0) {
                    continue;
                }
                let ix = u32(ix_num) / params.stride_w;
                if (ix >= params.in_w) {
                    continue;
                }

                let in_idx = ((b * params.in_channels + ic) * params.in_h + iy) * params.in_w + ix;
                let w_idx = ((ic * out_channels_per_group + oc_in_group) * params.kh + ky) * params.kw + kx;
                acc = acc + input[in_idx] * weight[w_idx];
            }
        }
    }

    out[idx] = acc;
}
