// aten.convolution.default — direct NCHW conv2d. One invocation per output element.
// Weight layout: (out_channels, in_channels / groups, kH, kW). Bias is optional (zero-filled when absent).
//
// Tried caching the per-(batch,out_channel) weight slice in workgroup-shared memory
// here (one workgroup per 16x16 output tile) -- measured *slower* on the real target
// model, not faster. Diagnosis: every conv2d call there has in_channels_per_group*kh*kw
// of just 9-48 (8 of 13 calls are 1x1 pointwise convs, in_channels_per_group up to 48;
// the other 5 are depthwise, in_channels_per_group=1, kh/kw up to 7) -- small enough
// that the GPU's automatic L1/L2 cache almost certainly already served those reads
// efficiently across threads, for free. The explicit barrier + cooperative shared-memory
// load added real cost with no corresponding benefit. Reverted to this straightforward
// version rather than carry that regression.
//
// The real redundancy is elsewhere, and differs by conv type: pointwise (1x1) convs are
// really a per-pixel channel-mixing matmul, where the redundancy is *input* reuse across
// output channels (not weight reuse); depthwise convs have no cross-channel work at all
// (in_channels_per_group=1), so the only real lever there is the overlap between
// neighboring output pixels' receptive fields, which needs halo-aware input tiling.
// Both are bigger, properly-scoped follow-ups, not a quick re-tune of this kernel.

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
    let ic_base = group * in_channels_per_group;

    var acc: f32 = bias[oc];

    for (var icg: u32 = 0u; icg < in_channels_per_group; icg = icg + 1u) {
        let ic = ic_base + icg;
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

                let in_idx = ((b * params.in_channels + ic) * params.in_h + iy) * params.in_w + ix;
                let w_idx = ((oc * in_channels_per_group + icg) * params.kh + ky) * params.kw + kx;
                acc = acc + input[in_idx] * weight[w_idx];
            }
        }
    }

    out[idx] = acc;
}
