// aten.convolution.default / aten.conv2d.default — depthwise conv2d (in_channels_per_group
// == 1: every output channel depends on exactly one input channel, no cross-channel
// reduction at all). Weight layout: (out_channels, 1, kH, kW). Bias optional.
//
// Depthwise has no weight-reuse-across-output-channels redundancy to exploit (that's
// what made the earlier weight-caching attempt at the general conv2d.wgsl kernel a net
// loss -- see its header) -- the real redundancy here is *spatial*: neighboring output
// pixels' receptive fields overlap, so the same input pixel gets re-read from global
// memory up to kh*kw times by different output pixels in the old direct kernel.
//
// One workgroup per (batch, out_channel), computing a 16x16 output spatial tile.
// Cooperatively loads the (15*stride_h+kh) x (15*stride_w+kw) input patch -- the output
// tile plus the halo needed for the kh x kw taps, zero-filled at padded/out-of-bounds
// positions -- into workgroup-shared memory once. Every thread then computes its own
// output pixel from kh*kw taps read out of that cached patch, with no further global
// memory reads of `input` at all. Weight (kh*kw, typically <=49 elements) is read
// directly from global memory -- small enough that caching it isn't worth the added
// complexity (the same lesson from the reverted weight-cache attempt).
//
// Dispatch convention: callers must use OpContext.dispatchKernelGrid with grid =
// (ceil(out_w/16), ceil(out_h/16), batch*out_channels). See ops/conv2d.ts.
//
// PATCH_CAPACITY (4096 floats = 16KB) is WebGPU's guaranteed-minimum
// maxComputeWorkgroupStorageSize. The halo patch size depends on stride/kh/kw, so
// ops/conv2d.ts computes it first and falls back to the general conv2d.wgsl kernel
// (never silently wrong, never throws) when it would exceed this -- e.g. an unusually
// large stride combined with a large kernel. Not tuned to this model's 3x3/7x7 kernels.

const TILE: u32 = 16u;
const THREADS: u32 = TILE * TILE;
const PATCH_CAPACITY: u32 = 4096u;

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

// "patch" is a reserved WGSL identifier (reserved for a possible future language
// feature, even though nothing currently uses it) -- "cache" instead.
var<workgroup> cache: array<f32, 4096>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let oc = wg_id.z % params.out_channels;
    let b = wg_id.z / params.out_channels;
    let out_channels_per_group = params.out_channels / params.groups;
    let ic = oc / out_channels_per_group;

    let oy0 = wg_id.y * TILE;
    let ox0 = wg_id.x * TILE;
    // Signed: padding can put the patch's first row/col before the image starts (a
    // negative index), which a u32 subtraction would silently wrap into a huge value.
    let iy_base = i32(oy0 * params.stride_h) - i32(params.pad_h);
    let ix_base = i32(ox0 * params.stride_w) - i32(params.pad_w);

    // Uniform across the workgroup (depends only on wg_id/global params, not
    // local_id) -- every thread takes the same number of loop iterations below and
    // reaches the same barrier.
    let patch_h = (TILE - 1u) * params.stride_h + params.kh;
    let patch_w = (TILE - 1u) * params.stride_w + params.kw;
    let patch_size = patch_h * patch_w;

    let tid = local_id.y * TILE + local_id.x;
    var i: u32 = tid;
    while (i < patch_size) {
        let py = i32(i / patch_w);
        let px = i32(i % patch_w);
        let iy = iy_base + py;
        let ix = ix_base + px;
        var v: f32 = 0.0;
        if (iy >= 0 && iy < i32(params.in_h) && ix >= 0 && ix < i32(params.in_w)) {
            v = input[((b * params.in_channels + ic) * params.in_h + u32(iy)) * params.in_w + u32(ix)];
        }
        cache[i] = v;
        i = i + THREADS;
    }
    workgroupBarrier();

    // From here on, oh/ow/b vary per-thread -- safe to diverge (e.g. return early)
    // since there are no more barriers downstream for that divergence to violate.
    let oh = oy0 + local_id.y;
    let ow = ox0 + local_id.x;
    if (oh >= params.out_h || ow >= params.out_w || b >= params.batch) {
        return;
    }

    // The patch is already zero-padded by construction, so this loop needs no bounds
    // checks at all -- a nice simplification over the direct kernel's per-tap checks.
    var acc: f32 = bias[oc];
    for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
        let prow = local_id.y * params.stride_h + ky;
        for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
            let pcol = local_id.x * params.stride_w + kx;
            acc = acc + cache[prow * patch_w + pcol] * weight[oc * params.kh * params.kw + ky * params.kw + kx];
        }
    }

    let out_idx = ((b * params.out_channels + oc) * params.out_h + oh) * params.out_w + ow;
    out[out_idx] = acc;
}
