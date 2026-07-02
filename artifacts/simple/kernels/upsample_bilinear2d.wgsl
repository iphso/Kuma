// aten.upsample_bilinear2d.vec — bilinear upsample, align_corners=False (PyTorch default):
// src = (out_idx + 0.5) * (in_size / out_size) - 0.5, clamped to >= 0.
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

fn src_coord(o: u32, in_size: u32, out_size: u32) -> vec2<f32> {
    // .x = floor(src) coordinate, .y = fractional lerp weight
    let scale = f32(in_size) / f32(out_size);
    let src = max((f32(o) + 0.5) * scale - 0.5, 0.0);
    let lo = floor(src);
    return vec2<f32>(lo, src - lo);
}

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

    let hy = src_coord(oh, params.in_h, params.out_h);
    let hx = src_coord(ow, params.in_w, params.out_w);

    let y0 = u32(hy.x);
    let x0 = u32(hx.x);
    let y1 = min(y0 + 1u, params.in_h - 1u);
    let x1 = min(x0 + 1u, params.in_w - 1u);
    let wy = hy.y;
    let wx = hx.y;

    let base = (b * params.channels + c) * params.in_h;
    let v00 = input[(base + y0) * params.in_w + x0];
    let v01 = input[(base + y0) * params.in_w + x1];
    let v10 = input[(base + y1) * params.in_w + x0];
    let v11 = input[(base + y1) * params.in_w + x1];

    let top = v00 + (v01 - v00) * wx;
    let bottom = v10 + (v11 - v10) * wx;
    out[idx] = top + (bottom - top) * wy;
}
