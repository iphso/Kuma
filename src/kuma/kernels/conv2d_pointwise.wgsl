// aten.conv2d.default / aten.convolution.default — pointwise (1x1 kernel, groups=1)
// conv2d as a strided GEMM without materializing an NHWC/im2col buffer.
//
// Matrix view:
//   A(row=(b,oh,ow), k=ic) = input[b, ic, oh*stride_h-pad_h, ow*stride_w-pad_w]
//   B(k=ic, col=oc)        = weight[oc, ic]
//   C(row, col=oc)         = out[b, oc, oh, ow]
//
// One workgroup computes a 64x64 output tile with a 16-wide K tile. Each thread owns a
// 4x4 register micro-tile, matching linear.wgsl/bmm.wgsl's reuse pattern while keeping
// the runtime memory footprint to the single real NCHW output buffer.

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const BLOCK_M: u32 = 4u;
const BLOCK_N: u32 = 4u;

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>; // (out_channels, in_channels) -- kh=kw=1, flattened
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> a_tile: array<f32, 1024>; // a_tile[row_in_tile*TILE_K+ic_in_tile]
var<workgroup> b_tile: array<f32, 1024>; // b_tile[ic_in_tile*TILE_N+oc_in_tile]

fn input_value(row: u32, ic: u32) -> f32 {
    let spatial = params.out_h * params.out_w;
    let b = row / spatial;
    let rem = row - b * spatial;
    let oh = rem / params.out_w;
    let ow = rem - oh * params.out_w;

    let iy = i32(oh * params.stride_h) - i32(params.pad_h);
    let ix = i32(ow * params.stride_w) - i32(params.pad_w);
    if (b >= params.batch || ic >= params.in_channels || iy < 0 || iy >= i32(params.in_h) || ix < 0 || ix >= i32(params.in_w)) {
        return 0.0;
    }

    return input[((b * params.in_channels + ic) * params.in_h + u32(iy)) * params.in_w + u32(ix)];
}

fn write_output(row: u32, oc: u32, value: f32) {
    let spatial = params.out_h * params.out_w;
    let b = row / spatial;
    let rem = row - b * spatial;
    let oh = rem / params.out_w;
    let ow = rem - oh * params.out_w;
    out[((b * params.out_channels + oc) * params.out_h + oh) * params.out_w + ow] = value + bias[oc];
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tx = local_id.x;
    let ty = local_id.y;
    let row_base = wg_id.y * TILE_M + ty * BLOCK_M;
    let col_base = wg_id.x * TILE_N + tx * BLOCK_N;
    let total_rows = params.batch * params.out_h * params.out_w;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc02: f32 = 0.0;
    var acc03: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;
    var acc12: f32 = 0.0;
    var acc13: f32 = 0.0;
    var acc20: f32 = 0.0;
    var acc21: f32 = 0.0;
    var acc22: f32 = 0.0;
    var acc23: f32 = 0.0;
    var acc30: f32 = 0.0;
    var acc31: f32 = 0.0;
    var acc32: f32 = 0.0;
    var acc33: f32 = 0.0;

    let num_tiles = (params.in_channels + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_base = t * TILE_K;

        for (var i: u32 = 0u; i < BLOCK_M; i = i + 1u) {
            let a_row_in_tile = ty * BLOCK_M + i;
            a_tile[a_row_in_tile * TILE_K + tx] = input_value(row_base + i, k_base + tx);
        }

        for (var i: u32 = 0u; i < BLOCK_N; i = i + 1u) {
            let b_col_in_tile = tx * BLOCK_N + i;
            let oc = col_base + i;
            let ic = k_base + ty;
            var b_value: f32 = 0.0;
            if (oc < params.out_channels && ic < params.in_channels) {
                b_value = weight[oc * params.in_channels + ic];
            }
            b_tile[ty * TILE_N + b_col_in_tile] = b_value;
        }

        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
            let a0 = a_tile[(ty * BLOCK_M + 0u) * TILE_K + kk];
            let a1 = a_tile[(ty * BLOCK_M + 1u) * TILE_K + kk];
            let a2 = a_tile[(ty * BLOCK_M + 2u) * TILE_K + kk];
            let a3 = a_tile[(ty * BLOCK_M + 3u) * TILE_K + kk];
            let b0 = b_tile[kk * TILE_N + tx * BLOCK_N + 0u];
            let b1 = b_tile[kk * TILE_N + tx * BLOCK_N + 1u];
            let b2 = b_tile[kk * TILE_N + tx * BLOCK_N + 2u];
            let b3 = b_tile[kk * TILE_N + tx * BLOCK_N + 3u];

            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc02 = acc02 + a0 * b2;
            acc03 = acc03 + a0 * b3;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
            acc12 = acc12 + a1 * b2;
            acc13 = acc13 + a1 * b3;
            acc20 = acc20 + a2 * b0;
            acc21 = acc21 + a2 * b1;
            acc22 = acc22 + a2 * b2;
            acc23 = acc23 + a2 * b3;
            acc30 = acc30 + a3 * b0;
            acc31 = acc31 + a3 * b1;
            acc32 = acc32 + a3 * b2;
            acc33 = acc33 + a3 * b3;
        }

        workgroupBarrier();
    }

    let row0 = row_base + 0u;
    let row1 = row_base + 1u;
    let row2 = row_base + 2u;
    let row3 = row_base + 3u;
    let col0 = col_base + 0u;
    let col1 = col_base + 1u;
    let col2 = col_base + 2u;
    let col3 = col_base + 3u;

    if (row0 < total_rows) {
        if (col0 < params.out_channels) { write_output(row0, col0, acc00); }
        if (col1 < params.out_channels) { write_output(row0, col1, acc01); }
        if (col2 < params.out_channels) { write_output(row0, col2, acc02); }
        if (col3 < params.out_channels) { write_output(row0, col3, acc03); }
    }
    if (row1 < total_rows) {
        if (col0 < params.out_channels) { write_output(row1, col0, acc10); }
        if (col1 < params.out_channels) { write_output(row1, col1, acc11); }
        if (col2 < params.out_channels) { write_output(row1, col2, acc12); }
        if (col3 < params.out_channels) { write_output(row1, col3, acc13); }
    }
    if (row2 < total_rows) {
        if (col0 < params.out_channels) { write_output(row2, col0, acc20); }
        if (col1 < params.out_channels) { write_output(row2, col1, acc21); }
        if (col2 < params.out_channels) { write_output(row2, col2, acc22); }
        if (col3 < params.out_channels) { write_output(row2, col3, acc23); }
    }
    if (row3 < total_rows) {
        if (col0 < params.out_channels) { write_output(row3, col0, acc30); }
        if (col1 < params.out_channels) { write_output(row3, col1, acc31); }
        if (col2 < params.out_channels) { write_output(row3, col2, acc32); }
        if (col3 < params.out_channels) { write_output(row3, col3, acc33); }
    }
}
