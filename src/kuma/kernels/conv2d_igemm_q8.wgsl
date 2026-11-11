// aten.conv2d.default / aten.convolution.default -- dense (groups=1) conv2d of any kernel
// size as an implicit GEMM with int8 weights: conv2d_igemm.wgsl with the weight tile
// loaded from int8.
//
// Weights are int8 (kuma.quantize, scheme int8_symmetric): four per u32, little-endian,
// row-major like the float32 tensor they replace, with one f32 scale per output channel.
// Each weight is dequantized as f32(q) * scale[oc] while filling b_tile (load_b_tile) --
// the same single f32 multiply a loader widening to float32 does -- so this kernel
// computes bit-for-bit what the float32 kernel computes on the widened weights, reading
// 1 byte per weight instead of 4. Uses only core WGSL (extractBits on i32 sign-extends); no f16/int8
// extensions.

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
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>; // int8 (out_channels, K) row-major, 4 per word
@group(0) @binding(2) var<storage, read> scales: array<f32>; // (out_channels)
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> a_tile: array<f32, 1024>; // a_tile[row_in_tile*TILE_K+k_in_tile]
var<workgroup> b_tile: array<f32, 1024>; // b_tile[k_in_tile*TILE_N+oc_in_tile]

// int8 weight element i (sign-extended), four per u32, little-endian byte order.
fn weight_q8(i: u32) -> f32 {
    return f32(extractBits(i32(weight[i >> 2u]), (i & 3u) * 8u, 8u));
}

fn input_value(row: u32, k: u32) -> f32 {
    let khw = params.kh * params.kw;
    let spatial = params.out_h * params.out_w;
    let b = row / spatial;
    let rem = row - b * spatial;
    let oh = rem / params.out_w;
    let ow = rem - oh * params.out_w;
    let ic = k / khw;
    let r = k - ic * khw;
    let ky = r / params.kw;
    let kx = r - ky * params.kw;

    let iy = i32(oh * params.stride_h + ky) - i32(params.pad_h);
    let ix = i32(ow * params.stride_w + kx) - i32(params.pad_w);
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


// Fills b_tile for the K tile starting at k_base: b_tile[kk * TILE_N + c] =
// f32(q[col0 + c, k_base + kk]) * scales[col0 + c] (0 outside the matrix), q being
// (n_cols, k_len) int8 row-major. When k_len is a multiple of 4, each of the 256 threads
// reads one u32 -- four consecutive k of one column -- instead of four separate bytes;
// adjacent threads take adjacent columns, so the b_tile stores don't bank-conflict.
fn load_b_tile(lid: u32, col0: u32, k_base: u32, n_cols: u32, k_len: u32) {
    if ((k_len & 3u) == 0u) {
        let c = lid & 63u;
        let kk = (lid >> 6u) * 4u;
        let col = col0 + c;
        let k = k_base + kk;
        var v = vec4<f32>(0.0);
        if (col < n_cols && k < k_len) {
            let word = i32(weight[(col * k_len + k) >> 2u]);
            let q = vec4<f32>(
                f32(extractBits(word, 0u, 8u)),
                f32(extractBits(word, 8u, 8u)),
                f32(extractBits(word, 16u, 8u)),
                f32(extractBits(word, 24u, 8u)),
            );
            v = q * scales[col];
        }
        b_tile[(kk + 0u) * TILE_N + c] = v.x;
        b_tile[(kk + 1u) * TILE_N + c] = v.y;
        b_tile[(kk + 2u) * TILE_N + c] = v.z;
        b_tile[(kk + 3u) * TILE_N + c] = v.w;
    } else {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let e = lid + j * 256u;
            let c = e & 63u;
            let kk = e >> 6u;
            let col = col0 + c;
            let k = k_base + kk;
            var b_value: f32 = 0.0;
            if (col < n_cols && k < k_len) {
                b_value = weight_q8(col * k_len + k) * scales[col];
            }
            b_tile[kk * TILE_N + c] = b_value;
        }
    }
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

    let k_total = params.in_channels * params.kh * params.kw;
    let num_tiles = (k_total + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_base = t * TILE_K;

        for (var i: u32 = 0u; i < BLOCK_M; i = i + 1u) {
            let a_row_in_tile = ty * BLOCK_M + i;
            a_tile[a_row_in_tile * TILE_K + tx] = input_value(row_base + i, k_base + tx);
        }

        load_b_tile(ty * 16u + tx, wg_id.x * TILE_N, k_base, params.out_channels, k_total);

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
