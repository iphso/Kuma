// aten.addmm.default / aten.mm.default — y = x @ weight^T + bias.
// x: (M, K), weight: (N, K), bias: (N).
//
// Shared-memory-tiled matmul: one workgroup per 64x64 output tile, instead of the
// original one-thread-per-output-element version, which read x/weight straight from
// global memory inside its K-length loop -- every element of x got re-read from global
// memory N times (once per output column reusing that row) and every element of weight
// got re-read M times, with zero reuse. That's memory-bandwidth-bound, not
// compute-bound, which is exactly backwards for a matmul.
//
// Here, each workgroup loads one 64x16 tile of x and one 16x64 tile of weight into
// workgroup-shared memory *once*. Each of the 256 threads computes a 4x4 register
// micro-tile, so the same shared-memory values feed 16 FMAs per thread per K step
// instead of just one. This is the usual register-blocking step on top of shared-memory
// tiling: many fewer workgroups, much more arithmetic per global/shared load.
//
// Dispatch convention: callers must use OpContext.dispatchKernelGrid (not dispatchKernel)
// with grid = (ceil(N/64), ceil(M/64)) -- workgroup_id is read directly as 2D tile
// coordinates here, not folded into a linear index the way dispatchKernel's other
// kernels are. See ops/linear.ts's dispatchLinear.

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const BLOCK_M: u32 = 4u;
const BLOCK_N: u32 = 4u;

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> a_tile: array<f32, 1024>; // a_tile[row_in_tile*TILE_K+kk] = x[row][k_base+kk]
var<workgroup> b_tile: array<f32, 1024>; // b_tile[kk*TILE_N+col_in_tile] = weight[col][k_base+kk]

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tx = local_id.x;
    let ty = local_id.y;
    let row_base = wg_id.y * TILE_M + ty * BLOCK_M;
    let col_base = wg_id.x * TILE_N + tx * BLOCK_N;

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

    let num_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_base = t * TILE_K;

        for (var i: u32 = 0u; i < BLOCK_M; i = i + 1u) {
            let a_row_in_tile = ty * BLOCK_M + i;
            let a_row = row_base + i;
            let a_k = k_base + tx;
            var a_value: f32 = 0.0;
            if (a_row < params.m && a_k < params.k) {
                a_value = x[a_row * params.k + a_k];
            }
            a_tile[a_row_in_tile * TILE_K + tx] = a_value;
        }

        for (var i: u32 = 0u; i < BLOCK_N; i = i + 1u) {
            let b_col_in_tile = tx * BLOCK_N + i;
            let b_col = col_base + i;
            let b_k = k_base + ty;
            var b_value: f32 = 0.0;
            if (b_col < params.n && b_k < params.k) {
                b_value = weight[b_col * params.k + b_k];
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

    if (row0 < params.m) {
        if (col0 < params.n) { out[row0 * params.n + col0] = acc00 + bias[col0]; }
        if (col1 < params.n) { out[row0 * params.n + col1] = acc01 + bias[col1]; }
        if (col2 < params.n) { out[row0 * params.n + col2] = acc02 + bias[col2]; }
        if (col3 < params.n) { out[row0 * params.n + col3] = acc03 + bias[col3]; }
    }
    if (row1 < params.m) {
        if (col0 < params.n) { out[row1 * params.n + col0] = acc10 + bias[col0]; }
        if (col1 < params.n) { out[row1 * params.n + col1] = acc11 + bias[col1]; }
        if (col2 < params.n) { out[row1 * params.n + col2] = acc12 + bias[col2]; }
        if (col3 < params.n) { out[row1 * params.n + col3] = acc13 + bias[col3]; }
    }
    if (row2 < params.m) {
        if (col0 < params.n) { out[row2 * params.n + col0] = acc20 + bias[col0]; }
        if (col1 < params.n) { out[row2 * params.n + col1] = acc21 + bias[col1]; }
        if (col2 < params.n) { out[row2 * params.n + col2] = acc22 + bias[col2]; }
        if (col3 < params.n) { out[row2 * params.n + col3] = acc23 + bias[col3]; }
    }
    if (row3 < params.m) {
        if (col0 < params.n) { out[row3 * params.n + col0] = acc30 + bias[col0]; }
        if (col1 < params.n) { out[row3 * params.n + col1] = acc31 + bias[col1]; }
        if (col2 < params.n) { out[row3 * params.n + col2] = acc32 + bias[col2]; }
        if (col3 < params.n) { out[row3 * params.n + col3] = acc33 + bias[col3]; }
    }
}
