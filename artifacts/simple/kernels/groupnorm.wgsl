// aten.native_group_norm.default — normalize each (batch, group) block of `group_size` =
// channels_per_group * H * W contiguous elements, then apply per-channel affine.
// NCHW layout; weight/bias are length `channels`.
//
// One WORKGROUP per (batch, group) row (not one *thread* per row, as this kernel used
// to be written). group_size can be huge (channels_per_group * H * W) -- a single
// thread looping over it sequentially, three separate times (mean, variance,
// normalize), left nearly the entire GPU idle: e.g. batch=3, groups=3 means only 9
// threads total ever did anything, each doing ~3 * group_size sequential float ops.
// Measured at 49% of a whole model's per-frame GPU time on one call.
//
// Here, every row's 256 threads cooperatively sum (and later normalize) their own
// strided slice of the row, combining partial sums via workgroup-shared memory and a
// standard binary-tree reduction -- the row's mean/rstd become a per-workgroup
// broadcast value once computed, same as the original's per-thread locals were. 256
// (not 64) threads per row: group_size is typically large enough that even with 64
// threads each one still does thousands of sequential iterations per phase, and 256 is
// the largest workgroup size every WebGPU-conformant device is guaranteed to support
// without requesting extra limits (maxComputeInvocationsPerWorkgroup's spec-mandated
// minimum), so this isn't tuned to any one model's shapes.
//
// Dispatch convention: callers must pass dispatchElements = rows * 64 (not just
// `rows`) to OpContext.dispatchKernel -- the "64" there is dispatchKernel's own
// element-count-to-workgroup-count conversion factor (see engine/dispatch.ts), which is
// independent of this shader's own @workgroup_size; it always yields exactly `rows`
// workgroups regardless of this kernel's thread count. See kuma-bart's ops/norm.ts.

struct Params {
    batch: u32,
    channels: u32,
    spatial: u32,        // H * W
    groups: u32,
    channels_per_group: u32,
    eps: f32,
    rows: u32,           // batch * groups
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partials: array<f32, 256>;
var<workgroup> row_mean: f32;
var<workgroup> row_rstd: f32;

// Binary-tree reduction of this workgroup's 256 partial sums (already written into
// `partials` by the caller) down to partials[0]. Every thread reaches every
// workgroupBarrier() unconditionally (the `if` only guards the add, not the barrier),
// satisfying WGSL's uniformity requirement for barriers within non-divergent control
// flow -- `stride` is identical across all threads at every iteration.
fn tree_reduce(tid: u32) {
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            partials[tid] = partials[tid] + partials[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = wg_id.x + wg_id.y * num_wg.x;
    if (row >= params.rows) {
        return;
    }
    let tid = local_id.x;
    let b = row / params.groups;
    let g = row % params.groups;
    let group_size = params.channels_per_group * params.spatial;
    let base = (b * params.channels + g * params.channels_per_group) * params.spatial;

    // Mean: each thread sums every 256th element starting at its own tid, then the
    // workgroup's 256 partial sums get tree-reduced into one.
    var local_sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < group_size) {
        local_sum = local_sum + input[base + i];
        i = i + 256u;
    }
    partials[tid] = local_sum;
    workgroupBarrier();
    tree_reduce(tid);
    if (tid == 0u) {
        row_mean = partials[0] / f32(group_size);
    }
    workgroupBarrier();
    let mean = row_mean;

    // Variance: identical pattern, now that every thread has the row's mean.
    var local_sq: f32 = 0.0;
    i = tid;
    while (i < group_size) {
        let d = input[base + i] - mean;
        local_sq = local_sq + d * d;
        i = i + 256u;
    }
    partials[tid] = local_sq;
    workgroupBarrier();
    tree_reduce(tid);
    if (tid == 0u) {
        let variance = partials[0] / f32(group_size);
        row_rstd = inverseSqrt(variance + params.eps);
    }
    workgroupBarrier();
    let rstd = row_rstd;

    // Normalize + affine: every position in the row shares this row's mean/rstd; only
    // the per-channel weight/bias differ, via cg = (flat row index) / spatial.
    i = tid;
    while (i < group_size) {
        let cg = i / params.spatial;
        let c = g * params.channels_per_group + cg;
        let idx = base + i;
        out[idx] = (input[idx] - mean) * rstd * weight[c] + bias[c];
        i = i + 256u;
    }
}
