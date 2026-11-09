// Widens an int8 weight tensor (kuma.quantize, scheme int8_symmetric) to float32 on the
// GPU: out[i] = f32(q[i]) * scales[i / group_size]. kuma-bart runs this for a consumer
// that has no int8 kernel (any op other than linear / pointwise / implicit-GEMM conv), so
// every op works on a quantized package; the hot ones read int8 directly instead.
// q is four int8 per u32, little-endian.

struct Params {
    n: u32,
    group_size: u32,
};

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * (nwg.x * 64u);
    if (i >= params.n) {
        return;
    }
    let q = extractBits(i32(weight[i >> 2u]), (i & 3u) * 8u, 8u);
    out[i] = f32(q) * scales[i / params.group_size];
}
