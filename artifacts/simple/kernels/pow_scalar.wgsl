// aten.pow.Tensor_Scalar — x ^ exponent, exponent baked into Params per call site.

struct Params {
    n: u32,
    exponent: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// WGSL's pow(x, y) is exp2(y * log2(x)) under the hood, undefined (NaN) for any x < 0
// regardless of y -- even though x^n for a negative x and integer n is a perfectly
// ordinary signed real number (e.g. (-2)^2 = 4). torch.pow matches real-number
// semantics: negative base ^ integer exponent is fine, negative base ^ non-integer
// exponent is NaN (the result would be complex) -- same as pow() already gives us, so
// only the integer-exponent/negative-base case needs special handling.
fn signed_pow(x: f32, exponent: f32) -> f32 {
    if (x < 0.0 && exponent == floor(exponent)) {
        let magnitude = pow(-x, exponent);
        let exponent_is_odd = (i32(exponent) & 1) != 0;
        return select(magnitude, -magnitude, exponent_is_odd);
    }
    return pow(x, exponent);
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * (num_wg.x * 64u);
    if (i >= params.n) {
        return;
    }
    out[i] = signed_pow(input[i], params.exponent);
}
