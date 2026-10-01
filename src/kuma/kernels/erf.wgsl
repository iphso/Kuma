// aten.erf.default — error function (A&S 7.1.26, abs error <= 1.5e-7).

struct Params {
    n: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26: |error| <= 1.5e-7 over all x (below f32 epsilon
    // near |erf| = 1). Odd extension for x < 0. Shared verbatim by erf.wgsl and
    // gelu_erf.wgsl -- keep the two copies in sync.
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * exp(-a * a);
    return select(y, -y, x < 0.0);
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
    out[i] = erf_approx(input[i]);
}
