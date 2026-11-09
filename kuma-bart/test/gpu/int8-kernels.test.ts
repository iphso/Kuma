import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { KumaModel } from "../../src/index.js";
import { GraphBuilder } from "../graph-builder.js";
import { buildIph, pattern } from "../quant-fixture.js";

/**
 * The int8 kernels (linear_q8, conv2d_pointwise_q8, conv2d_igemm_q8) and the widening
 * fallback (dequant_q8 before conv2d_depthwise) on the real GPU, on a synthetic package:
 * a model loaded with int8 weights kept on the GPU must give exactly the outputs of the
 * same package widened to float32 on load (the int8 kernels dequantize with the same f32
 * multiply, then run the float32 kernels' arithmetic). Shapes cover K not a multiple of
 * 4 (the per-byte load path) and a multiple of 4 (one u32 per thread), more than one
 * 64-wide tile in M and N, and a bias.
 */

const KERNEL_DIR = fileURLToPath(new URL("../../../src/kuma/kernels/", import.meta.url));
const kernels = new Map(readdirSync(KERNEL_DIR).filter((f) => f.endsWith(".wgsl")).map((f) => [f, readFileSync(KERNEL_DIR + f, "utf8")]));

function model() {
  const g = new GraphBuilder();
  const values = new Map<string, Float32Array>();
  const w = (name: string, shape: number[], phase: number) => {
    values.set(name, pattern(shape.reduce((a, b) => a * b, 1), phase, 0.5));
    return g.weight(name, shape);
  };
  const H = 12;
  const x3 = g.input("x3", [1, 3, H, H]);
  const x4 = g.input("x4", [1, 4, H, H]);
  const x6 = g.input("x6", [1, 6, H, H]);
  const x8 = g.input("x8", [1, 8, H, H]);
  const xl1 = g.input("xl1", [70, 7]);
  const xl2 = g.input("xl2", [3, 16]);
  const conv = (name: string, x: ReturnType<typeof g.input>, weight: ReturnType<typeof g.weight>, bias: ReturnType<typeof g.weight> | null, outC: number, pad: number, groups = 1) =>
    g.op(name, "aten.conv2d.default", [x, weight, bias, [1, 1], [pad, pad], [1, 1], groups], [1, outC, H, H]);
  const outs = [
    conv("igemm_k27", x3, w("w_igemm_k27", [70, 3, 3, 3], 0.1), w("b70", [70], 0.2), 70, 1),
    conv("igemm_k36", x4, w("w_igemm_k36", [8, 4, 3, 3], 0.3), null, 8, 1),
    conv("pw_k6", x6, w("w_pw_k6", [70, 6, 1, 1], 0.4), null, 70, 0),
    conv("pw_k8", x8, w("w_pw_k8", [9, 8, 1, 1], 0.5), w("b9", [9], 0.6), 9, 0),
    g.op("lin_k7", "aten.linear.default", [xl1, w("w_lin_k7", [66, 7], 0.7), w("b66", [66], 0.8)], [70, 66]),
    g.op("lin_k16", "aten.linear.default", [xl2, w("w_lin_k16", [5, 16], 0.9)], [3, 5]),
    conv("depthwise", x4, w("w_dw", [4, 1, 3, 3], 1.0), null, 4, 1, 4),
  ];
  const manifest = g.output(...outs);
  const quantize = new Set([...values.keys()].filter((n) => n.startsWith("w_")));
  const inputs: Record<string, Float32Array> = {};
  for (const [i, spec] of manifest.inputs.entries()) inputs[spec.name] = pattern(spec.shape!.reduce((a, b) => a * b, 1), 2 + i, 1.5);
  return { bytes: buildIph(manifest, values, { quantize, kernels }), inputs, outputs: outs.map((o) => o.node_ref) };
}

describe("int8 kernels on the GPU", () => {
  it("int8 weights kept on the GPU give exactly the widened-on-load outputs", async () => {
    const { bytes, inputs, outputs } = model();
    const gpu = await KumaModel.load(bytes, { quantizedWeights: "gpu" });
    const widened = await KumaModel.load(bytes, { quantizedWeights: "widen" });
    try {
      const a = await gpu.run(inputs);
      const b = await widened.run(inputs);
      for (const name of outputs) {
        const x = a[name]!;
        const y = b[name]!;
        expect(x.length, name).toBe(y.length);
        let diffs = 0;
        let maxAbs = 0;
        for (let i = 0; i < x.length; i++) {
          if (x[i] !== y[i]) diffs++;
          maxAbs = Math.max(maxAbs, Math.abs(y[i]!));
        }
        expect(diffs, name).toBe(0);
        expect(maxAbs, `${name} is not all zeros`).toBeGreaterThan(0);
      }
      expect(gpu.weightBytes).toBeLessThan(widened.weightBytes / 2);
    } finally {
      gpu.destroy();
      widened.destroy();
    }
  });
});
