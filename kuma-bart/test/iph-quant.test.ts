import { strToU8, zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { loadIphPackage } from "../src/iph/loader.js";
import { dequantizeWeight, hasQuantizedWeights, uploadWeights, weightBytes, widenQuantizedWeights } from "../src/iph/weights.js";
import { KumaManifestError } from "../src/errors.js";
import { GraphBuilder, ref, runMock } from "./graph-builder.js";
import { createMockDevice } from "./mock-gpu.js";
import { buildIph, pattern, quantizeInt8 } from "./quant-fixture.js";
import type { KumaManifest } from "../src/types/manifest.js";

/** linear(x, w_big) + small float32 bias; w_big (5 x 7 = 35 int8, not a multiple of 4). */
function linearModel(): { manifest: KumaManifest; values: Map<string, Float32Array> } {
  const g = new GraphBuilder();
  const x = g.input("x", [2, 7]);
  const w = g.weight("w", [5, 7]);
  const b = g.weight("b", [5]);
  const manifest = g.output(g.op("y", "aten.linear.default", [x, w, b], [2, 5]));
  return { manifest, values: new Map([["w", pattern(35, 0.3, 0.8)], ["b", pattern(5, 1)]]) };
}

describe("int8 weights in a .iph", () => {
  it("quantizeInt8 is symmetric per output channel, |q| <= 127, max hits 127", () => {
    const v = new Float32Array([0.5, -1, 0.25, 0, 0, 0, 3, -1.5, 0.75]);
    const { q, scales, groupSize } = quantizeInt8(v, [3, 3]);
    expect(groupSize).toBe(3);
    expect(Array.from(scales)).toEqual([Math.fround(1 / 127), 1, Math.fround(3 / 127)]);
    expect(Array.from(q)).toEqual([64, -127, 32, 0, 0, 0, 127, -64, 32]);
  });

  it("loads an int8 package as-is: quantWeights + int8 entries", async () => {
    const { manifest, values } = linearModel();
    const pkg = await loadIphPackage(buildIph(manifest, values, { quantize: new Set(["w"]) }));
    expect(pkg.manifest.format_version).toBe(1);
    expect(hasQuantizedWeights(pkg.manifest)).toBe(true);
    expect(pkg.quantWeights?.byteLength).toBe(36); // 35 bytes, padded to 4
    const w = pkg.manifest.weights.find((e) => e.name === "w")!;
    expect(w.dtype).toBe("int8");
    expect(w.quant).toMatchObject({ scheme: "int8_symmetric", group_size: 7, scale_count: 5 });
  });

  it("widens on load to exactly f32(q) * scale, leaving float32 weights untouched", async () => {
    const { manifest, values } = linearModel();
    const bytes = buildIph(manifest, values, { quantize: new Set(["w"]) });
    const pkg = await loadIphPackage(bytes, { widenQuantized: true });
    expect(pkg.quantWeights).toBeUndefined();
    expect(pkg.manifest.quant_weight_file).toBeUndefined();
    expect(pkg.manifest.weights.every((w) => w.dtype === "float32")).toBe(true);

    const { q, scales } = quantizeInt8(values.get("w")!, [5, 7]);
    const expected = new Float32Array(35).map((_, i) => Math.fround(q[i]! * scales[Math.floor(i / 7)]!));
    const w = pkg.manifest.weights.find((e) => e.name === "w")!;
    expect(Array.from(new Float32Array(pkg.weights.slice(w.byte_offset, w.byte_offset + w.byte_length).buffer))).toEqual(Array.from(expected));
    const b = pkg.manifest.weights.find((e) => e.name === "b")!;
    expect(Array.from(new Float32Array(pkg.weights.slice(b.byte_offset, b.byte_offset + b.byte_length).buffer))).toEqual(Array.from(values.get("b")!));
    // Within half a quantization step of the original.
    const orig = values.get("w")!;
    for (let i = 0; i < 35; i++) expect(Math.abs(expected[i]! - orig[i]!)).toBeLessThanOrEqual(scales[Math.floor(i / 7)]! / 2 + 1e-7);
  });

  it("a float32 package loads unchanged (format_version 0, no quantWeights)", async () => {
    const { manifest, values } = linearModel();
    const bytes = buildIph(manifest, values);
    const plain = await loadIphPackage(bytes);
    const widened = await loadIphPackage(bytes, { widenQuantized: true });
    expect(plain.manifest.format_version).toBe(0);
    expect(plain.quantWeights).toBeUndefined();
    expect(widened.manifest).toEqual(plain.manifest);
    expect(Array.from(widened.weights)).toEqual(Array.from(plain.weights));
    expect(widenQuantizedWeights(plain).weights).toBe(plain.weights);
  });

  it("rejects a format_version newer than the runtime reads", async () => {
    const files = {
      "manifest.json": strToU8(JSON.stringify({ ...linearModel().manifest, format_version: 2 })),
      "weights.f32.bin": new Uint8Array(4),
    };
    await expect(loadIphPackage(zipSync(files).slice().buffer as ArrayBuffer)).rejects.toThrow(/format_version 2/);
  });

  it("rejects an int8 package whose int8 blob is missing", async () => {
    const files = {
      "manifest.json": strToU8(JSON.stringify({ ...linearModel().manifest, format_version: 1, quant_weight_file: "weights.q8.bin" })),
      "weights.f32.bin": new Uint8Array(4),
    };
    await expect(loadIphPackage(zipSync(files).slice().buffer as ArrayBuffer)).rejects.toThrow(/int8 weight file/);
  });

  it("rejects scales inconsistent with the tensor size", async () => {
    const { manifest, values } = linearModel();
    const pkg = await loadIphPackage(buildIph(manifest, values, { quantize: new Set(["w"]) }));
    const w = pkg.manifest.weights.find((e) => e.name === "w")!;
    expect(() => dequantizeWeight(pkg, { ...w, quant: { ...w.quant!, scale_count: 4 } })).toThrow(KumaManifestError);
  });

  it("uploads int8 weights packed (1 byte per weight + scales) and pads an unpadded tail", async () => {
    const { manifest, values } = linearModel();
    for (const padInt8 of [true, false]) {
      const pkg = await loadIphPackage(buildIph(manifest, values, { quantize: new Set(["w"]), padInt8 }));
      const mock = createMockDevice();
      const weights = uploadWeights(mock.device, pkg);
      const w = weights.get("w")!;
      expect(w.buffer.size).toBe(36);
      expect(w.quant).toMatchObject({ name: "w", groupSize: 7 });
      expect(w.quant!.scales.size).toBe(20);
      expect(weights.get("b")!.quant).toBeUndefined();
      expect(weightBytes(weights)).toBe(36 + 20 + 20);
    }
  });
});

describe("int8 weights through the scheduler (mock GPU)", () => {
  async function kernelsFor(manifest: KumaManifest, quantized: string[], missingKernels: string[] = []): Promise<string[]> {
    const mock = createMockDevice();
    const weightBuffers = new Map(
      manifest.weights.map((w) => {
        const buffer = mock.device.createBuffer({ size: Math.ceil(w.n_elements / 4) * 4, usage: GPUBufferUsage.STORAGE });
        const scales = mock.device.createBuffer({ size: w.shape[0]! * 4, usage: GPUBufferUsage.STORAGE });
        const groupSize = w.n_elements / (w.shape.length > 1 ? w.shape[0]! : 1);
        return [w.name, quantized.includes(w.name) ? { buffer, shape: w.shape, quant: { name: w.name, groupSize, scales } } : { buffer, shape: w.shape }];
      }),
    );
    await runMock(manifest, { mock, weightBuffers, missingKernels });
    return mock.kernels;
  }

  it("linear reads an int8 weight with linear_q8.wgsl", async () => {
    expect(await kernelsFor(linearModel().manifest, ["w"])).toEqual(["linear_q8.wgsl"]);
    expect(await kernelsFor(linearModel().manifest, [])).toEqual(["linear.wgsl"]);
  });

  it("a package without the q8 kernel widens on the GPU, then runs the float32 kernel", async () => {
    expect(await kernelsFor(linearModel().manifest, ["w"], ["linear_q8.wgsl"])).toEqual(["dequant_q8.wgsl", "linear.wgsl"]);
  });

  function convModel(weightShape: number[], groups: number, inC: number, outC: number): KumaManifest {
    const g = new GraphBuilder();
    const x = g.input("x", [1, inC, 8, 8]);
    const w = g.weight("w", weightShape);
    const pad = weightShape[2] === 3 ? 1 : 0;
    return g.output(g.op("y", "aten.conv2d.default", [x, w, null, [1, 1], [pad, pad], [1, 1], groups], [1, outC, 8, 8]));
  }

  it("3x3 and 1x1 dense convs read int8 with their q8 kernels", async () => {
    expect(await kernelsFor(convModel([6, 3, 3, 3], 1, 3, 6), ["w"])).toEqual(["conv2d_igemm_q8.wgsl"]);
    expect(await kernelsFor(convModel([6, 3, 1, 1], 1, 3, 6), ["w"])).toEqual(["conv2d_pointwise_q8.wgsl"]);
  });

  it("ops without an int8 kernel (depthwise conv, elementwise) get a widened float32 copy", async () => {
    expect(await kernelsFor(convModel([4, 1, 3, 3], 4, 4, 4), ["w"])).toEqual(["dequant_q8.wgsl", "conv2d_depthwise.wgsl"]);
    const g = new GraphBuilder();
    const w = g.weight("w", [4, 4]);
    const manifest = g.output(g.op("y", "aten.mul.Tensor", [w, w], [4, 4]));
    // Widened once per encode, however often it is resolved.
    expect(await kernelsFor(manifest, ["w"])).toEqual(["dequant_q8.wgsl", "mul.wgsl"]);
  });

  it("an elided linear-weight transpose keeps the weight int8", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 7]);
    const w = g.weight("w", [5, 7]);
    const t = g.op("t", "aten.t.default", [w], [7, 5]);
    const manifest = g.output(g.op("y", "aten.mm.default", [x, ref(t.node_ref)], [2, 5]));
    expect(await kernelsFor(manifest, ["w"])).toEqual(["linear_q8.wgsl"]);
  });
});
