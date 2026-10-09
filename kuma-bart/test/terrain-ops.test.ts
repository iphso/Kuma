import { describe, expect, it } from "vitest";
import { KumaUnsupportedOpError } from "../src/errors.js";
import { GraphBuilder, runMock } from "./graph-builder.js";
import { createMockDevice } from "./mock-gpu.js";

// Ops added for the terrain-diffusion EDM2 U-Nets (Loom): mean.dim, ones_like, the clip
// alias of clamp, nearest 2D upsampling and softmax. Mock-GPU: dispatch sequence and
// output shapes only; numerics are covered by golden verification on a real GPU
// (test/gpu/terrain-models.test.ts).

describe("aten.mean.dim", () => {
  it("reduces a contiguous axis block with one mean.wgsl dispatch and keeps the manifest shape", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 3, 4]);
    const { outputs, mock } = await runMock(g.output(g.op("m", "aten.mean.dim", [x, [1], true], [2, 1, 4])));
    expect(outputs[0]!.shape).toEqual([2, 1, 4]);
    expect(mock.kernels).toEqual(["mean.wgsl"]);
    expect(mock.dispatches).toEqual([1]); // 8 output rows -> 1 workgroup of 64
  });

  it("accepts a bare int dim and a negative dim", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 3, 4]);
    const { outputs, mock } = await runMock(g.output(g.op("m", "aten.mean.dim", [x, -1, false], [2, 3])));
    expect(outputs[0]!.shape).toEqual([2, 3]);
    expect(mock.kernels).toEqual(["mean.wgsl"]);
  });

  it("refuses a non-contiguous axis block instead of guessing", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 3, 4]);
    await expect(runMock(g.output(g.op("m", "aten.mean.dim", [x, [0, 2], true], [1, 3, 1])))).rejects.toThrow(KumaUnsupportedOpError);
  });
});

describe("aten.ones_like.default", () => {
  it("is dispatch-free and reuses one cached constant buffer across runs", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [1, 1, 4, 4]);
    const manifest = g.output(g.op("ones", "aten.ones_like.default", [x], [1, 1, 4, 4]));
    const mock = createMockDevice();
    const constantCache = new Map<string, GPUBuffer>();
    const first = await runMock(manifest, { mock, constantCache });
    const second = await runMock(manifest, { mock, constantCache });
    expect(first.outputs[0]!.shape).toEqual([1, 1, 4, 4]);
    expect(mock.kernels).toEqual([]);
    expect(second.outputs[0]!.buffer).toBe(first.outputs[0]!.buffer);
    expect(constantCache.size).toBe(1);
  });
});

describe("aten.clip.default", () => {
  it("is the clamp kernel under its alias", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [8]);
    const { outputs, mock } = await runMock(g.output(g.op("c", "aten.clip.default", [x, -256, 256], [8])));
    expect(outputs[0]!.shape).toEqual([8]);
    expect(mock.kernels).toEqual(["clamp.wgsl"]);
  });
});

describe("aten.upsample_nearest2d", () => {
  it(".vec with scale_factors doubles H and W through upsample_nearest2d.wgsl", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [1, 3, 4, 5]);
    const { outputs, mock } = await runMock(g.output(g.op("up", "aten.upsample_nearest2d.vec", [x, null, [2.0, 2.0]], [1, 3, 8, 10])));
    expect(outputs[0]!.shape).toEqual([1, 3, 8, 10]);
    expect(mock.kernels).toEqual(["upsample_nearest2d.wgsl"]);
    expect(mock.dispatches).toEqual([Math.ceil(240 / 64)]);
  });

  it(".default with output_size derives the shape when the manifest has none", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 1, 3, 3]);
    const { outputs, mock } = await runMock(g.output(g.op("up", "aten.upsample_nearest2d.default", [x, [6, 6]])));
    expect(outputs[0]!.shape).toEqual([2, 1, 6, 6]);
    expect(mock.kernels).toEqual(["upsample_nearest2d.wgsl"]);
  });

  it(".vec with scale_factors derives the shape when the manifest has none", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [1, 2, 4, 4]);
    const { outputs } = await runMock(g.output(g.op("up", "aten.upsample_nearest2d.vec", [x, null, [2.0, 2.0]])));
    expect(outputs[0]!.shape).toEqual([1, 2, 8, 8]);
  });
});

describe("softmax", () => {
  it("aten.softmax.int dispatches one softmax.wgsl row per (outer, inner) pair", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [1, 2, 64, 64]);
    const { outputs, mock } = await runMock(g.output(g.op("s", "aten.softmax.int", [x, 3, null], [1, 2, 64, 64])));
    expect(outputs[0]!.shape).toEqual([1, 2, 64, 64]);
    expect(mock.kernels).toEqual(["softmax.wgsl"]);
    expect(mock.dispatches).toEqual([Math.ceil(128 / 64)]); // 128 rows of 64 logits
  });

  it("aten._softmax.default(half_to_float=false) and a middle dim both run", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 5, 3]);
    const { outputs, mock } = await runMock(g.output(g.op("s", "aten._softmax.default", [x, 1, false], [2, 5, 3])));
    expect(outputs[0]!.shape).toEqual([2, 5, 3]);
    expect(mock.kernels).toEqual(["softmax.wgsl"]);
    expect(mock.dispatches).toEqual([1]); // 6 rows
  });

  it("refuses a non-float32 dtype request", async () => {
    const g = new GraphBuilder();
    const x = g.input("x", [4, 4]);
    await expect(runMock(g.output(g.op("s", "aten.softmax.int", [x, 1, "float16"], [4, 4])))).rejects.toThrow(KumaUnsupportedOpError);
  });
});
