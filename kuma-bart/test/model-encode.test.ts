import { strToU8, zipSync } from "fflate";
import { beforeEach, describe, expect, it } from "vitest";
import { KumaModel } from "../src/model.js";
import { KumaManifestError, KumaShapeError } from "../src/errors.js";
import type { GpuTensor } from "../src/types/tensor.js";
import { GraphBuilder, ref } from "./graph-builder.js";
import { createMockDevice, type MockDevice } from "./mock-gpu.js";

/** x[4] -> relu -> sigmoid -> y, packaged as a .iph zip (no golden.json). */
function reluIph(extra: Record<string, string> = {}, manifestOverride?: object): ArrayBuffer {
  const g = new GraphBuilder();
  const x = g.input("x", [4]);
  const r = g.op("r", "aten.relu.default", [x], [4]);
  const manifest = manifestOverride ?? g.output(g.op("y", "aten.sigmoid.default", [r], [4]));
  const files: Record<string, Uint8Array> = {
    "manifest.json": strToU8(JSON.stringify(manifest)),
    "weights.f32.bin": new Uint8Array([0, 0, 0, 0]),
    "kernels/relu.wgsl": strToU8("// fake"),
    "kernels/sigmoid.wgsl": strToU8("// fake"),
    "kernels/gelu.wgsl": strToU8("// fake"),
  };
  for (const [k, v] of Object.entries(extra)) files[k] = strToU8(v);
  return zipSync(files).slice().buffer as ArrayBuffer;
}

/** A switch model: norm_t routes to relu (t < 0.5) or gelu. */
function switchIph(): ArrayBuffer {
  const g = new GraphBuilder();
  const normT = g.input("norm_t", [1]);
  const route = g.node({ name: "route", op: "js_snippet", target: "route.js", args: [normT], kwargs: {}, meta: { outputs: [{ shape: [1] }, { shape: [1] }] } });
  const segId = g.op("seg_id", "getitem", [route, 0]);
  const localVal = g.op("local_val", "getitem", [route, 1]);
  const branch = (i: number, target: string) => ({
    inputs: [ref(`b${i}_input`)],
    output: ref(`b${i}_out`),
    nodes: [
      { id: 0, name: `b${i}_input`, op: "placeholder" as const, target: `b${i}_input`, args: [], kwargs: {}, meta: { shape: [1] }, kind: "user_input" as const },
      { id: 1, name: `b${i}_out`, op: "call_function" as const, target, args: [ref(`b${i}_input`)], kwargs: {}, meta: { shape: [1] } },
    ],
  });
  const switched = g.node({ name: "switched", op: "switch", target: "switch", args: [localVal], kwargs: {}, meta: { shape: [1] }, selector: segId, branches: [branch(0, "aten.relu.default"), branch(1, "aten.gelu.default")] });
  return reluIph({ "snippets/route.js": "function main(inputs) { return [[inputs[0][0] < 0.5 ? 0 : 1], inputs[0]]; }" }, g.output(switched));
}

function storage(mock: MockDevice, floats: number, extraUsage = 0): GPUBuffer {
  return mock.device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage });
}

function tensor(mock: MockDevice, shape: number[], overrides: Partial<GpuTensor> = {}): GpuTensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return { buffer: storage(mock, n + (overrides.byteOffset ?? 0) / 4), shape, dtype: "f32", ...overrides };
}

describe("KumaModel on a caller-provided device", () => {
  let mock: MockDevice;
  let model: KumaModel;
  beforeEach(async () => {
    mock = createMockDevice();
    // Node has no navigator.gpu: load() can only succeed by using the device it was given.
    model = await KumaModel.load(reluIph(), { device: mock.device });
  });

  it("loads onto the given device and never destroys it", () => {
    expect(model.gpuDevice).toBe(mock.device);
    model.destroy();
    expect(mock.deviceDestroyed).toBe(false);
  });

  it("destroy() on a shared device frees the model's own buffers and the idle scratch", () => {
    const encoder = mock.device.createCommandEncoder();
    const run = model.encode(new Map([["x", tensor(mock, [4])]]), encoder);
    run.release();
    const scratch = run.outputs[0]!.buffer;
    model.destroy();
    expect(mock.destroyedBuffers.has(scratch)).toBe(true);
    expect(model.scratchBytes).toBe(0);
  });

  it("encode() records into the caller's encoder, returns outputs by order and by name, and never submits", () => {
    const encoder = mock.device.createCommandEncoder();
    const input = tensor(mock, [4]);
    const run = model.encode(new Map([["x", input]]), encoder);
    expect(mock.kernels).toEqual(["relu.wgsl", "sigmoid.wgsl"]);
    expect(mock.submits).toBe(0);
    expect(run.outputs).toHaveLength(1);
    expect(run.outputs[0]).toMatchObject({ shape: [4], dtype: "f32", byteOffset: 0 });
    expect(run.outputsByName.get("y")).toBe(run.outputs[0]);
    expect(run.outputs[0]!.buffer).not.toBe(input.buffer);
    expect(model.activeEncodes).toBe(1);
    run.release();
    run.release(); // idempotent
    expect(model.activeEncodes).toBe(0);
    expect(mock.submits).toBe(0);
  });

  it("gives concurrent encodes in one encoder disjoint scratch, then reuses it once released", () => {
    const encoder = mock.device.createCommandEncoder();
    const a = model.encode(new Map([["x", tensor(mock, [4])]]), encoder);
    const b = model.encode(new Map([["x", tensor(mock, [4])]]), encoder);
    expect(a.outputs[0]!.buffer).not.toBe(b.outputs[0]!.buffer);
    expect(model.activeEncodes).toBe(2);
    const bytes = model.scratchBytes;
    a.release();
    b.release();
    const inputs = () => new Map([["x", tensor(mock, [4])]]);
    const created = mock.createdBuffers.size + 2; // the two inputs below
    const c = model.encode(inputs(), encoder);
    const d = model.encode(inputs(), encoder);
    expect(model.scratchBytes).toBe(bytes); // nothing new allocated...
    expect(mock.createdBuffers.size).toBe(created); // ...anywhere
    expect(c.outputs[0]!.buffer).not.toBe(d.outputs[0]!.buffer);
    c.release();
    d.release();
  });

  it("trimScratch() destroys idle sets beyond the kept count and leaves busy ones alone", () => {
    const encoder = mock.device.createCommandEncoder();
    const runs = Array.from({ length: 5 }, () => model.encode(new Map([["x", tensor(mock, [4])]]), encoder));
    const held = runs[4]!;
    runs.slice(0, 4).forEach((r) => r.release());
    expect(model.activeEncodes).toBe(1);
    const before = model.scratchBytes;
    model.trimScratch(1);
    expect(model.scratchBytes).toBe((before * 2) / 5); // one idle kept + the busy one
    expect(mock.destroyedBuffers.has(held.outputs[0]!.buffer)).toBe(false);
    held.release();
    model.trimScratch(0);
    expect(model.scratchBytes).toBe(0);
  });

  it("rejects a missing input, a wrong shape, a wrong dtype, a short buffer, a misaligned offset and missing usage", () => {
    const encoder = mock.device.createCommandEncoder();
    expect(() => model.encode(new Map(), encoder)).toThrow(KumaManifestError);
    expect(() => model.encode(new Map([["x", tensor(mock, [2, 2])]]), encoder)).toThrow(/expected shape \[4\], got \[2, 2\]/);
    expect(() => model.encode(new Map([["x", tensor(mock, [2, 2])]]), encoder)).toThrow(KumaShapeError);
    expect(() => model.encode(new Map([["x", { ...tensor(mock, [4]), dtype: "f16" as unknown as "f32" }]]), encoder)).toThrow(/dtype "f16"/);
    expect(() => model.encode(new Map([["x", { buffer: storage(mock, 2), shape: [4], dtype: "f32" }]]), encoder)).toThrow(/8 bytes.*needs 16/);
    expect(() => model.encode(new Map([["x", tensor(mock, [4], { byteOffset: 6 })]]), encoder)).toThrow(/byteOffset 6/);
    const noStorage = mock.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
    expect(() => model.encode(new Map([["x", { buffer: noStorage, shape: [4], dtype: "f32" }]]), encoder)).toThrow(/STORAGE/);
    // Nothing was recorded or left held by any of those.
    expect(mock.kernels).toEqual([]);
    expect(model.activeEncodes).toBe(0);
  });

  it("stages an input at a non-zero byteOffset through one copy", () => {
    const encoder = mock.device.createCommandEncoder();
    const input = tensor(mock, [4], { byteOffset: 16 });
    const run = model.encode(new Map([["x", input]]), encoder);
    expect(mock.copies).toHaveLength(1);
    const [src, srcOffset, dst, dstOffset, size] = mock.copies[0]!;
    expect(src).toBe(input.buffer);
    expect(srcOffset).toBe(16);
    expect(dstOffset).toBe(0);
    expect(size).toBe(16);
    expect(dst).not.toBe(input.buffer);
    run.release();
  });

  it("copies outputs into caller tensors when asked, honouring their byteOffset", () => {
    const encoder = mock.device.createCommandEncoder();
    const target = tensor(mock, [4], { byteOffset: 32 });
    const run = model.encode(new Map([["x", tensor(mock, [4])]]), encoder, { outputs: new Map([["y", target]]) });
    expect(run.outputs[0]).toBe(target);
    expect(mock.copies).toHaveLength(1);
    const [, srcOffset, dst, dstOffset, size] = mock.copies[0]!;
    expect(dst).toBe(target.buffer);
    expect(srcOffset).toBe(0);
    expect(dstOffset).toBe(32);
    expect(size).toBe(16);
    expect(() => model.encode(new Map([["x", tensor(mock, [4])]]), encoder, { outputs: new Map([["y", tensor(mock, [3])]]) })).toThrow(
      /Output "y" expected shape \[4\], got \[3\]/,
    );
    run.release();
  });

  it("run()/runRaw()/runToGpu() go through encode: one submit per call, scratch and uploads freed on release", async () => {
    const out = await model.run({ x: new Float32Array(4) });
    expect(out.y).toHaveLength(4);
    expect(model.activeEncodes).toBe(0);
    // run(): the graph submit plus the readback's own copy submit.
    expect(mock.submits).toBe(2);

    const raw = await model.runToGpu({ x: new Float32Array(4) });
    expect(mock.submits).toBe(3);
    expect(raw[0]!.data).toHaveLength(0);
    expect(model.activeEncodes).toBe(1);
    const liveUploads = [...mock.createdBuffers].filter((b) => !mock.destroyedBuffers.has(b) && (b.usage & GPUBufferUsage.STORAGE));
    raw[0]!.release();
    expect(model.activeEncodes).toBe(0);
    // Exactly one more storage buffer (the input upload) is gone after release.
    const afterRelease = [...mock.createdBuffers].filter((b) => !mock.destroyedBuffers.has(b) && (b.usage & GPUBufferUsage.STORAGE));
    expect(liveUploads.length - afterRelease.length).toBe(1);

    await expect(model.run({ x: new Float32Array(3) })).rejects.toThrow(KumaManifestError);
    await expect(model.run({})).rejects.toThrow(KumaManifestError);
  });

  it("a switch model needs rawInputs to route on the CPU", async () => {
    const sw = await KumaModel.load(switchIph(), { device: mock.device });
    const encoder = mock.device.createCommandEncoder();
    expect(() => sw.encode(new Map([["norm_t", tensor(mock, [1])]]), encoder)).toThrow(/rawInputs/);
    const run = sw.encode(new Map([["norm_t", tensor(mock, [1])]]), encoder, { rawInputs: new Map([["norm_t", new Float32Array([0.9])]]) });
    expect(mock.kernels).toEqual(["gelu.wgsl"]); // the fake package has no gelu_erf.wgsl
    run.release();
    // run() supplies them itself.
    await sw.run({ norm_t: new Float32Array([0.1]) });
    expect(mock.kernels).toEqual(["gelu.wgsl", "relu.wgsl"]);
  });
});
