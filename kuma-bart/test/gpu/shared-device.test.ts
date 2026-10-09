import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { createStorageBuffer, readBuffers, requestComputeDevice, uploadFloat32 } from "webgpu-compute-kit";
import { KumaModel, KumaShapeError, type GpuTensor } from "../../src/index.js";

// The smallest committed packages: simple.iph (40 KB, conv -> gelu -> conv -> add on a
// 1x3x32x32 input, no golden.json) for the byte-for-byte encode/run comparisons, and
// halfshake.iph (7.4 MB, a 2-branch switch-routed model with golden.json, which verify()
// passes on Dawn -- halfbunny/halfbeauty are a little smaller but each carry one
// pre-existing golden mismatch on this backend) for verify().
const ARTIFACTS = new URL("../../../artifacts/", import.meta.url);
function loadArtifact(name: string): ArrayBuffer {
  const bytes = readFileSync(fileURLToPath(new URL(name, ARTIFACTS)));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

const bits = (a: Float32Array): Uint32Array => new Uint32Array(a.buffer, a.byteOffset, a.length);
const numel = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);
function pattern(n: number, phase = 0): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin(0.37 * i + phase) * 2;
  return out;
}

let device: GPUDevice;
const gpuErrors: string[] = [];
beforeAll(async () => {
  ({ device } = await requestComputeDevice({ onUncapturedError: (e) => gpuErrors.push(e.message) }));
});
afterAll(() => device.destroy());
afterEach(() => expect(gpuErrors).toEqual([]));

/** Runs `fn` under a validation error scope and fails on any error it caught. */
async function validated<T>(fn: () => Promise<T> | T): Promise<T> {
  device.pushErrorScope("validation");
  const result = await fn();
  const error = await device.popErrorScope();
  expect(error?.message).toBeUndefined();
  return result;
}

describe("simple.iph on a shared device", () => {
  let model: KumaModel;
  let inputSpec: { name: string; shape: number[] };
  let outputName: string;
  let n: number;
  beforeAll(async () => {
    model = await KumaModel.load(loadArtifact("simple.iph"), { device });
    const spec = model.inputs[0]!;
    inputSpec = { name: spec.name, shape: spec.shape! };
    outputName = model.outputs[0]!.name;
    n = numel(inputSpec.shape);
  });
  afterAll(() => model.destroy());

  it("loads with the kit's requestComputeDevice() device and reports it", () => {
    expect(model.gpuDevice).toBe(device);
    expect(inputSpec.shape).toEqual([1, 3, 32, 32]);
  });

  it("(a) encode() on kit-uploaded inputs matches the CPU-input run() path bit for bit", async () => {
    const data = pattern(n);
    const viaRun = (await model.run({ [inputSpec.name]: data }))[outputName]!;

    const inputBuffer = uploadFloat32(device, data);
    const viaEncode = await validated(async () => {
      const encoder = device.createCommandEncoder();
      const run = model.encode(new Map([[inputSpec.name, { buffer: inputBuffer, shape: inputSpec.shape, dtype: "f32" }]]), encoder);
      device.queue.submit([encoder.finish()]);
      const out = run.outputs[0]!;
      expect(out.shape).toEqual(model.outputs[0]!.shape);
      expect(run.outputsByName.get(outputName)).toBe(out);
      const [read] = await readBuffers(device, [{ buffer: out.buffer, byteLength: numel(out.shape) * 4 }]);
      run.release();
      return read!;
    });
    inputBuffer.destroy();

    expect(viaEncode.length).toBe(viaRun.length);
    expect(viaEncode.some((v) => v !== 0)).toBe(true);
    expect(bits(viaEncode)).toEqual(bits(viaRun));

    // And against a model that owns its own device: same bytes across devices.
    const own = await KumaModel.load(loadArtifact("simple.iph"));
    try {
      const viaOwn = (await own.run({ [inputSpec.name]: data }))[outputName]!;
      expect(bits(viaOwn)).toEqual(bits(viaRun));
    } finally {
      own.destroy();
    }
  });

  it("(b) two encode() calls in one encoder give the same bytes as two separate run() calls", async () => {
    const a = pattern(n, 0.1);
    const b = pattern(n, 2.3);
    const runA = (await model.run({ [inputSpec.name]: a }))[outputName]!;
    const runB = (await model.run({ [inputSpec.name]: b }))[outputName]!;
    expect(bits(runA)).not.toEqual(bits(runB));

    const bufA = uploadFloat32(device, a);
    const bufB = uploadFloat32(device, b);
    const [encA, encB] = await validated(async () => {
      const encoder = device.createCommandEncoder();
      const first = model.encode(new Map([[inputSpec.name, { buffer: bufA, shape: inputSpec.shape, dtype: "f32" }]]), encoder);
      const second = model.encode(new Map([[inputSpec.name, { buffer: bufB, shape: inputSpec.shape, dtype: "f32" }]]), encoder);
      expect(first.outputs[0]!.buffer).not.toBe(second.outputs[0]!.buffer);
      expect(model.activeEncodes).toBe(2);
      device.queue.submit([encoder.finish()]);
      const reads = await readBuffers(device, [first.outputs[0]!, second.outputs[0]!].map((o) => ({ buffer: o.buffer, byteLength: numel(o.shape) * 4 })));
      first.release();
      second.release();
      expect(model.activeEncodes).toBe(0);
      return reads;
    });
    bufA.destroy();
    bufB.destroy();
    expect(bits(encA!)).toEqual(bits(runA));
    expect(bits(encB!)).toEqual(bits(runB));
  });

  it("(b') windows packed in one buffer: offset inputs in, offset outputs out, one submission", async () => {
    // Loom's shape of use: N windows' inputs side by side in one buffer, outputs written
    // side by side into another, all encoded back to back and submitted once.
    const windows = 4;
    const bytes = n * 4;
    const inputs = pattern(n * windows, 0.5);
    const packedIn = uploadFloat32(device, inputs);
    const packedOut = createStorageBuffer(device, bytes * windows);
    const expected: Float32Array[] = [];
    for (let w = 0; w < windows; w++) {
      expected.push((await model.run({ [inputSpec.name]: inputs.subarray(w * n, (w + 1) * n) }))[outputName]!);
    }

    const got = await validated(async () => {
      const encoder = device.createCommandEncoder();
      const runs = [];
      for (let w = 0; w < windows; w++) {
        const input: GpuTensor = { buffer: packedIn, shape: inputSpec.shape, dtype: "f32", byteOffset: w * bytes };
        const output: GpuTensor = { buffer: packedOut, shape: inputSpec.shape, dtype: "f32", byteOffset: w * bytes };
        const run = model.encode(new Map([[inputSpec.name, input]]), encoder, { outputs: new Map([[outputName, output]]) });
        expect(run.outputs[0]).toBe(output);
        runs.push(run);
      }
      device.queue.submit([encoder.finish()]);
      for (const run of runs) run.release();
      const [read] = await readBuffers(device, [{ buffer: packedOut, byteLength: bytes * windows }]);
      return read!;
    });
    packedIn.destroy();
    packedOut.destroy();
    for (let w = 0; w < windows; w++) {
      expect(bits(got.subarray(w * n, (w + 1) * n))).toEqual(bits(expected[w]!));
    }
  });

  it("rejects a wrong-shape input before recording anything", () => {
    const buffer = createStorageBuffer(device, 16 * 4);
    const encoder = device.createCommandEncoder();
    expect(() => model.encode(new Map([[inputSpec.name, { buffer, shape: [1, 1, 4, 4], dtype: "f32" }]]), encoder)).toThrow(KumaShapeError);
    encoder.finish(); // still a valid, empty command buffer
    buffer.destroy();
    expect(model.activeEncodes).toBe(0);
  });

  it("encode time per call (CPU, recording only)", async () => {
    const inputBuffer = uploadFloat32(device, pattern(n));
    const inputs = new Map([[inputSpec.name, { buffer: inputBuffer, shape: inputSpec.shape, dtype: "f32" as const }]]);
    const encodeMany = (count: number): number => {
      const encoder = device.createCommandEncoder();
      const t0 = performance.now();
      const runs = Array.from({ length: count }, () => model.encode(inputs, encoder));
      const elapsed = performance.now() - t0;
      device.queue.submit([encoder.finish()]);
      for (const run of runs) run.release();
      return elapsed / count;
    };
    encodeMany(20); // warm pipelines, scratch and bind groups
    await device.queue.onSubmittedWorkDone();
    const perCall = encodeMany(200);
    await device.queue.onSubmittedWorkDone();
    // 200 concurrent encodes left 200 scratch sets cached; trim back to the usual few.
    const burstBytes = model.scratchBytes;
    model.trimScratch();
    expect(model.scratchBytes).toBeLessThan(burstBytes / 10);
    const t0 = performance.now();
    const outputs = await model.runToGpu({ [inputSpec.name]: pattern(n) });
    await device.queue.onSubmittedWorkDone();
    const runToGpuMs = performance.now() - t0;
    for (const o of outputs) o.release();
    inputBuffer.destroy();
    console.log(
      `[gpu] simple.iph: encode ${perCall.toFixed(3)} ms/call (${model.outputs.length} output, 4 dispatches), ` +
        `runToGpu+onSubmittedWorkDone ${runToGpuMs.toFixed(2)} ms, scratch ${model.scratchBytes} bytes after trim (${burstBytes} during the 200-call burst)`,
    );
    expect(perCall).toBeLessThan(5);
  });
});

describe("halfshake.iph (switch-routed, with golden.json) on a shared device", () => {
  let model: KumaModel;
  beforeAll(async () => {
    model = await KumaModel.load(loadArtifact("halfshake.iph"), { device });
  });
  afterAll(() => model.destroy());

  it("(c) verify() passes on the shared device", async () => {
    const report = await model.verify();
    expect(report.branches.map((b) => b.mismatches)).toEqual(report.branches.map(() => []));
    expect(report.branches.map((b) => b.nodesMissing)).toEqual(report.branches.map(() => []));
    expect(report.ok).toBe(true);
  });

  it("encode() of a switch model routes on rawInputs and matches run()", async () => {
    const t = new Float32Array([0.75]);
    const viaRun = (await model.run({ norm_t: t })).output!;
    const inputBuffer = uploadFloat32(device, t);
    const inputs = new Map([["norm_t", { buffer: inputBuffer, shape: [1], dtype: "f32" as const }]]);
    const rawInputs = new Map([["norm_t", t]]);
    const viaEncode = await validated(async () => {
      // One warm-up so the timed encode reuses scratch and pipelines (as a steady state would).
      const warm = device.createCommandEncoder();
      model.encode(inputs, warm, { rawInputs }).release();
      device.queue.submit([warm.finish()]);
      await device.queue.onSubmittedWorkDone();

      const encoder = device.createCommandEncoder();
      const t0 = performance.now();
      const run = model.encode(inputs, encoder, { rawInputs });
      const encodeMs = performance.now() - t0;
      device.queue.submit([encoder.finish()]);
      const out = run.outputs[0]!;
      const [read] = await readBuffers(device, [{ buffer: out.buffer, byteLength: numel(out.shape) * 4 }]);
      run.release();
      console.log(`[gpu] halfshake.iph: encode ${encodeMs.toFixed(2)} ms (one branch, warm), output ${out.shape.join("x")}, scratch ${(model.scratchBytes / 1e6).toFixed(1)} MB`);
      return read!;
    });
    inputBuffer.destroy();
    expect(bits(viaEncode)).toEqual(bits(viaRun));
  });
});
