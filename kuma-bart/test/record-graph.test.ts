import { describe, expect, it } from "vitest";
import type { KumaManifest } from "../src/types/manifest.js";
import { recordGraph, runGraph, type GraphParams } from "../src/engine/scheduler.js";
import { TransientSet } from "../src/engine/transients.js";
import { createBufferPoolState } from "../src/engine/context.js";
import { KumaUnsupportedOpError } from "../src/errors.js";
import { createMockDevice, type MockDevice } from "./mock-gpu.js";
import { GraphBuilder, fakeKernels, ref, tensorOn } from "./graph-builder.js";

/** x -> relu -> sigmoid -> relu -> sigmoid: four same-size intermediates in a chain. */
function chainManifest(): KumaManifest {
  const g = new GraphBuilder();
  let cur = g.input("x", [4]);
  const targets = ["aten.relu.default", "aten.sigmoid.default", "aten.relu.default", "aten.sigmoid.default"];
  targets.forEach((target, i) => {
    cur = g.op(`n${i}`, target, [cur], [4]);
  });
  return g.output(cur);
}

function paramsFor(mock: MockDevice, manifest: KumaManifest, extra: Partial<GraphParams> = {}): GraphParams {
  return {
    device: mock.device,
    manifest,
    kernels: fakeKernels(),
    pipelineCache: new Map(),
    weightBuffers: new Map(manifest.weights.map((w) => [w.name, tensorOn(mock.device, w.shape)])),
    inputBuffers: new Map(manifest.inputs.map((spec) => [spec.name, tensorOn(mock.device, spec.shape ?? [1])])),
    ...extra,
  };
}

describe("TransientSet", () => {
  it("recycles a freed buffer for the next same-size request and never for a different size", () => {
    const mock = createMockDevice();
    const set = new TransientSet(mock.device);
    const a = set.acquire("k", 16);
    set.free(a);
    expect(set.acquire("k", 16)).toBe(a);
    expect(set.acquire("k", 32)).not.toBe(a);
    expect(set.size).toBe(2);
  });

  it("free() ignores buffers it doesn't own and double frees; reset() idles everything", () => {
    const mock = createMockDevice();
    const set = new TransientSet(mock.device);
    const foreign = tensorOn(mock.device, [4]).buffer;
    set.free(foreign);
    expect(set.owns(foreign)).toBe(false);
    const a = set.acquire("k", 16);
    const b = set.acquire("k", 16);
    set.free(a);
    set.free(a); // second free must not put `a` on the free list twice
    expect(set.acquire("k", 16)).toBe(a);
    expect(set.acquire("k", 16)).not.toBe(b); // b is still busy
    expect(set.size).toBe(3);
    set.reset();
    const after = new Set([set.acquire("k", 16), set.acquire("k", 16), set.acquire("k", 16)]);
    expect(after.size).toBe(3);
    expect(set.size).toBe(3);
    set.destroy();
    expect([...mock.destroyedBuffers].length).toBe(3);
    expect(set.size).toBe(0);
  });
});

describe("recordGraph (mocked GPUDevice)", () => {
  it("records every dispatch into the caller's encoder and never submits", () => {
    const mock = createMockDevice();
    const set = new TransientSet(mock.device);
    const encoder = mock.device.createCommandEncoder();
    const rec = recordGraph(paramsFor(mock, chainManifest()), { encoder, transients: set });
    expect(mock.kernels).toEqual(["relu.wgsl", "sigmoid.wgsl", "relu.wgsl", "sigmoid.wgsl"]);
    expect(mock.submits).toBe(0);
    expect(rec.outputTensors).toHaveLength(1);
    expect(rec.outputTensors[0]!.shape).toEqual([4]);
    expect(set.owns(rec.outputTensors[0]!.buffer)).toBe(true);
    expect(mock.destroyedBuffers.size).toBe(0);
  });

  it("recycles dead intermediates within one encode, never the output", () => {
    const mock = createMockDevice();
    const set = new TransientSet(mock.device);
    const rec = recordGraph(paramsFor(mock, chainManifest()), { encoder: mock.device.createCommandEncoder(), transients: set });
    // Four same-size intermediates; at most two are live at once (a node's input and
    // its output), so the set needs two buffers, not four.
    expect(set.size).toBe(2);
    // The next encode on the same set reuses them rather than growing.
    set.reset();
    const again = recordGraph(paramsFor(mock, chainManifest()), { encoder: mock.device.createCommandEncoder(), transients: set });
    expect(set.size).toBe(2);
    expect(set.owns(again.outputTensors[0]!.buffer)).toBe(true);
    void rec;
  });

  it("two encodes into one encoder with their own sets share no buffer", () => {
    const mock = createMockDevice();
    const encoder = mock.device.createCommandEncoder();
    const a = new TransientSet(mock.device);
    const b = new TransientSet(mock.device);
    const ra = recordGraph(paramsFor(mock, chainManifest()), { encoder, transients: a });
    const rb = recordGraph(paramsFor(mock, chainManifest()), { encoder, transients: b });
    expect(ra.outputTensors[0]!.buffer).not.toBe(rb.outputTensors[0]!.buffer);
    expect(a.owns(rb.outputTensors[0]!.buffer)).toBe(false);
    expect(b.owns(ra.outputTensors[0]!.buffer)).toBe(false);
    expect(mock.submits).toBe(0);
  });

  it("an output that aliases a caller input is handed back as that input, untouched", () => {
    const mock = createMockDevice();
    const g = new GraphBuilder();
    const manifest = g.output(g.op("aliased", "aten.alias.default", [g.input("x", [4])], [4]));
    const params = paramsFor(mock, manifest);
    const set = new TransientSet(mock.device);
    const rec = recordGraph(params, { encoder: mock.device.createCommandEncoder(), transients: set });
    expect(rec.outputTensors[0]!.buffer).toBe(params.inputBuffers.get("x")!.buffer);
    expect(set.size).toBe(0);
  });

  it("clears the buffer of a node that allocated but dispatched nothing (zeros_like relies on zero fill)", () => {
    const mock = createMockDevice();
    const g = new GraphBuilder();
    const z = g.op("z", "aten.zeros_like.default", [g.input("x", [4])], [4]);
    const manifest = g.output(g.op("y", "aten.relu.default", [z], [4]));
    const set = new TransientSet(mock.device);
    const rec = recordGraph(paramsFor(mock, manifest), { encoder: mock.device.createCommandEncoder(), transients: set });
    expect(mock.clears).toHaveLength(1);
    expect(mock.kernels).toEqual(["relu.wgsl"]);
    // Still true on the ring-pool path, which also recycles buffers across calls.
    void rec;
    const ringMock = createMockDevice();
    return runGraph({ ...paramsFor(ringMock, manifest), bufferPool: createBufferPoolState() }).then(() => {
      expect(ringMock.clears).toHaveLength(1);
    });
  });

  it("seeds switch-branch inputs from the transient set, not fresh uploads", () => {
    const mock = createMockDevice();
    const g = new GraphBuilder();
    const normT = g.input("norm_t", [1]);
    const route = g.node({
      name: "route",
      op: "js_snippet",
      target: "route.js",
      args: [normT],
      kwargs: {},
      meta: { outputs: [{ shape: [1] }, { shape: [1] }] },
    });
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
    const switched = g.node({
      name: "switched",
      op: "switch",
      target: "switch",
      args: [localVal],
      kwargs: {},
      meta: { shape: [1] },
      selector: segId,
      branches: [branch(0, "aten.relu.default"), branch(1, "aten.gelu.default")],
    });
    const manifest = g.output(switched);
    const set = new TransientSet(mock.device);
    const snippets = new Map([["route.js", "function main(inputs) { return [[inputs[0][0] < 0.5 ? 0 : 1], inputs[0]]; }"]]);
    const rec = recordGraph(
      paramsFor(mock, manifest, { rawInputs: new Map([["norm_t", new Float32Array([0.9])]]), snippets, snippetCache: new Map() }),
      { encoder: mock.device.createCommandEncoder(), transients: set },
    );
    expect(mock.kernels).toEqual(["gelu_erf.wgsl"]);
    // The seeded branch input and the branch's one output both live in the set.
    expect(set.size).toBe(2);
    expect(set.owns(rec.outputTensors[0]!.buffer)).toBe(true);
    expect(mock.destroyedBuffers.size).toBe(0);
  });

  it("fails before recording anything for an unsupported op, leaving the set empty", () => {
    const mock = createMockDevice();
    const manifest = chainManifest();
    manifest.graph.nodes[2]!.target = "aten.native_layer_norm.default";
    const set = new TransientSet(mock.device);
    expect(() => recordGraph(paramsFor(mock, manifest), { encoder: mock.device.createCommandEncoder(), transients: set })).toThrow(
      KumaUnsupportedOpError,
    );
    expect(mock.kernels).toEqual([]);
    expect(set.size).toBe(0);
  });
});
