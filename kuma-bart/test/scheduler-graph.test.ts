import { afterEach, describe, expect, it, vi } from "vitest";
import type { KumaManifest } from "../src/types/manifest.js";
import { runGraph } from "../src/engine/scheduler.js";
import { profileGraph } from "../src/engine/profile.js";
import { findLinearWeightElisions } from "../src/ops/linear.js";
import { KumaUnsupportedOpError } from "../src/errors.js";
import { BUFFER_POOL_DEPTH, createBufferPoolState } from "../src/engine/context.js";
import { createMockDevice } from "./mock-gpu.js";
import { GraphBuilder, fakeKernels, ref, runMock, tensorOn } from "./graph-builder.js";

/** Live (created, never destroyed) non-uniform buffers -- activation memory. Uniform
 * params buffers persist in constantCache by design. */
function liveStorage(mock: ReturnType<typeof createMockDevice>): object[] {
  return [...mock.createdBuffers].filter((b) => !(b.usage & GPUBufferUsage.UNIFORM) && !mock.destroyedBuffers.has(b));
}

describe("findLinearWeightElisions", () => {
  it("elides a transpose whose sole consumer is addmm and whose input is a weight parameter", () => {
    const g = new GraphBuilder();
    const w = g.weight("w", [3, 2]);
    const b = g.weight("b", [3]);
    const x = g.input("x", [1, 2]);
    const t = g.op("t", "aten.t.default", [w]);
    g.op("addmm", "aten.addmm.default", [b, x, t]);
    expect(findLinearWeightElisions(g.nodes).get("t")).toBe("w");
  });

  it("does not elide a transpose with more than one consumer", () => {
    const g = new GraphBuilder();
    const w = g.weight("w", [3, 2]);
    const x = g.input("x", [1, 2]);
    const t = g.op("t", "aten.t.default", [w]);
    g.op("mm1", "aten.mm.default", [x, t]);
    g.op("mm2", "aten.mm.default", [x, t]);
    expect(findLinearWeightElisions(g.nodes).size).toBe(0);
  });

  it("does not elide a transpose whose input is not a literal weight parameter", () => {
    const g = new GraphBuilder();
    const x = g.input("x", [2, 2]);
    const x2 = g.input("x2", [2, 2]);
    const t = g.op("t", "aten.t.default", [x]);
    g.op("mm", "aten.mm.default", [x2, t]);
    expect(findLinearWeightElisions(g.nodes).size).toBe(0);
  });
});

function reluManifest(): KumaManifest {
  const g = new GraphBuilder();
  const x = g.input("x", [4]);
  return g.output(g.op("relu_out", "aten.relu.default", [x], [4]));
}

/** z = complex(re, im); combined = op(z, r); out_imag = imag(combined). */
function complexBinaryManifest(target: string): KumaManifest {
  const g = new GraphBuilder();
  const z = g.op("z", "aten.complex.default", [g.input("re", [4]), g.input("im", [4])], [4]);
  const combined = g.op("combined", target, [z, g.input("r", [4])], [4]);
  return g.output(g.op("out_imag", "aten.imag.default", [combined], [4]));
}

function divModeManifest(mode: string, bShape: number[] = [2, 3], b?: number): KumaManifest {
  const g = new GraphBuilder();
  const a = g.input("a", [2, 3]);
  const bArg = b ?? g.input("b", bShape);
  return g.output(g.op("divided", "aten.div.Tensor_mode", [a, bArg, mode], [2, 3]));
}

const ROUTE_SNIPPET = "function main(inputs) { const normT = inputs[0]; const segId = normT[0] < 0.5 ? 0 : 1; return [[segId], normT]; }";

function switchManifest(): KumaManifest {
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
  return g.output(switched);
}

describe("runGraph (mocked GPUDevice — structural checks only, no real compute)", () => {
  it("resolves a placeholder input through a single op to the output", async () => {
    const { outputs, mock } = await runMock(reluManifest());
    expect(outputs).toHaveLength(1);
    expect(outputs[0]!.name).toBe("relu_out");
    expect(outputs[0]!.shape).toEqual([4]);
    expect(outputs[0]!.data).toHaveLength(4);
    expect(mock.kernels).toEqual(["relu.wgsl"]);
  });

  it("throws KumaUnsupportedOpError for a target with no registered kernel", async () => {
    const manifest = reluManifest();
    manifest.graph.nodes[1]!.target = "aten.native_layer_norm.default";
    await expect(runMock(manifest)).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("resolves a multi-output node (chunk) through getitem, dispatching one slice per piece", async () => {
    const g = new GraphBuilder();
    const chunk = g.op("chunk", "aten.chunk.default", [g.input("x", [4, 3]), 2, 0], undefined, { outputs: [[2, 3], [2, 3]] });
    const manifest = g.output(g.op("piece0", "getitem", [chunk, 0], [2, 3]), g.op("piece1", "getitem", [chunk, 1], [2, 3]));
    const { outputs, mock } = await runMock(manifest);
    expect(outputs.map((o) => o.name)).toEqual(["piece0", "piece1"]);
    expect(outputs[0]!.shape).toEqual([2, 3]);
    expect(outputs[1]!.shape).toEqual([2, 3]);
    // getitem is a pure lookup — only chunk's two slices dispatch.
    expect(mock.kernels).toEqual(["slice.wgsl", "slice.wgsl"]);
  });

  it("a free passthrough op (alias) dispatches zero kernels", async () => {
    const g = new GraphBuilder();
    const { outputs, mock } = await runMock(g.output(g.op("aliased", "aten.alias.default", [g.input("x", [4])], [4])));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual([]);
  });

  it("low-memory early-free does not destroy a buffer still aliased by a live node (sizeThreshold 0)", async () => {
    // x → a=relu(x) → { c=alias(a)→e=relu(c), f=relu(a) }. `c` shares a's buffer; `e`
    // (c's last consumer) is encoded before `f` (a's last consumer). A per-name refcount
    // would free a's buffer at `e`, then `f` would bind a destroyed buffer -- the mock
    // throws on that, so passing means the buffer survived until its true last use.
    const g = new GraphBuilder();
    const a = g.op("a", "aten.relu.default", [g.input("x", [4])], [4]);
    const c = g.op("c", "aten.alias.default", [a], [4]);
    const e = g.op("e", "aten.relu.default", [c], [4]);
    const f = g.op("f", "aten.relu.default", [a], [4]);
    const { outputs } = await runMock(g.output(e, f), {
      bufferPool: createBufferPoolState({ depth: 1, sizeThreshold: 0 }),
      earlyFreeSubmitBudget: 0, // flush per-buffer so mid-graph early-free is exercised
    });
    expect(outputs.map((o) => o.name)).toEqual(["e", "f"]);
  });

  it("complex/real pairing round-trips through aten.complex/aten.real with zero dispatch", async () => {
    const g = new GraphBuilder();
    const z = g.op("z", "aten.complex.default", [g.input("re", [4]), g.input("im", [4])], [4]);
    const { outputs, mock } = await runMock(g.output(g.op("back_to_real", "aten.real.default", [z], [4])));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual([]);
  });

  it("aten.mul.Tensor(complex, real) propagates the imaginary part (real mul + imag mul)", async () => {
    const { outputs, mock } = await runMock(complexBinaryManifest("aten.mul.Tensor"));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual(["mul.wgsl", "mul.wgsl"]);
  });

  it("aten.add.Tensor(complex, real) propagates the imaginary part with zero extra dispatch", async () => {
    const { outputs, mock } = await runMock(complexBinaryManifest("aten.add.Tensor"));
    expect(outputs[0]!.shape).toEqual([4]);
    // out_im = a_im + 0 reuses a's own imag buffer directly.
    expect(mock.kernels).toEqual(["add.wgsl"]);
  });

  it("aten.div.Tensor(complex, real) propagates the imaginary part (real div + imag div)", async () => {
    const { outputs, mock } = await runMock(complexBinaryManifest("aten.div.Tensor"));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual(["div.wgsl", "div.wgsl"]);
  });

  it("aten.mul.Tensor(complex, complex) dispatches the full cross-term product", async () => {
    const g = new GraphBuilder();
    const z = g.op("z", "aten.complex.default", [g.input("re", [4]), g.input("im", [4])], [4]);
    const sq = g.op("sq", "aten.mul.Tensor", [z, z], [4]);
    const { mock } = await runMock(g.output(g.op("out_imag", "aten.imag.default", [sq], [4])));
    expect(mock.kernels).toEqual(["mul.wgsl", "mul.wgsl", "sub.wgsl", "mul.wgsl", "mul.wgsl", "add.wgsl"]);
  });

  it("aten.div.Tensor with a complex denominator fails loudly instead of silently dropping it", async () => {
    const g = new GraphBuilder();
    const z = g.op("z", "aten.complex.default", [g.input("re", [4]), g.input("im", [4])], [4]);
    await expect(runMock(g.output(g.op("out", "aten.div.Tensor", [z, z], [4])))).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("aten.rsub.Scalar negates the imaginary part of a complex input", async () => {
    const g = new GraphBuilder();
    const z = g.op("z", "aten.complex.default", [g.input("re", [4]), g.input("im", [4])], [4]);
    const rs = g.op("rs", "aten.rsub.Scalar", [z, 1.0], [4]);
    const { outputs, mock } = await runMock(g.output(g.op("out_imag", "aten.imag.default", [rs], [4])));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual(["sub.wgsl", "neg.wgsl"]);
  });

  it("stack of 3 tensors dispatches 2 concat.wgsl calls (N-1 chain, via free unsqueeze)", async () => {
    const g = new GraphBuilder();
    const parts = ["a", "b", "c"].map((n) => g.input(n, [2, 3]));
    const { outputs, mock } = await runMock(g.output(g.op("stacked", "aten.stack.default", [parts, 0], [3, 2, 3])));
    expect(outputs[0]!.shape).toEqual([3, 2, 3]);
    expect(mock.kernels).toEqual(["concat.wgsl", "concat.wgsl"]);
  });

  it('div_mode="floor" dispatches div.wgsl then floor.wgsl', async () => {
    const { outputs, mock } = await runMock(divModeManifest("floor"));
    expect(outputs[0]!.shape).toEqual([2, 3]);
    expect(mock.kernels).toEqual(["div.wgsl", "floor.wgsl"]);
  });

  it('div_mode="floor" broadcasts a smaller tensor operand and accepts a scalar one', async () => {
    const broadcast = await runMock(divModeManifest("floor", [1, 3]));
    expect(broadcast.mock.kernels).toEqual(["permute.wgsl", "div.wgsl", "floor.wgsl"]);
    const scalar = await runMock(divModeManifest("floor", [], 2));
    expect(scalar.mock.kernels).toEqual(["div.wgsl", "floor.wgsl"]);
  });

  it("low-memory (sizeThreshold 0) frees op-internal intermediates, not just node outputs", async () => {
    // floor-div allocates an intra-op `divided` buffer (never in `resolved`) plus the
    // node output; at sizeThreshold 0 both are fresh allocations. After the run the only
    // live storage buffers must be the caller's inputs and the output it was handed --
    // and releasing the output leaves just the inputs.
    const { outputs, mock, inputBuffers } = await runMock(divModeManifest("floor"), {
      bufferPool: createBufferPoolState({ depth: 1, sizeThreshold: 0 }),
      earlyFreeSubmitBudget: 0,
    });
    const inputs = [...inputBuffers.values()].map((t) => t.buffer);
    expect(new Set(liveStorage(mock))).toEqual(new Set([...inputs, outputs[0]!.buffer]));
    outputs[0]!.release();
    expect(new Set(liveStorage(mock))).toEqual(new Set(inputs));
  });

  it('div_mode="trunc" (unsupported) fails loudly instead of guessing', async () => {
    await expect(runMock(divModeManifest("trunc"))).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("js_snippet -> getitem -> switch picks exactly one branch (segment 0), dispatching only its kernel", async () => {
    const { outputs, mock } = await runMock(switchManifest(), {
      rawInputs: new Map([["norm_t", new Float32Array([0.2])]]),
      snippets: new Map([["route.js", ROUTE_SNIPPET]]),
    });
    expect(outputs[0]!.shape).toEqual([1]);
    expect(mock.kernels).toEqual(["relu.wgsl"]);
  });

  it("...and the other branch (segment 1) when the snippet routes differently", async () => {
    const { mock } = await runMock(switchManifest(), {
      rawInputs: new Map([["norm_t", new Float32Array([0.8])]]),
      snippets: new Map([["route.js", ROUTE_SNIPPET]]),
    });
    expect(mock.kernels).toEqual(["gelu_erf.wgsl"]);
  });

  it("switch-seeded branch inputs are freed by runGraph (it allocated them)", async () => {
    const { mock, inputBuffers, outputs } = await runMock(switchManifest(), {
      rawInputs: new Map([["norm_t", new Float32Array([0.2])]]),
      snippets: new Map([["route.js", ROUTE_SNIPPET]]),
    });
    releaseAll(outputs);
    const callerInputs = [...inputBuffers.values()].map((t) => t.buffer);
    // The pooled relu output stays (pool ring); the uploaded seed is gone.
    const live = liveStorage(mock).filter((b) => !callerInputs.includes(b as GPUBuffer));
    expect(live).toEqual([outputs[0]!.buffer]);
  });

  it("aten.mul.Tensor tolerates a literal scalar second operand (not just tensor refs)", async () => {
    const g = new GraphBuilder();
    const { outputs, mock } = await runMock(g.output(g.op("scaled", "aten.mul.Tensor", [g.input("x", [4]), 32], [4])));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual(["mul.wgsl"]);
  });

  it("aten.rsub.Scalar (scalar - tensor) dispatches sub.wgsl with a broadcast constant", async () => {
    const g = new GraphBuilder();
    const { outputs, mock } = await runMock(g.output(g.op("result", "aten.rsub.Scalar", [g.input("x", [4]), 1.0], [4])));
    expect(outputs[0]!.shape).toEqual([4]);
    expect(mock.kernels).toEqual(["sub.wgsl"]);
  });

  it("literal-operand constants are keyed by element count, not node name alone", async () => {
    const constantCache = new Map<string, GPUBuffer>();
    for (const n of [4, 8]) {
      const g = new GraphBuilder();
      const x = g.input("x", [n]);
      await runMock(g.output(g.op("result", "aten.rsub.Scalar", [x, 1.0], [n]), g.op("scaled", "aten.mul.Tensor", [x, 2], [n])), { constantCache });
    }
    const constantKeys = [...constantCache.keys()].filter((k) => !k.startsWith("params:")).sort();
    expect(constantKeys).toEqual(["operand:scaled:b:4", "operand:scaled:b:8", "rsub:result:4", "rsub:result:8"]);
  });

  it("aten.rsub.Scalar with alpha != 1 fails loudly instead of computing the wrong thing", async () => {
    const g = new GraphBuilder();
    const manifest = g.output(g.op("result", "aten.rsub.Scalar", [g.input("x", [4]), 1.0, 2.0], [4]));
    await expect(runMock(manifest)).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("aten.conv2d.default with the bare 3-arg form (stride/padding/dilation/groups omitted at their defaults)", async () => {
    const g = new GraphBuilder();
    const args = [g.input("x", [1, 3, 4, 4]), g.weight("w", [2, 3, 3, 3]), g.weight("b", [2])];
    const { outputs, mock } = await runMock(g.output(g.op("y", "aten.conv2d.default", args, [1, 2, 4, 4])));
    expect(outputs[0]!.shape).toEqual([1, 2, 4, 4]);
    expect(mock.kernels).toEqual(["conv2d_igemm.wgsl"]);
  });

  it("a dense conv2d falls back to conv2d.wgsl for packages exported before conv2d_igemm.wgsl", async () => {
    const g = new GraphBuilder();
    const args = [g.input("x", [1, 3, 4, 4]), g.weight("w", [2, 3, 3, 3]), g.weight("b", [2])];
    const { mock } = await runMock(g.output(g.op("y", "aten.conv2d.default", args, [1, 2, 4, 4])), { missingKernels: ["conv2d_igemm.wgsl"] });
    expect(mock.kernels).toEqual(["conv2d.wgsl"]);
  });

  it("aten.group_norm.default with the 4-arg form (eps omitted at its default)", async () => {
    const g = new GraphBuilder();
    const args = [g.input("x", [1, 4, 2, 2]), 2, g.weight("w", [4]), g.weight("b", [4])];
    const { outputs, mock } = await runMock(g.output(g.op("y", "aten.group_norm.default", args, [1, 4, 2, 2])));
    expect(outputs[0]!.shape).toEqual([1, 4, 2, 2]);
    expect(mock.kernels).toEqual(["groupnorm.wgsl"]);
  });

  it("aten.linear.default applied channel-last to a 4D input (e.g. nn.Linear over [B,H,W,C])", async () => {
    const g = new GraphBuilder();
    const args = [g.input("x", [1, 3, 2, 4]), g.weight("w", [5, 4]), g.weight("b", [5])];
    const { outputs, mock } = await runMock(g.output(g.op("y", "aten.linear.default", args, [1, 3, 2, 5])));
    // leading dims [1,3,2] preserved, only the last dim (4 -> 5) changes.
    expect(outputs[0]!.shape).toEqual([1, 3, 2, 5]);
    expect(mock.kernels).toEqual(["linear.wgsl"]);
  });
});

describe("unary elementwise ops", () => {
  afterEach(() => vi.restoreAllMocks());

  it.each([
    ["aten.sigmoid.default", "sigmoid.wgsl"],
    ["aten.tanh.default", "tanh.wgsl"],
    ["aten.silu.default", "silu.wgsl"],
    ["aten.abs.default", "abs.wgsl"],
    ["aten.exp.default", "exp.wgsl"],
    ["aten.log.default", "log.wgsl"],
    ["aten.rsqrt.default", "rsqrt.wgsl"],
    ["aten.hardswish.default", "hardswish.wgsl"],
  ])("%s dispatches %s", async (target, kernel) => {
    const g = new GraphBuilder();
    const { outputs, mock } = await runMock(g.output(g.op("y", target, [g.input("x", [2, 3])], [2, 3])));
    expect(outputs[0]!.shape).toEqual([2, 3]);
    expect(mock.kernels).toEqual([kernel]);
  });

  function geluManifest(kwargs: Record<string, string> = {}): KumaManifest {
    const g = new GraphBuilder();
    return g.output(g.op("y", "aten.gelu.default", [g.input("x", [4])], [4], { kwargs }));
  }

  it("aten.gelu.default runs exact erf GELU (gelu_erf.wgsl) by default", async () => {
    expect((await runMock(geluManifest())).mock.kernels).toEqual(["gelu_erf.wgsl"]);
    expect((await runMock(geluManifest({ approximate: "none" }))).mock.kernels).toEqual(["gelu_erf.wgsl"]);
  });

  it('aten.gelu.default with approximate="tanh" runs gelu.wgsl', async () => {
    expect((await runMock(geluManifest({ approximate: "tanh" }))).mock.kernels).toEqual(["gelu.wgsl"]);
  });

  it("aten.gelu.default falls back to gelu.wgsl, warning once, for packages without gelu_erf.wgsl", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const first = await runMock(geluManifest(), { missingKernels: ["gelu_erf.wgsl"] });
    const second = await runMock(geluManifest(), { missingKernels: ["gelu_erf.wgsl"] });
    expect(first.mock.kernels).toEqual(["gelu.wgsl"]);
    expect(second.mock.kernels).toEqual(["gelu.wgsl"]);
    expect(warn).toHaveBeenCalledTimes(1);
  });
});

function releaseAll(outputs: { release(): void }[]): void {
  for (const o of outputs) o.release();
}

describe("buffer ownership", () => {
  it("runGraph never destroys caller-provided inputs, so one input buffer can serve many calls", async () => {
    const mock = createMockDevice();
    const inputBuffers = new Map([["x", tensorOn(mock.device, [4])]]);
    const bufferPool = createBufferPoolState({ sizeThreshold: 0 });
    for (let i = 0; i < 3; i++) {
      const { outputs } = await runMock(reluManifest(), { mock, inputBuffers, bufferPool });
      releaseAll(outputs);
    }
    expect(mock.destroyedBuffers.has(inputBuffers.get("x")!.buffer)).toBe(false);
  });

  it("release() destroys a fresh (non-pooled) output, and is a no-op for pooled ones", async () => {
    const fresh = await runMock(reluManifest(), { bufferPool: createBufferPoolState({ sizeThreshold: 0 }) });
    fresh.outputs[0]!.release();
    expect(fresh.mock.destroyedBuffers.has(fresh.outputs[0]!.buffer)).toBe(true);

    const pooled = await runMock(reluManifest(), { bufferPool: createBufferPoolState() });
    pooled.outputs[0]!.release();
    expect(pooled.mock.destroyedBuffers.has(pooled.outputs[0]!.buffer)).toBe(false);
  });

  it("release() on an output that aliases a caller input leaves the input alone", async () => {
    const g = new GraphBuilder();
    const manifest = g.output(g.op("aliased", "aten.alias.default", [g.input("x", [4])], [4]));
    const { outputs, mock, inputBuffers } = await runMock(manifest, { bufferPool: createBufferPoolState({ sizeThreshold: 0 }) });
    outputs[0]!.release();
    expect(mock.destroyedBuffers.has(inputBuffers.get("x")!.buffer)).toBe(false);
  });

  it("early-free never destroys a synthesized weight, even on the cold call that creates it", async () => {
    // The weight is rebuilt by synthetic_weights.ts (not in weights[]) and cached in
    // constantCache *during* this call; its only consumer drops its refcount to zero.
    const g = new GraphBuilder();
    const w = g.weight("layer.complex_tucker.lifted_tensor_0", [4, 4], false);
    const manifest = g.output(g.op("y", "aten.relu.default", [w], [4, 4]));
    const constantCache = new Map<string, GPUBuffer>();
    const { mock } = await runMock(manifest, {
      constantCache,
      bufferPool: createBufferPoolState({ depth: 1, sizeThreshold: 0 }),
      earlyFreeSubmitBudget: 0,
    });
    const synthesized = constantCache.get("layer.complex_tucker.lifted_tensor_0");
    expect(synthesized).toBeDefined();
    expect(mock.destroyedBuffers.has(synthesized!)).toBe(false);
  });
});

describe("profileGraph", () => {
  it("times every dispatching node through the same encode path as runGraph, freeing its own buffers", async () => {
    const mock = createMockDevice({ features: ["timestamp-query"] });
    const g = new GraphBuilder();
    const a = g.op("a", "aten.relu.default", [g.input("x", [4])], [4]);
    const c = g.op("c", "aten.alias.default", [a], [4]);
    const manifest = g.output(g.op("e", "aten.sigmoid.default", [c], [4]));
    const inputBuffers = new Map([["x", tensorOn(mock.device, [4])]]);
    const report = await profileGraph({
      device: mock.device,
      manifest,
      kernels: fakeKernels(),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers,
      bufferPool: createBufferPoolState({ sizeThreshold: 0 }),
    });
    expect(report.perNode.map((t) => t.name)).toEqual(["a", "c", "e"]);
    expect(mock.kernels).toEqual(["relu.wgsl", "sigmoid.wgsl"]);
    const live = liveStorage(mock).filter((b) => (b as { usage: number }).usage & GPUBufferUsage.STORAGE);
    expect(live).toEqual([inputBuffers.get("x")!.buffer]);
  });
});

describe("runGraph buffer pooling (BufferPoolState — see engine/context.ts)", () => {
  it("without a shared bufferPool (the default), repeated calls never alias each other's buffers", async () => {
    const mock = createMockDevice();
    const first = await runMock(reluManifest(), { mock });
    const second = await runMock(reluManifest(), { mock });
    expect(first.outputs[0]!.buffer).not.toBe(second.outputs[0]!.buffer);
  });

  it("with a shared bufferPool, a node's output buffer is reused exactly every BUFFER_POOL_DEPTH calls", async () => {
    const mock = createMockDevice();
    const bufferPool = createBufferPoolState();
    // Sequential calls, BUFFER_POOL_DEPTH + 2 of them, so the ring wraps at least once.
    const buffers: GPUBuffer[] = [];
    for (let i = 0; i < BUFFER_POOL_DEPTH + 2; i++) {
      const { outputs } = await runMock(reluManifest(), { mock, bufferPool });
      buffers.push(outputs[0]!.buffer);
    }
    // Generation N and generation N + BUFFER_POOL_DEPTH land on the same ring slot.
    expect(buffers[BUFFER_POOL_DEPTH]).toBe(buffers[0]);
    expect(buffers[BUFFER_POOL_DEPTH + 1]).toBe(buffers[1]);
    // Every other pair within one rotation is a genuinely distinct buffer.
    expect(new Set(buffers.slice(0, BUFFER_POOL_DEPTH)).size).toBe(BUFFER_POOL_DEPTH);
  });

  it("pooled buffers are never destroyed, by runGraph's cleanup or by release()", async () => {
    const mock = createMockDevice();
    const bufferPool = createBufferPoolState();
    const seenBuffers = new Set<GPUBuffer>();
    for (let i = 0; i < BUFFER_POOL_DEPTH + 2; i++) {
      const { outputs } = await runMock(reluManifest(), { mock, bufferPool });
      seenBuffers.add(outputs[0]!.buffer);
      releaseAll(outputs);
    }
    for (const buffer of seenBuffers) {
      expect(mock.destroyedBuffers.has(buffer)).toBe(false);
    }
  });

  it("concurrent (not individually-awaited) calls sharing a pool never lose or double-assign a generation", async () => {
    // Mirrors KumaPlayer firing several runToGpu() calls without awaiting each, relying on
    // acquireGenerationSlot/releaseGenerationSlot to keep them safe -- the scenario where
    // an off-by-one or lost update in the generation counter would show up. Doesn't
    // assume array index == assigned generation (microtask order isn't pinned down).
    const mock = createMockDevice();
    const bufferPool = createBufferPoolState();
    const callCount = BUFFER_POOL_DEPTH + 2;
    const manifest = reluManifest();
    const results = await Promise.all(
      Array.from({ length: callCount }, () =>
        runGraph({
          device: mock.device,
          manifest,
          kernels: fakeKernels(),
          pipelineCache: new Map(),
          weightBuffers: new Map(),
          inputBuffers: new Map([["x", tensorOn(mock.device, [4])]]),
          bufferPool,
        }),
      ),
    );
    // Every call advanced callGeneration exactly once.
    expect(bufferPool.callGeneration).toBe(callCount);
    // Exactly BUFFER_POOL_DEPTH distinct buffers across all calls: fewer means two
    // generations collided on one slot; more means the ring didn't wrap.
    const distinctBuffers = new Set(results.map((outputs) => outputs[0]!.buffer));
    expect(distinctBuffers.size).toBe(Math.min(callCount, BUFFER_POOL_DEPTH));
  });
});
