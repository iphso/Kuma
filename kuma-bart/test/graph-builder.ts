import type { ArgValue, GraphNode, IOSpec, KumaManifest, NodeRef, WeightEntry } from "../src/types/manifest.js";
import type { ResolvedTensor } from "../src/engine/context.js";
import { runGraph, type RunGraphOutput, type RunGraphParams } from "../src/engine/scheduler.js";
import { numElements } from "../src/engine/shape.js";
import { createMockDevice, type MockDevice } from "./mock-gpu.js";

export const ref = (name: string): NodeRef => ({ node_ref: name });

/** Builds a KumaManifest node by node. Every method returns a ref to the node it added. */
export class GraphBuilder {
  readonly nodes: GraphNode[] = [];
  private readonly inputs: IOSpec[] = [];
  private readonly weights: WeightEntry[] = [];

  private add(node: Omit<GraphNode, "id" | "kwargs"> & { kwargs?: GraphNode["kwargs"] }): NodeRef {
    this.nodes.push({ kwargs: {}, ...node, id: this.nodes.length });
    return ref(node.name);
  }

  /** A user_input placeholder (also declared in manifest.inputs). */
  input(name: string, shape: number[]): NodeRef {
    this.inputs.push({ name, shape });
    return this.add({ name, op: "placeholder", target: name, args: [], meta: { shape }, kind: "user_input" });
  }

  /** A parameter placeholder backed by a manifest weight of the same name (pass
   * `declare: false` for a weight the runtime synthesizes instead). */
  weight(name: string, shape: number[], declare = true): NodeRef {
    if (declare) {
      const n = numElements(shape);
      const offset = this.weights.reduce((sum, w) => sum + w.byte_length, 0);
      this.weights.push({ name, shape, dtype: "float32", n_elements: n, byte_offset: offset, byte_length: n * 4 });
    }
    return this.add({ name, op: "placeholder", target: name, args: [], meta: { shape }, kind: "parameter", weight_name: name });
  }

  /** A call_function node. `shape` is its output shape (omit for e.g. elision tests);
   * pass `outputs` instead for a multi-output node. */
  op(
    name: string,
    target: string,
    args: ArgValue[],
    shape?: number[],
    extra: { kwargs?: GraphNode["kwargs"]; outputs?: number[][] } = {},
  ): NodeRef {
    const meta = extra.outputs ? { outputs: extra.outputs.map((s) => ({ shape: s })) } : shape ? { shape } : {};
    return this.add({ name, op: "call_function", target, args, meta, kwargs: extra.kwargs ?? {} });
  }

  /** Any other node (js_snippet, switch), added verbatim. */
  node(node: Omit<GraphNode, "id">): NodeRef {
    this.nodes.push({ ...node, id: this.nodes.length });
    return ref(node.name);
  }

  /** Appends the output node and returns the manifest. Declared output names/shapes come
   * from the referenced nodes. */
  output(...outs: NodeRef[]): KumaManifest {
    this.nodes.push({ id: this.nodes.length, name: "output", op: "output", target: "output", args: [outs], kwargs: {}, meta: {} });
    const shapeOf = (name: string) => this.nodes.find((n) => n.name === name)?.meta.shape;
    return {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [...this.inputs],
      outputs: outs.map((o) => ({ name: o.node_ref, shape: shapeOf(o.node_ref) })),
      weights: [...this.weights],
      graph: { node_count: this.nodes.length, op_counts: {}, nodes: [...this.nodes] },
      warnings: [],
      unsupported_ops: [],
    };
  }
}

/** Every kernel name resolves to fake WGSL, except `missing` ones. */
export function fakeKernels(missing: readonly string[] = []): ReadonlyMap<string, string> {
  const all = new (class extends Map<string, string> {
    override get(key: string): string | undefined {
      return missing.includes(key) ? undefined : "// fake";
    }
    override has(key: string): boolean {
      return !missing.includes(key);
    }
  })();
  return all;
}

/** A storage buffer sized for `shape`, created on the mock device. */
export function tensorOn(device: GPUDevice, shape: number[]): ResolvedTensor {
  return { buffer: device.createBuffer({ size: numElements(shape) * 4, usage: GPUBufferUsage.STORAGE }), shape };
}

export interface MockRun {
  outputs: RunGraphOutput[];
  mock: MockDevice;
  inputBuffers: Map<string, ResolvedTensor>;
  weightBuffers: Map<string, ResolvedTensor>;
}

/** Runs `manifest` through runGraph on a mock device, creating a buffer for every
 * declared input and weight. Any RunGraphParams field can be overridden. */
export async function runMock(
  manifest: KumaManifest,
  overrides: Partial<RunGraphParams> & { mock?: MockDevice; missingKernels?: string[] } = {},
): Promise<MockRun> {
  const { mock: givenMock, missingKernels, ...params } = overrides;
  const mock = givenMock ?? createMockDevice();
  const inputBuffers =
    (params.inputBuffers as Map<string, ResolvedTensor> | undefined) ??
    new Map(manifest.inputs.map((spec) => [spec.name, tensorOn(mock.device, spec.shape ?? [1])]));
  const weightBuffers =
    (params.weightBuffers as Map<string, ResolvedTensor> | undefined) ??
    new Map(manifest.weights.map((w) => [w.name, tensorOn(mock.device, w.shape)]));
  const outputs = await runGraph({
    device: mock.device,
    manifest,
    kernels: fakeKernels(missingKernels),
    pipelineCache: new Map(),
    ...params,
    weightBuffers,
    inputBuffers,
  });
  return { outputs, mock, inputBuffers, weightBuffers };
}
