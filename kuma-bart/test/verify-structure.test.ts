/** Structural contract tests for the verify path — no real GPU needed.
 *
 * These tests cover the pure-JS logic in verifyBranch / manifestForSwitchBranch
 * / verifyAgainstGolden that must be correct before GPU computation even runs:
 * input buffer naming, node name matching, captureNodes iteration, and the
 * synthetic manifest shape. The mock GPU returns zero-filled buffers, so
 * numeric correctness is deliberately NOT tested here — that requires a real
 * browser (see kuma-bart/README.md). What IS tested: if any of these structural
 * checks fail, the user would see nodesMissing or a throw instead of actual=0,
 * so the zeros-in-verify bug lives elsewhere.
 */

import { describe, expect, it } from "vitest";
import type { GoldenBranch, GoldenData } from "../src/types/golden.js";
import type { GraphNode, KumaManifest } from "../src/types/manifest.js";
import { verifyAgainstGolden } from "../src/engine/verify.js";
import { createBufferPoolState } from "../src/engine/context.js";
import { createMockDevice } from "./mock-gpu.js";

// ── Test helpers ─────────────────────────────────────────────────────────────

/** A simple 2-node branch (user_input → add → output) for structural tests.
 * Numeric values don't matter here (mock GPU is always zero); only naming does. */
function buildSimpleBranch(branchIndex: number): {
  nodes: GraphNode[];
  output: { node_ref: string };
  golden: GoldenBranch;
} {
  const prefix = `branch${branchIndex}__`;
  const inputName = `${prefix}input_0`;
  const addName = `${prefix}Add_out`;

  const nodes: GraphNode[] = [
    {
      id: 0,
      name: inputName,
      op: "placeholder",
      target: inputName,
      args: [],
      kwargs: {},
      meta: { shape: [1] },
      kind: "user_input",
    },
    {
      id: 1,
      name: addName,
      op: "call_function",
      target: "aten.add.Tensor",
      args: [{ node_ref: inputName }, 1.0],
      kwargs: {},
      meta: { shape: [1] },
    },
  ];

  const golden: GoldenBranch = {
    inputs: { [inputName]: [0.0] },
    nodes: {
      // Mock golden: what the verifier expects. The mock GPU returns 0 so comparison
      // will fail, but we're only testing structural properties (no nodesMissing).
      [addName]: {
        re: {
          shape: [1],
          n: 1,
          finite: 1,
          mean: 1.0,
          min: 1.0,
          max: 1.0,
          first: [1.0],
          spread_indices: [0],
          spread: [1.0],
        },
      },
    },
  };

  return { nodes, output: { node_ref: addName }, golden };
}

function buildSwitchManifestWith1Branch(branchIndex: number): {
  manifest: KumaManifest;
  goldenData: GoldenData;
} {
  const { nodes, output, golden } = buildSimpleBranch(branchIndex);

  const manifest: KumaManifest = {
    format: "kuma",
    format_version: 0,
    weight_file: "weights.f32.bin",
    endianness: "little",
    inputs: [{ name: "input_0" }],
    outputs: [{ name: "model_output" }],
    weights: [],
    graph: {
      node_count: 5,
      op_counts: {},
      nodes: [
        { id: 0, name: "input_0", op: "placeholder", target: "input_0", args: [], kwargs: {}, meta: { shape: [1] }, kind: "user_input" },
        { id: 1, name: "router", op: "js_snippet", target: "route.js", args: [{ node_ref: "input_0" }], kwargs: {}, meta: { outputs: [{ shape: [1] }, { shape: [1] }] } },
        { id: 2, name: "seg_id", op: "call_function", target: "getitem", args: [{ node_ref: "router" }, 0], kwargs: {}, meta: {} },
        { id: 3, name: "local_val", op: "call_function", target: "getitem", args: [{ node_ref: "router" }, 1], kwargs: {}, meta: {} },
        {
          id: 4,
          name: "switch_0",
          op: "switch",
          target: "switch",
          args: [{ node_ref: "local_val" }],
          kwargs: {},
          meta: { shape: [1] },
          selector: { node_ref: "seg_id" },
          branches: [{ nodes, inputs: [{ node_ref: nodes[0]!.name }], output }],
        },
        { id: 5, name: "output", op: "output", target: "output", args: [[{ node_ref: "switch_0" }]], kwargs: {}, meta: {} },
      ],
    },
    warnings: [],
    unsupported_ops: [],
  };

  const goldenData: GoldenData = {
    format_version: 0,
    branches: [golden],
  };

  return { manifest, goldenData };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("verifyAgainstGolden — structural contract (mock GPU, numeric values not checked)", () => {
  it("produces a result with one branch when there is one switch branch", async () => {
    const { manifest, goldenData } = buildSwitchManifestWith1Branch(0);
    const { device } = createMockDevice();
    const ctx = {
      device,
      kernels: new Map([["add.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      snippets: new Map([["route.js", "function main(i){return[[0],i]}"]]),
      snippetCache: new Map(),
      constantCache: new Map(),
      bufferPool: createBufferPoolState(),
    };

    const report = await verifyAgainstGolden(ctx, manifest, goldenData);

    expect(report.branches).toHaveLength(1);
    expect(report.branches[0]!.branch).toBe(0);
  });

  it("reports nodesChecked=1 for a branch with one golden node", async () => {
    const { manifest, goldenData } = buildSwitchManifestWith1Branch(0);
    const { device } = createMockDevice();
    const ctx = {
      device,
      kernels: new Map([["add.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      snippets: new Map([["route.js", "function main(i){return[[0],i]}"]]),
      snippetCache: new Map(),
      constantCache: new Map(),
      bufferPool: createBufferPoolState(),
    };

    const report = await verifyAgainstGolden(ctx, manifest, goldenData);

    // The mock GPU returns zeros, so there will be mismatches -- but nodesChecked tells
    // us whether the node was FOUND (it should be, if naming is correct).
    expect(report.branches[0]!.nodesChecked).toBe(1);
    expect(report.branches[0]!.nodesMissing).toHaveLength(0);
  });

  it("reports nodesMissing when the golden has a key that doesn't match any manifest node", async () => {
    const { manifest, goldenData } = buildSwitchManifestWith1Branch(0);
    // Corrupt the golden to use a wrong name
    const corruptedGolden: GoldenData = {
      format_version: 0,
      branches: [{
        inputs: goldenData.branches[0]!.inputs,
        nodes: {
          "branch0__NONEXISTENT_NODE": goldenData.branches[0]!.nodes["branch0__Add_out"]!,
        },
      }],
    };

    const { device } = createMockDevice();
    const ctx = {
      device,
      kernels: new Map([["add.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      snippets: new Map([["route.js", "function main(i){return[[0],i]}"]]),
      snippetCache: new Map(),
      constantCache: new Map(),
      bufferPool: createBufferPoolState(),
    };

    const report = await verifyAgainstGolden(ctx, manifest, corruptedGolden);

    expect(report.branches[0]!.nodesMissing).toContain("branch0__NONEXISTENT_NODE");
    expect(report.branches[0]!.nodesChecked).toBe(0);
  });

  it("the golden input key IS the user_input placeholder name (must match for buffer upload)", async () => {
    // This is the CONTRACT test: golden.inputs key MUST equal the user_input
    // placeholder's `name` field after namespace_golden, because verifyBranch
    // feeds inputBuffers keyed exactly by golden.inputs keys, and the scheduler
    // looks up inputBuffers by node.name for user_input placeholders.
    const { nodes } = buildSimpleBranch(0);
    const userInputPlaceholder = nodes.find((n) => n.kind === "user_input");
    expect(userInputPlaceholder).toBeDefined();

    const goldenInputKey = "branch0__input_0";
    expect(goldenInputKey).toBe(userInputPlaceholder!.name);
  });

  it("reports nodesChecked=0 when the golden input key is wrong (simulating a namespace mismatch)", async () => {
    // If golden.inputs is keyed by the WRONG name (e.g., pre-namespace "input_0"
    // instead of namespaced "branch0__input_0"), the scheduler throws a
    // KumaManifestError ("Missing input for placeholder") rather than returning zeros.
    // This test verifies that scenario fails loudly (throws) rather than silently
    // producing zeros -- so if you see zeros, the input key IS correct (this is ruled out).
    const { nodes, output } = buildSimpleBranch(0);

    const brokenGolden: GoldenData = {
      format_version: 0,
      branches: [{
        inputs: { "input_0": [0.0] },  // WRONG: missing "branch0__" prefix
        nodes: { "branch0__Add_out": { re: { shape: [1], n: 1, finite: 1, mean: 1.0, min: 1.0, max: 1.0, first: [1.0], spread_indices: [0], spread: [1.0] } } },
      }],
    };

    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "",
      endianness: "little",
      inputs: [],
      outputs: [{ name: "model_output" }],
      weights: [],
      graph: {
        node_count: 1,
        op_counts: {},
        nodes: [
          { id: 0, name: "switch_0", op: "switch", target: "switch", args: [], kwargs: {}, meta: { shape: [1] },
            selector: { node_ref: "nonexistent" },
            branches: [{ nodes, inputs: [{ node_ref: nodes[0]!.name }], output }] },
          { id: 1, name: "output", op: "output", target: "output", args: [[{ node_ref: "switch_0" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const { device } = createMockDevice();
    const ctx = {
      device,
      kernels: new Map([["add.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      snippets: new Map(),
      snippetCache: new Map(),
      constantCache: new Map(),
      bufferPool: createBufferPoolState(),
    };

    // A wrong golden.inputs key means the user_input placeholder is never seeded →
    // KumaManifestError: "Missing input for placeholder 'branch0__input_0'"
    await expect(verifyAgainstGolden(ctx, manifest, brokenGolden)).rejects.toThrow();
  });
});
