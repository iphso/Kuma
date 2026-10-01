import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { strToU8, zipSync } from "fflate";
import { describe, expect, it } from "vitest";
import { loadIphPackage, peekIphFormat } from "../src/iph/loader.js";
import { KumaManifestError } from "../src/errors.js";

const BASE_MANIFEST = {
  format: "kuma",
  format_version: 0,
  weight_file: "weights.f32.bin",
  endianness: "little",
  inputs: [{ name: "x", shape: [2], dtype: "float32" }],
  outputs: [{ name: "y", shape: [2] }],
  weights: [],
  graph: { node_count: 0, op_counts: {}, nodes: [] },
  warnings: [],
  unsupported_ops: [],
};

function buildFakeIph(opts: { manifest?: object; includeManifest?: boolean; includeWeights?: boolean } = {}) {
  const files: Record<string, Uint8Array> = {};
  if (opts.includeManifest !== false) {
    files["manifest.json"] = strToU8(JSON.stringify(opts.manifest ?? BASE_MANIFEST));
  }
  if (opts.includeWeights !== false) {
    files["weights.f32.bin"] = new Uint8Array([1, 2, 3, 4]);
  }
  files["kernels/relu.wgsl"] = strToU8("// fake kernel");
  files["debug_report.md"] = strToU8("# debug");
  const zipped = zipSync(files);
  return zipped.slice().buffer as ArrayBuffer;
}

describe("loadIphPackage", () => {
  it("parses manifest, weights, and kernels from a real-shaped zip", async () => {
    const pkg = await loadIphPackage(buildFakeIph());
    expect(pkg.manifest.format).toBe("kuma");
    expect(pkg.weights.byteLength).toBe(4);
    expect(pkg.kernels.get("relu.wgsl")).toBe("// fake kernel");
  });

  it("throws KumaManifestError when manifest.json is missing", async () => {
    await expect(loadIphPackage(buildFakeIph({ includeManifest: false }))).rejects.toThrow(KumaManifestError);
  });

  it("throws KumaManifestError for an unrecognized format", async () => {
    await expect(
      loadIphPackage(buildFakeIph({ manifest: { ...BASE_MANIFEST, format: "not-kuma" } })),
    ).rejects.toThrow(KumaManifestError);
  });

  it("throws KumaManifestError when the declared weight file is missing", async () => {
    await expect(loadIphPackage(buildFakeIph({ includeWeights: false }))).rejects.toThrow(KumaManifestError);
  });

  it("matches the real artifacts/simple/manifest.json shape, if present", () => {
    const manifestPath = fileURLToPath(new URL("../../artifacts/simple/manifest.json", import.meta.url));
    if (!existsSync(manifestPath)) return; // acceptance artifact not generated in this environment
    const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
    expect(manifest.format).toBe("kuma");
    expect(Array.isArray(manifest.graph.nodes)).toBe(true);
    expect(manifest.inputs[0]).toHaveProperty("shape");
    expect(manifest.weights[0]).toHaveProperty("byte_offset");
  });
});

describe("peekIphFormat", () => {
  it("reads the manifest's format without needing the rest of the package", () => {
    expect(peekIphFormat(buildFakeIph())).toBe("kuma");
    expect(peekIphFormat(buildFakeIph({ manifest: { ...BASE_MANIFEST, format: "onnx" } }))).toBe("onnx");
    expect(peekIphFormat(buildFakeIph({ includeManifest: false }))).toBeUndefined();
    expect(peekIphFormat(new Uint8Array([1, 2, 3]))).toBeUndefined();
  });
});
