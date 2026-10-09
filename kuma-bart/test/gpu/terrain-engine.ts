import { requestComputeDevice, uploadBufferSlice, uploadFloat32 } from "webgpu-compute-kit";
import { loadIphPackage } from "../../src/iph/loader.js";
import { runGraph, releaseOutputs } from "../../src/engine/scheduler.js";
import { profileGraph, type ProfileReport } from "../../src/engine/profile.js";
import { verifyAgainstGolden, type VerifyReport } from "../../src/engine/verify.js";
import { createBufferPoolState, type BufferPoolState, type ResolvedTensor } from "../../src/engine/context.js";
import type { SnippetFn } from "../../src/engine/snippets.js";
import type { KumaManifest } from "../../src/types/manifest.js";
import type { GoldenData } from "../../src/types/golden.js";

/**
 * KumaModel.load/run/verify, mirrored on a device whose storage-binding and buffer
 * limits are raised to the adapter's maximum. The 512x512 decoder binds 192-channel
 * activations (192 MiB), above WebGPU's default 128 MiB maxStorageBufferBindingSize;
 * KumaModel.load requests a default-limits device, which drops every submission with a
 * validation error. Until the shared-device rework lets a caller supply the device,
 * the GPU tests drive the engine directly through this.
 */
export class TerrainModel {
  readonly pipelineCache = new Map<string, GPUComputePipeline>();
  readonly constantCache = new Map<string, GPUBuffer>();
  readonly snippetCache = new Map<string, SnippetFn>();
  readonly bufferPool: BufferPoolState = createBufferPoolState({ depth: 1 });

  private constructor(
    readonly device: GPUDevice,
    readonly manifest: KumaManifest,
    readonly kernels: ReadonlyMap<string, string>,
    readonly snippets: ReadonlyMap<string, string>,
    readonly weightBuffers: Map<string, ResolvedTensor>,
    readonly golden: GoldenData | undefined,
    readonly limits: { maxStorageBufferBindingSize: number; maxBufferSize: number },
    /** Time spent in loadIphPackage (unzip + inflate of the weight blob). */
    readonly parseMs: number,
  ) {}

  static async load(bytes: ArrayBuffer): Promise<TerrainModel> {
    const tParse0 = performance.now();
    const { manifest, weights, kernels, snippets, golden } = await loadIphPackage(bytes);
    const parseMs = performance.now() - tParse0;
    const { device } = await requestComputeDevice({
      // A minimum of 1 asks the kit for the adapter's full available value.
      requiredLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
      onUncapturedError: (e) => {
        throw new Error(`[webgpu] uncaptured error: ${e.message}`);
      },
    });
    const weightBuffers = new Map<string, ResolvedTensor>();
    for (const w of manifest.weights) {
      weightBuffers.set(w.name, { buffer: uploadBufferSlice(device, weights, w.byte_offset, w.byte_length), shape: w.shape });
    }
    return new TerrainModel(device, manifest, kernels, snippets, weightBuffers, golden, {
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
    }, parseMs);
  }

  private ctx() {
    return {
      device: this.device,
      kernels: this.kernels,
      pipelineCache: this.pipelineCache,
      weightBuffers: this.weightBuffers,
      snippets: this.snippets,
      snippetCache: this.snippetCache,
      constantCache: this.constantCache,
      bufferPool: this.bufferPool,
    };
  }

  verify(): Promise<VerifyReport> {
    if (!this.golden) throw new Error("package has no golden.json");
    return verifyAgainstGolden(this.ctx(), this.manifest, this.golden);
  }

  /** run(): uploads inputs at their manifest shapes, reads every declared output back. */
  async run(inputs: Record<string, Float32Array>): Promise<Record<string, Float32Array>> {
    const inputBuffers = new Map<string, ResolvedTensor>();
    const rawInputs = new Map<string, Float32Array>();
    for (const spec of this.manifest.inputs) {
      const data = inputs[spec.name];
      if (!data) throw new Error(`missing input "${spec.name}"`);
      inputBuffers.set(spec.name, { buffer: uploadFloat32(this.device, data), shape: spec.shape ?? [data.length] });
      rawInputs.set(spec.name, data);
    }
    try {
      const outputs = await runGraph({ ...this.ctx(), manifest: this.manifest, inputBuffers, rawInputs });
      releaseOutputs(outputs);
      const result: Record<string, Float32Array> = {};
      for (const out of outputs) result[out.name] = out.data;
      return result;
    } finally {
      for (const t of inputBuffers.values()) t.buffer.destroy();
    }
  }

  /** profileGraph with the same inputs as run(): GPU-timestamped time per op class. */
  async profile(inputs: Record<string, Float32Array>): Promise<ProfileReport> {
    const inputBuffers = new Map<string, ResolvedTensor>();
    const rawInputs = new Map<string, Float32Array>();
    for (const spec of this.manifest.inputs) {
      const data = inputs[spec.name];
      if (!data) throw new Error(`missing input "${spec.name}"`);
      inputBuffers.set(spec.name, { buffer: uploadFloat32(this.device, data), shape: spec.shape ?? [data.length] });
      rawInputs.set(spec.name, data);
    }
    try {
      return await profileGraph({ ...this.ctx(), manifest: this.manifest, inputBuffers, rawInputs });
    } finally {
      for (const t of inputBuffers.values()) t.buffer.destroy();
    }
  }

  destroy(): void {
    for (const t of this.weightBuffers.values()) t.buffer.destroy();
    for (const b of this.constantCache.values()) b.destroy();
    this.device.destroy();
  }
}
