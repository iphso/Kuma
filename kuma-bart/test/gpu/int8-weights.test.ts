import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { describeNvidiaGpu, nvidiaGpuLoad } from "webgpu-compute-kit/node-gpu";
import { loadIphPackage } from "../../src/iph/loader.js";
import { TerrainModel } from "./terrain-engine.js";

/**
 * int8 weights on the real GPU, against the terrain packages:
 *
 *   <stem>.iph     float32 package (Loom/models)
 *   <stem>.q8.iph  the same converted by `python -m kuma.quantize` (int8 weights)
 *   <stem>.fq.iph  the same fake-quantized by Loom/py/quant/fakequant.py int8 (float32
 *                  file holding the int8-rounded weights)
 *
 * Checks, per package:
 *   1. every weight of the q8 package, widened on load, equals fakequant's exactly
 *      (as float32 values: fakequant keeps np.round's -0.0 for tiny negative weights,
 *      int8 has no -0, so those come back +0 -- equal values, different sign bit);
 *   2. the q8 package widened on load ("widen") gives exactly the fakequant package's
 *      outputs (same weight values, same kernels);
 *   3. the q8 package kept int8 on the GPU ("gpu", *_q8 kernels) vs "widen": max |err|
 *      (the int8 kernels dequantize with the same f32 multiply, so expected 0);
 * and reports GPU weight bytes and forward time for f32 vs int8-on-GPU.
 *
 * Packages come from $INT8_MODELS (q8 + fq + ref files) and $LOOM_MODELS (float32);
 * missing ones skip.
 */

const LOOM_MODELS = process.env.LOOM_MODELS ?? resolve(import.meta.dirname, "../../../../Loom/models");
const INT8_MODELS = process.env.INT8_MODELS ?? resolve(import.meta.dirname, "../../../../Kuma-int8-models");
const FORWARD_ITERS = Number(process.env.TERRAIN_ITERS ?? 5);

interface RefEntry {
  name: string;
  shape: number[];
  byte_offset: number;
  byte_length: number;
}

function loadRefInputs(stem: string): Record<string, Float32Array> {
  const spec = JSON.parse(readFileSync(resolve(LOOM_MODELS, `${stem}.ref.json`), "utf8")) as { inputs: RefEntry[]; file: string };
  const bin = readFileSync(resolve(LOOM_MODELS, spec.file));
  const inputs: Record<string, Float32Array> = {};
  for (const e of spec.inputs) inputs[e.name] = new Float32Array(bin.buffer.slice(bin.byteOffset + e.byte_offset, bin.byteOffset + e.byte_offset + e.byte_length));
  return inputs;
}

function bytesOf(path: string): ArrayBuffer {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

/** `valueDiffs` counts elements whose float32 values differ (-0 == +0, NaN != NaN). */
function maxAbsDiff(a: Float32Array, b: Float32Array): { maxAbs: number; valueDiffs: number; refMax: number } {
  let maxAbs = 0;
  let valueDiffs = 0;
  let refMax = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) valueDiffs++;
    maxAbs = Math.max(maxAbs, Math.abs(a[i]! - b[i]!));
    refMax = Math.max(refMax, Math.abs(b[i]!));
  }
  return { maxAbs, valueDiffs, refMax };
}

async function forward(model: TerrainModel, inputs: Record<string, Float32Array>): Promise<{ out: Float32Array; medianMs: number }> {
  const first = await model.run(inputs);
  const out = first[Object.keys(first)[0]!]!;
  const times: number[] = [];
  for (let i = 0; i < FORWARD_ITERS; i++) {
    const t0 = performance.now();
    await model.run(inputs);
    times.push(performance.now() - t0);
  }
  times.sort((x, y) => x - y);
  return { out, medianMs: times[Math.floor(times.length / 2)]! };
}

const PACKAGES = ["coarse_model", "decoder_model_256", "decoder_model", "base_model"] as const;
const MB = (n: number) => `${(n / 1e6).toFixed(1)} MB`;

describe("int8 weights on the GPU", () => {
  for (const stem of PACKAGES) {
    const f32Path = resolve(LOOM_MODELS, `${stem}.iph`);
    const q8Path = resolve(INT8_MODELS, `${stem}.q8.iph`);
    const fqPath = resolve(INT8_MODELS, `${stem}.fq.iph`);
    const present = [f32Path, q8Path, fqPath, resolve(LOOM_MODELS, `${stem}.ref.json`)].every((p) => existsSync(p));

    it.skipIf(!present)(`${stem}: widened = fakequant bit for bit; int8 kernels vs widened; memory and time`, async () => {
      const load = await nvidiaGpuLoad(1);
      const inputs = loadRefInputs(stem);

      // 1. Weight-level parity: q8 widened on load vs fakequant's float32 weights.
      {
        const fq = await loadIphPackage(bytesOf(fqPath));
        const q8 = await loadIphPackage(bytesOf(q8Path), { widenQuantized: true });
        const fqByName = new Map(fq.manifest.weights.map((w) => [w.name, w]));
        expect(q8.manifest.weights.length).toBe(fq.manifest.weights.length);
        let differing = 0;
        for (const w of q8.manifest.weights) {
          const f = fqByName.get(w.name)!;
          expect(w.dtype).toBe("float32");
          expect(w.byte_length).toBe(f.byte_length);
          const a = new Float32Array(q8.weights.slice(w.byte_offset, w.byte_offset + w.byte_length).buffer);
          const b = new Float32Array(fq.weights.slice(f.byte_offset, f.byte_offset + f.byte_length).buffer);
          differing += maxAbsDiff(a, b).valueDiffs;
        }
        expect(differing).toBe(0);
      }

      // 2./3. Outputs: fakequant (f32) vs q8 widened vs q8 int8-on-GPU, plus the original.
      const runOne = async (path: string, mode: "gpu" | "widen") => {
        const model = await TerrainModel.load(bytesOf(path), { quantizedWeights: mode });
        try {
          const r = await forward(model, inputs);
          // float32 copies made for consumers without an int8 kernel (OpContext.widen).
          const widenedCopies = [...model.constantCache.keys()].filter((k) => k.startsWith("dequant:")).length;
          // GPU-timestamped time of the weight-consuming ops (one pass per node, so the
          // total runs high; the conv/linear split is the signal).
          let gpu = "";
          try {
            const prof = await model.profile(inputs);
            const ms = (t: string) => (prof.byTarget.find((x) => x.target === t)?.totalMicroseconds ?? 0) / 1000;
            gpu = `GPU ${(prof.totalMicroseconds / 1000).toFixed(1)} ms (conv2d ${ms("aten.conv2d.default").toFixed(1)}, linear ${ms("aten.linear.default").toFixed(2)})`;
          } catch (err) {
            gpu = `profile unavailable: ${(err as Error).message}`;
          }
          return { ...r, weightBytes: model.weightBytes, widenedCopies, gpu };
        } finally {
          model.destroy();
        }
      };
      const f32 = await runOne(f32Path, "gpu");
      const fq = await runOne(fqPath, "gpu");
      const widened = await runOne(q8Path, "widen");
      const int8 = await runOne(q8Path, "gpu");

      const widenVsFq = maxAbsDiff(widened.out, fq.out);
      const int8VsWiden = maxAbsDiff(int8.out, widened.out);
      const int8VsF32 = maxAbsDiff(int8.out, f32.out);
      console.log(
        `[int8] ${stem} on ${describeNvidiaGpu(load)}: package ${MB(readFileSync(f32Path).byteLength)} -> ${MB(readFileSync(q8Path).byteLength)}` +
          `\n  widened vs fakequant: ${widenVsFq.valueDiffs} of ${fq.out.length} outputs differ (max |err| ${widenVsFq.maxAbs.toExponential(2)})` +
          `\n  int8-on-GPU vs widened: max |err| ${int8VsWiden.maxAbs.toExponential(2)} (${int8VsWiden.valueDiffs} outputs differ), |y| max ${int8VsWiden.refMax.toFixed(3)}` +
          `\n  int8 vs float32 model (quantization error): max |err| ${int8VsF32.maxAbs.toExponential(2)}, |y| max ${int8VsF32.refMax.toFixed(3)}` +
          `\n  GPU weight bytes: f32 ${MB(f32.weightBytes)}, widened ${MB(widened.weightBytes)}, int8 ${MB(int8.weightBytes)}` +
          `\n  forward (run(), with readback, median of ${FORWARD_ITERS}): f32 ${f32.medianMs.toFixed(1)} ms, fakequant ${fq.medianMs.toFixed(1)} ms, widened ${widened.medianMs.toFixed(1)} ms, int8 ${int8.medianMs.toFixed(1)} ms` +
          `\n  profile: f32 ${f32.gpu}; int8 ${int8.gpu}`,
      );

      expect(widenVsFq.valueDiffs).toBe(0);
      expect(int8VsWiden.maxAbs).toBeLessThanOrEqual(1e-5 * Math.max(1, int8VsWiden.refMax));
      expect(int8.weightBytes).toBeLessThan(0.3 * f32.weightBytes);
      // Every int8 weight of these U-Nets feeds a linear / 1x1 / 3x3 conv: all int8 kernels.
      expect(int8.widenedCopies).toBe(0);
    });
  }
});
