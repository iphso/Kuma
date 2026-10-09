import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { describeNvidiaGpu, nvidiaGpuLoad } from "webgpu-compute-kit/node-gpu";
import { TerrainModel } from "./terrain-engine.js";

/**
 * Runs the terrain-diffusion packages exported by Loom/py/export_terrain.py on the real
 * GPU: kuma-bart's own golden verification (per-node summary stats, allclose with
 * atol 4e-4 / rtol 2e-2), then an exact comparison of the full output against the
 * eager-PyTorch reference the exporter writes next to each package
 * (<name>.ref.json + <name>.ref.bin), and wall time per forward at pipeline shapes.
 *
 * The engine is driven through test/gpu/terrain-engine.ts (a device with raised
 * binding/buffer limits) rather than KumaModel -- see that file for why.
 *
 * Packages are looked up in $LOOM_MODELS (default: ../Loom/models next to this repo's
 * parent); a missing package skips its test rather than failing.
 */

const MODELS_DIR = process.env.LOOM_MODELS ?? resolve(import.meta.dirname, "../../../../Loom/models");
const FORWARD_ITERS = Number(process.env.TERRAIN_ITERS ?? 5);

interface RefEntry {
  name: string;
  shape: number[];
  byte_offset: number;
  byte_length: number;
}
interface RefSpec {
  inputs: RefEntry[];
  output: RefEntry;
  file: string;
}

function loadRef(stem: string): { inputs: Record<string, Float32Array>; output: Float32Array; outputShape: number[] } {
  const spec = JSON.parse(readFileSync(resolve(MODELS_DIR, `${stem}.ref.json`), "utf8")) as RefSpec;
  const bin = readFileSync(resolve(MODELS_DIR, spec.file));
  const slice = (e: RefEntry) => new Float32Array(bin.buffer.slice(bin.byteOffset + e.byte_offset, bin.byteOffset + e.byte_offset + e.byte_length));
  const inputs: Record<string, Float32Array> = {};
  for (const e of spec.inputs) inputs[e.name] = slice(e);
  return { inputs, output: slice(spec.output), outputShape: spec.output.shape };
}

function errorStats(actual: Float32Array, ref: Float32Array) {
  let maxAbs = 0;
  let sumAbs = 0;
  let refMax = 0;
  let maxRel = 0; // elementwise |a-r|/|r| where |r| >= 1% of the reference's max magnitude
  let nonFinite = 0;
  for (let i = 0; i < ref.length; i++) refMax = Math.max(refMax, Math.abs(ref[i]!));
  const relFloor = 0.01 * refMax;
  for (let i = 0; i < ref.length; i++) {
    const a = actual[i]!;
    const r = ref[i]!;
    if (!Number.isFinite(a)) {
      nonFinite++;
      continue;
    }
    const d = Math.abs(a - r);
    maxAbs = Math.max(maxAbs, d);
    sumAbs += d;
    if (Math.abs(r) >= relFloor) maxRel = Math.max(maxRel, d / Math.abs(r));
  }
  return { maxAbs, meanAbs: sumAbs / ref.length, refMax, maxRelToScale: maxAbs / refMax, maxRel, nonFinite };
}

function fmt(stats: ReturnType<typeof errorStats>): string {
  return (
    `max|err| ${stats.maxAbs.toExponential(2)}, mean|err| ${stats.meanAbs.toExponential(2)}, ` +
    `ref max|y| ${stats.refMax.toFixed(3)}, max|err|/max|y| ${stats.maxRelToScale.toExponential(2)}, ` +
    `max elementwise rel (|y|>=1% of max) ${stats.maxRel.toExponential(2)}, non-finite ${stats.nonFinite}`
  );
}

const PACKAGES = ["coarse_model", "decoder_model_256", "decoder_model", "base_model"] as const;

describe("terrain-diffusion packages on the GPU", () => {
  for (const stem of PACKAGES) {
    const iph = resolve(MODELS_DIR, `${stem}.iph`);
    const present = existsSync(iph) && existsSync(resolve(MODELS_DIR, `${stem}.ref.json`));
    it.skipIf(!present)(`${stem}: verify, output error vs eager torch, forward time`, async () => {
      const load = await nvidiaGpuLoad(1);
      const bytes = readFileSync(iph);
      const tLoad0 = performance.now();
      const model = await TerrainModel.load(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      const loadMs = performance.now() - tLoad0;
      const report = await model.verify();
      const branch = report.branches[0]!;
      const summary = branch.mismatches.slice(0, 5).map((m) => `${m.node}: ${m.diffs.map((d) => `${d.field} golden=${d.golden} actual=${d.actual}`).join("; ")}`);

      const ref = loadRef(stem);
      const out = await model.run(ref.inputs);
      const outName = Object.keys(out)[0]!;
      const stats = errorStats(out[outName]!, ref.output);

      // Wall time per forward (run() includes the output readback), after one warm pass.
      const times: number[] = [];
      for (let i = 0; i < FORWARD_ITERS; i++) {
        const t0 = performance.now();
        await model.run(ref.inputs);
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      const median = times[Math.floor(times.length / 2)]!;

      // GPU-timestamped per-op-class breakdown (percentages are the signal; the total
      // runs high because every node gets its own pass).
      let profileLine = "";
      try {
        const prof = await model.profile(ref.inputs);
        const top = prof.byTarget.slice(0, 6).map((t) => `${t.target} ${(100 * t.totalMicroseconds / prof.totalMicroseconds).toFixed(0)}% (${t.count}x, ${(t.totalMicroseconds / 1000).toFixed(1)} ms)`);
        profileLine = `\n  profile: ${(prof.totalMicroseconds / 1000).toFixed(1)} ms GPU over ${prof.perNode.length} passes; top: ${top.join(", ")}`;
      } catch (err) {
        profileLine = `\n  profile unavailable: ${(err as Error).message}`;
      }

      console.log(
        `[terrain] ${stem} on ${describeNvidiaGpu(load)}: package ${(bytes.byteLength / 1e6).toFixed(1)} MB, load ${loadMs.toFixed(0)} ms, ` +
          `limits storage-binding ${(model.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB / buffer ${(model.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB, ` +
          `verify ${report.ok ? "ok" : "MISMATCH"} (${branch.nodesChecked} nodes, ${branch.mismatches.length} mismatching, ${branch.nodesMissing.length} missing)` +
          (summary.length ? `\n  first mismatches: ${summary.join("\n    ")}` : "") +
          `\n  output ${outName} ${JSON.stringify(ref.outputShape)}: ${fmt(stats)}` +
          `\n  forward (run(), with readback): median ${median.toFixed(1)} ms, min ${times[0]!.toFixed(1)} ms over ${FORWARD_ITERS}` +
          profileLine,
      );

      expect(stats.nonFinite).toBe(0);
      expect(report.ok).toBe(true);
      // Output-level tolerance: float32 reduction-order differences across ~100 convs
      // stay well under 1% of the output scale; anything larger is a real bug.
      expect(stats.maxRelToScale).toBeLessThan(1e-2);
      model.destroy();
    });
  }
});
