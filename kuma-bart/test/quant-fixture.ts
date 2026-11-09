import { strToU8, zipSync } from "fflate";
import type { KumaManifest, WeightEntry } from "../src/types/manifest.js";

/** A TS twin of kuma.quantize.quantize_int8 (symmetric int8, one scale per dim-0 slice,
 * float32 throughout, round half to even), for building int8 fixtures without Python. */
export function quantizeInt8(values: Float32Array, shape: readonly number[]): { q: Int8Array; scales: Float32Array; groupSize: number } {
  const rows = shape.length > 1 ? shape[0]! : 1;
  const g = values.length / rows;
  const q = new Int8Array(values.length);
  const scales = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let max = 0;
    for (let i = r * g; i < (r + 1) * g; i++) max = Math.max(max, Math.abs(values[i]!));
    const s = max === 0 ? 1 : Math.fround(max / 127);
    scales[r] = s;
    for (let i = r * g; i < (r + 1) * g; i++) {
      const x = Math.fround(values[i]! / s);
      // round half to even, as np.round
      let n = Math.round(x);
      if (Math.abs(x % 1) === 0.5 && n % 2 !== 0) n -= 1;
      q[i] = Math.max(-127, Math.min(127, n));
    }
  }
  return { q, scales, groupSize: g };
}

/** Deterministic weights in [-amp, amp]. */
export function pattern(n: number, phase = 0, amp = 1): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.fround(Math.sin(0.731 * i + phase) * amp);
  return out;
}

/** Packs `manifest` (float32 weight entries without offsets) + `values` per weight into
 * a .iph zip -- int8 for names in `quantize` (format_version 1, the kuma.quantize
 * layout), float32 otherwise. `kernels` are written under kernels/. */
export function buildIph(
  manifest: KumaManifest,
  values: ReadonlyMap<string, Float32Array>,
  options: { quantize?: ReadonlySet<string>; kernels?: ReadonlyMap<string, string>; padInt8?: boolean } = {},
): ArrayBuffer {
  const f32: number[] = [];
  const q8: number[] = [];
  const entries: WeightEntry[] = [];
  const quantize = options.quantize ?? new Set<string>();
  for (const w of manifest.weights) {
    const v = values.get(w.name)!;
    if (!quantize.has(w.name)) {
      entries.push({ ...w, byte_offset: f32.length * 4, byte_length: v.length * 4 });
      f32.push(...v);
      continue;
    }
    const { q, scales, groupSize } = quantizeInt8(v, w.shape);
    const qOffset = q8.length;
    q8.push(...new Uint8Array(q.buffer));
    if (options.padInt8 !== false) while (q8.length % 4) q8.push(0);
    const sOffset = f32.length * 4;
    f32.push(...scales);
    entries.push({
      name: w.name,
      shape: w.shape,
      n_elements: v.length,
      dtype: "int8",
      file: "weights.q8.bin",
      byte_offset: qOffset,
      byte_length: v.length,
      quant: { scheme: "int8_symmetric", group_size: groupSize, scale_dtype: "float32", scale_offset: sOffset, scale_count: scales.length },
    });
  }
  const anyQuant = quantize.size > 0;
  const out: KumaManifest = {
    ...manifest,
    format_version: anyQuant ? 1 : 0,
    ...(anyQuant ? { quant_weight_file: "weights.q8.bin" } : {}),
    weights: entries,
  };
  const files: Record<string, Uint8Array> = {
    "manifest.json": strToU8(JSON.stringify(out)),
    "weights.f32.bin": new Uint8Array(new Float32Array(f32).buffer),
  };
  if (anyQuant) files["weights.q8.bin"] = new Uint8Array(q8);
  for (const [name, src] of options.kernels ?? []) files[`kernels/${name}`] = strToU8(src);
  return zipSync(files).slice().buffer as ArrayBuffer;
}
