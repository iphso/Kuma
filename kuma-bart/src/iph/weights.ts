import { uploadBufferSlice } from "webgpu-compute-kit";
import type { KumaManifest, WeightEntry } from "../types/manifest.js";
import type { ResolvedTensor } from "../engine/context.js";
import { KumaManifestError } from "../errors.js";

/** The weight blobs of a loaded package: `weights` is the manifest's weight_file (float32
 * tensors, plus the scales of int8 ones), `quantWeights` its quant_weight_file (int8). */
export interface WeightBlobs {
  manifest: KumaManifest;
  weights: Uint8Array;
  quantWeights?: Uint8Array;
}

/** Kernels a package must ship for its int8 weights to stay int8 on the GPU: the widening
 * kernel every non-int8 consumer falls back to. The *_q8 GEMM kernels are optional (a
 * consumer without one widens). */
export const DEQUANT_KERNEL = "dequant_q8.wgsl";

export function isQuantized(entry: WeightEntry): boolean {
  return entry.quant !== undefined;
}

export function hasQuantizedWeights(manifest: KumaManifest): boolean {
  return manifest.weights.some(isQuantized);
}

function quantBlob(blobs: WeightBlobs, entry: WeightEntry): Uint8Array {
  if (!blobs.quantWeights) {
    throw new KumaManifestError(
      `weight "${entry.name}" is int8 but the package has no "${entry.file ?? blobs.manifest.quant_weight_file}"`,
    );
  }
  return blobs.quantWeights;
}

function checkQuant(entry: WeightEntry): NonNullable<WeightEntry["quant"]> {
  const q = entry.quant!;
  if (q.scheme !== "int8_symmetric" || entry.dtype !== "int8") {
    throw new KumaManifestError(`weight "${entry.name}": unsupported quantization ${entry.dtype}/${q.scheme}`);
  }
  if (q.group_size <= 0 || q.group_size * q.scale_count !== entry.n_elements) {
    throw new KumaManifestError(
      `weight "${entry.name}": ${q.scale_count} scales x group ${q.group_size} != ${entry.n_elements} elements`,
    );
  }
  return q;
}

/** A float32 copy of `byteLength` bytes at `byteOffset` (the blob may sit at any
 * alignment inside the unzipped archive's buffer). */
function f32At(blob: Uint8Array, byteOffset: number, byteLength: number): Float32Array {
  return new Float32Array(blob.slice(byteOffset, byteOffset + byteLength).buffer);
}

/** An int8 weight widened on the CPU: f32(q[i]) * scale[floor(i / group_size)], rounded
 * once to float32 (q * scale is exact in a double: 8 x 24 significant bits), i.e. exactly
 * the single f32 multiply the GPU kernels and kuma.quantize.dequantize_int8 do. */
export function dequantizeWeight(blobs: WeightBlobs, entry: WeightEntry): Float32Array {
  const q = checkQuant(entry);
  const src = quantBlob(blobs, entry);
  const values = new Int8Array(src.buffer, src.byteOffset + entry.byte_offset, entry.n_elements);
  const scales = f32At(blobs.weights, q.scale_offset, q.scale_count * 4);
  const out = new Float32Array(entry.n_elements);
  const g = q.group_size;
  for (let row = 0, i = 0; row < q.scale_count; row++) {
    const s = scales[row]!;
    for (const end = i + g; i < end; i++) out[i] = values[i]! * s;
  }
  return out;
}

/** Step one of int8 support: the same package with every int8 weight widened to float32
 * on the CPU (download stays 4x smaller; GPU memory and kernels are the float32 ones).
 * The returned manifest lists only float32 weights, all in the returned `weights` blob. */
export function widenQuantizedWeights(blobs: WeightBlobs): { manifest: KumaManifest; weights: Uint8Array } {
  if (!hasQuantizedWeights(blobs.manifest)) return { manifest: blobs.manifest, weights: blobs.weights };
  let total = blobs.weights.byteLength;
  for (const w of blobs.manifest.weights) if (isQuantized(w)) total += w.n_elements * 4;
  const out = new Uint8Array(total);
  out.set(blobs.weights, 0);
  let offset = blobs.weights.byteLength; // a multiple of 4: every packed tensor is 4-aligned and 4-sized
  const entries: WeightEntry[] = [];
  for (const w of blobs.manifest.weights) {
    if (!isQuantized(w)) {
      entries.push(w);
      continue;
    }
    const values = dequantizeWeight(blobs, w);
    out.set(new Uint8Array(values.buffer), offset);
    entries.push({ name: w.name, shape: w.shape, dtype: "float32", n_elements: w.n_elements, byte_offset: offset, byte_length: values.byteLength });
    offset += values.byteLength;
  }
  const { quant_weight_file: _dropped, ...rest } = blobs.manifest;
  return { manifest: { ...rest, weights: entries }, weights: out };
}

/** Uploads every weight to its own GPU buffer. float32 tensors as-is; int8 tensors stay
 * int8 (four per u32, `ResolvedTensor.quant` carrying the scales), unless the blobs were
 * widened first (widenQuantizedWeights). */
export function uploadWeights(device: GPUDevice, blobs: WeightBlobs): Map<string, ResolvedTensor> {
  const out = new Map<string, ResolvedTensor>();
  for (const w of blobs.manifest.weights) {
    if (!isQuantized(w)) {
      out.set(w.name, { buffer: uploadBufferSlice(device, blobs.weights, w.byte_offset, w.byte_length), shape: w.shape });
      continue;
    }
    const q = checkQuant(w);
    const src = quantBlob(blobs, w);
    const padded = Math.ceil(w.byte_length / 4) * 4;
    let buffer: GPUBuffer;
    if (w.byte_offset + padded <= src.byteLength) {
      buffer = uploadBufferSlice(device, src, w.byte_offset, padded);
    } else {
      // A writer that didn't pad the last tensor to 4 bytes: pad a copy.
      const copy = new Uint8Array(padded);
      copy.set(src.subarray(w.byte_offset, w.byte_offset + w.byte_length));
      buffer = uploadBufferSlice(device, copy, 0, padded);
    }
    const scales = uploadBufferSlice(device, blobs.weights, q.scale_offset, q.scale_count * 4);
    out.set(w.name, { buffer, shape: w.shape, quant: { name: w.name, groupSize: q.group_size, scales } });
  }
  return out;
}

/** GPU bytes held by a weight map (int8 payloads, scales, float32 tensors). */
export function weightBytes(weights: ReadonlyMap<string, ResolvedTensor>): number {
  let bytes = 0;
  for (const t of weights.values()) bytes += t.buffer.size + (t.quant?.scales.size ?? 0);
  return bytes;
}

/** Frees a weight map's buffers, scales included. */
export function destroyWeights(weights: ReadonlyMap<string, ResolvedTensor>): void {
  for (const t of weights.values()) {
    t.buffer.destroy();
    t.quant?.scales.destroy();
  }
}
