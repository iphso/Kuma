import { unzipSync } from "fflate";
import { loadZipPack, PackError, type PackSource, type ZipPack } from "webgpu-compute-kit";
import { MAX_FORMAT_VERSION, type KumaManifest } from "../types/manifest.js";
import { widenQuantizedWeights } from "./weights.js";
import type { GoldenData } from "../types/golden.js";
import { KumaManifestError } from "../errors.js";

export interface IphPackage {
  manifest: KumaManifest;
  /** The manifest's weight_file: float32 tensors (and the scales of int8 ones). */
  weights: Uint8Array;
  /** The manifest's quant_weight_file (int8 tensors), when the package has one and it
   * was not widened on load. See iph/weights.ts. */
  quantWeights?: Uint8Array;
  /** kernel filename (e.g. "conv2d.wgsl") -> WGSL source, read from the package itself. */
  kernels: Map<string, string>;
  /** snippet filename (e.g. "route_by_time.js") -> JS source, read from the package
   * itself — see engine/snippets.ts for how these get executed. */
  snippets: Map<string, string>;
  /** Present only when the exporter had example inputs to capture it with — see
   * types/golden.ts. */
  golden?: GoldenData;
}

/** webgpu-compute-kit's loadZipPack, with its PackErrors surfaced as KumaManifestError. */
export async function loadIph<M extends { format?: string }>(
  source: PackSource,
  expectedFormat?: string,
): Promise<ZipPack<M>> {
  try {
    return await loadZipPack<M>(source, { expectedFormat, what: ".iph package" });
  } catch (err) {
    if (err instanceof PackError) throw new KumaManifestError(err.message);
    throw err;
  }
}

/** Read the `format` field from a zip's manifest.json, decompressing only that entry.
 * Accepts the already-fetched bytes so the caller doesn't pay for a second fetch. */
export function peekIphFormat(source: ArrayBuffer | Uint8Array): string | undefined {
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(bytes, { filter: (file) => file.name === "manifest.json" });
  } catch {
    return undefined;
  }
  const manifestBytes = files["manifest.json"];
  if (!manifestBytes) return undefined;
  try {
    const m = JSON.parse(new TextDecoder().decode(manifestBytes)) as { format?: unknown };
    return typeof m.format === "string" ? m.format : undefined;
  } catch {
    return undefined;
  }
}

export interface LoadIphOptions {
  /** Widen int8 weights to float32 on the CPU while loading (the returned package then
   * looks like a float32 one: no quantWeights, every manifest weight float32). Default
   * false: int8 weights are returned as-is for uploadWeights to keep int8 on the GPU. */
  widenQuantized?: boolean;
}

export async function loadIphPackage(source: ArrayBuffer | Response | string, options: LoadIphOptions = {}): Promise<IphPackage> {
  const loaded = await loadIph<KumaManifest>(source, "kuma");
  let manifest = loaded.manifest;
  const files = loaded.files;
  const version = manifest.format_version ?? 0;
  if (version > MAX_FORMAT_VERSION) {
    throw new KumaManifestError(
      `.iph format_version ${version} is newer than this kuma-bart reads (<= ${MAX_FORMAT_VERSION}); update kuma-bart.`,
    );
  }

  const weightFile = manifest.weight_file ?? "weights.f32.bin";
  let weights = files[weightFile];
  if (!weights) {
    throw new KumaManifestError(`.iph package is missing weight file "${weightFile}"`);
  }
  let quantWeights: Uint8Array | undefined;
  if (manifest.quant_weight_file) {
    quantWeights = files[manifest.quant_weight_file];
    if (!quantWeights) {
      throw new KumaManifestError(`.iph package is missing int8 weight file "${manifest.quant_weight_file}"`);
    }
    if (options.widenQuantized) {
      ({ manifest, weights } = widenQuantizedWeights({ manifest, weights, quantWeights }));
      quantWeights = undefined;
    }
  }

  const kernels = new Map<string, string>();
  const snippets = new Map<string, string>();
  const decoder = new TextDecoder();
  for (const [path, data] of Object.entries(files)) {
    if (path.startsWith("kernels/") && path.endsWith(".wgsl")) {
      kernels.set(path.slice("kernels/".length), decoder.decode(data));
    } else if (path.startsWith("snippets/") && path.endsWith(".js")) {
      snippets.set(path.slice("snippets/".length), decoder.decode(data));
    }
  }

  const goldenBytes = files["golden.json"];
  const golden = goldenBytes ? (JSON.parse(decoder.decode(goldenBytes)) as GoldenData) : undefined;

  return { manifest, weights, quantWeights, kernels, snippets, golden };
}
