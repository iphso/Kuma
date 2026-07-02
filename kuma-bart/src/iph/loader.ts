import { unzipSync } from "fflate";
import type { KumaManifest } from "../types/manifest.js";
import type { GoldenData } from "../types/golden.js";
import { KumaManifestError } from "../errors.js";

export interface IphPackage {
  manifest: KumaManifest;
  weights: Uint8Array;
  /** kernel filename (e.g. "conv2d.wgsl") -> WGSL source, read from the package itself. */
  kernels: Map<string, string>;
  /** snippet filename (e.g. "route_by_time.js") -> JS source, read from the package
   * itself — see engine/snippets.ts for how these get executed. */
  snippets: Map<string, string>;
  /** Present only when the exporter had example inputs to capture it with — see
   * types/golden.ts. */
  golden?: GoldenData;
}

async function toBytes(source: ArrayBuffer | Response | string): Promise<Uint8Array> {
  if (typeof source === "string") {
    const res = await fetch(source);
    if (!res.ok) {
      throw new KumaManifestError(
        `Failed to fetch .iph package from "${source}": ${res.status} ${res.statusText}`,
      );
    }
    return new Uint8Array(await res.arrayBuffer());
  }
  if (source instanceof Response) {
    if (!source.ok) {
      throw new KumaManifestError(`Failed to fetch .iph package: ${source.status} ${source.statusText}`);
    }
    return new Uint8Array(await source.arrayBuffer());
  }
  return new Uint8Array(source);
}

/** Read the `format` field from a zip's manifest.json without fully parsing the package.
 * Accepts the already-fetched bytes so the caller doesn't pay for a second fetch. */
export function peekIphFormat(source: ArrayBuffer | Uint8Array): string | undefined {
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
  let files: Record<string, Uint8Array>;
  try { files = unzipSync(bytes); } catch { return undefined; }
  const manifestBytes = files["manifest.json"];
  if (!manifestBytes) return undefined;
  try {
    const m = JSON.parse(new TextDecoder().decode(manifestBytes)) as { format?: unknown };
    return typeof m.format === "string" ? m.format : undefined;
  } catch { return undefined; }
}

export async function loadIphPackage(source: ArrayBuffer | Response | string): Promise<IphPackage> {
  const bytes = await toBytes(source);

  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(bytes);
  } catch (err) {
    throw new KumaManifestError(`Failed to unzip .iph package: ${(err as Error).message}`);
  }

  const manifestBytes = files["manifest.json"];
  if (!manifestBytes) {
    throw new KumaManifestError('.iph package is missing "manifest.json"');
  }
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as KumaManifest;

  if (manifest.format !== "kuma") {
    throw new KumaManifestError(`Unrecognized package format "${manifest.format}" (expected "kuma")`);
  }

  const weightFile = manifest.weight_file ?? "weights.f32.bin";
  const weights = files[weightFile];
  if (!weights) {
    throw new KumaManifestError(`.iph package is missing weight file "${weightFile}"`);
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

  return { manifest, weights, kernels, snippets, golden };
}
