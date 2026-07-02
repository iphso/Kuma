import { unzipSync } from "fflate";
import type { InferenceSession, Tensor } from "onnxruntime-web";
import { KumaManifestError } from "../errors.js";
import type { PlaybackMeta } from "../types/manifest.js";

// Lazy import so onnxruntime-web is only pulled in when OnnxModel is actually used.
async function getOrt() {
  const ort = await import("onnxruntime-web");
  // Steer WASM fetches to root-relative URLs served by the Vite dev-server middleware
  // (see vite.demo.config.ts onnxWasmPlugin).  Must be set before the first
  // InferenceSession.create() call; safe to set repeatedly since it's idempotent.
  ort.env.wasm.wasmPaths = "/";
  // Single-threaded mode avoids the SharedArrayBuffer requirement (needs COOP/COEP
  // response headers that the Vite dev server doesn't send by default).
  ort.env.wasm.numThreads = 1;
  return ort;
}

interface OnnxBranchingManifest {
  format: "onnx-branching";
  total_frames: number;
  segment_size: number;
  num_segments: number;
  inputs: Array<{ name: string }>;
  outputs: Array<{ name: string; shape?: number[] }>;
  playback?: PlaybackMeta;
  graph: { nodes: Array<{ op: string; branches?: Array<{ model_file: string }> }> };
}

interface OnnxSingleManifest {
  format: "onnx";
  total_frames?: number;
  inputs: Array<{ name: string }>;
  outputs: Array<{ name: string; shape?: number[] }>;
  playback?: PlaybackMeta;
  graph: { nodes: Array<{ op: string; model_file?: string }> };
}

type OnnxManifest = OnnxBranchingManifest | OnnxSingleManifest;

function routeFrame(
  normT: number,
  totalFrames: number,
  segmentSize: number,
  numSegments: number,
): { segId: number; localNorm: number } {
  const position = Math.min(Math.max(normT * Math.max(totalFrames - 1, 1), 0), totalFrames - 1);
  const frameId = Math.floor(position + 1e-4);
  const segId = Math.min(Math.floor(frameId / segmentSize), numSegments - 1);
  const segStart = segId * segmentSize;
  const segEnd = Math.min((segId + 1) * segmentSize, totalFrames);
  const denom = Math.max(segEnd - segStart - 1, 1);
  const localNorm = Math.min(Math.max((position - segStart) / denom, 0), 1);
  return { segId, localNorm };
}

async function toBytes(source: ArrayBuffer | Response | string): Promise<Uint8Array> {
  if (typeof source === "string") {
    const res = await fetch(source);
    if (!res.ok) throw new KumaManifestError(`Failed to fetch .iph from "${source}": ${res.status} ${res.statusText}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  if (source instanceof Response) {
    if (!source.ok) throw new KumaManifestError(`Failed to fetch .iph: ${source.status} ${source.statusText}`);
    return new Uint8Array(await source.arrayBuffer());
  }
  return new Uint8Array(source);
}

/** A loaded ONNX-format `.iph` package, run via onnxruntime-web. */
export class OnnxModel {
  /** Raw ONNX bytes per segment index — sessions are created lazily on first use. */
  private readonly sessionBytes: Map<number, Uint8Array>;
  /** Session cache — populated on demand, never pre-filled to avoid loading every
   * segment's weights into GPU memory upfront (which crashes the browser for large
   * multi-segment models). */
  private readonly sessionCache = new Map<number, InferenceSession>();
  /** Inflight creation promises so concurrent run() calls don't double-create. */
  private readonly sessionPending = new Map<number, Promise<InferenceSession>>();

  private constructor(
    sessionBytes: Map<number, Uint8Array>,
    private readonly manifest: OnnxManifest,
    private readonly sessionOptions: InferenceSession.SessionOptions | undefined,
  ) {
    this.sessionBytes = sessionBytes;
  }

  /** Pre-compile all segments serially so the WebGPU shader compilation cost is
   * paid during a known loading phase, not mid-playback. */
  /** Pre-compile all segments serially so the WebGPU shader compilation cost is
   * paid during a known loading phase, not mid-playback.
   *
   * ort compiles shaders lazily on the FIRST session.run(), not during
   * InferenceSession.create() — so we must run a dummy inference per segment here
   * to force compilation now.  Chrome caches compiled pipelines on disk, so
   * subsequent loads of the same model skip compilation entirely. */
  async warmAll(onProgress?: (done: number, total: number) => void): Promise<void> {
    const ort = await getOrt();
    const ids = Array.from(this.sessionBytes.keys()).sort((a, b) => a - b);
    for (let i = 0; i < ids.length; i++) {
      const session = await this.getSession(ids[i]!);
      // Dummy run with a single zero-valued input — forces shader compilation.
      const dummy = new ort.Tensor("float32", new Float32Array([0.0]), [1]);
      await session.run({ [session.inputNames[0]!]: dummy });
      onProgress?.(i + 1, ids.length);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }

  private async getSession(segId: number): Promise<InferenceSession> {
    const cached = this.sessionCache.get(segId);
    if (cached) return cached;

    let pending = this.sessionPending.get(segId);
    if (!pending) {
      const bytes = this.sessionBytes.get(segId);
      if (!bytes) throw new KumaManifestError(`Missing ONNX model bytes for segment ${segId}`);
      const ort = await getOrt();
      pending = ort.InferenceSession.create(bytes.buffer as ArrayBuffer, this.sessionOptions).then((s) => {
        this.sessionCache.set(segId, s);
        this.sessionPending.delete(segId);
        return s;
      });
      this.sessionPending.set(segId, pending);
    }
    return pending;
  }

  static async load(
    source: ArrayBuffer | Response | string,
    sessionOptions?: InferenceSession.SessionOptions,
  ): Promise<OnnxModel> {
    await getOrt(); // initialise env (wasmPaths, numThreads) before anything else

    const bytes = await toBytes(source);

    let files: Record<string, Uint8Array>;
    try {
      files = unzipSync(bytes);
    } catch (err) {
      throw new KumaManifestError(`Failed to unzip .iph: ${(err as Error).message}`);
    }

    const manifestBytes = files["manifest.json"];
    if (!manifestBytes) throw new KumaManifestError('.iph is missing "manifest.json"');
    const raw = JSON.parse(new TextDecoder().decode(manifestBytes)) as { format?: string };

    if (raw.format !== "onnx" && raw.format !== "onnx-branching") {
      throw new KumaManifestError(
        `OnnxModel requires format "onnx" or "onnx-branching", got "${raw.format}". Use KumaModel for WebGPU/WGSL packages.`,
      );
    }
    const manifest = raw as unknown as OnnxManifest;

    const sessionBytes = new Map<number, Uint8Array>();

    if (manifest.format === "onnx-branching") {
      const switchNode = manifest.graph.nodes.find((n) => n.op === "switch");
      if (!switchNode?.branches?.length) {
        throw new KumaManifestError('onnx-branching manifest is missing a "switch" node with branches');
      }
      for (let i = 0; i < switchNode.branches.length; i++) {
        const modelFile = switchNode.branches[i]!.model_file;
        const onnxBytes = files[modelFile];
        if (!onnxBytes) throw new KumaManifestError(`Missing ONNX model file "${modelFile}"`);
        sessionBytes.set(i, onnxBytes);
      }
    } else {
      const modelFile =
        manifest.graph.nodes.find((n) => n.model_file)?.model_file ??
        Object.keys(files).find((p) => p.endsWith(".onnx"));
      if (!modelFile) throw new KumaManifestError("No ONNX model file found in .iph");
      const onnxBytes = files[modelFile];
      if (!onnxBytes) throw new KumaManifestError(`Missing ONNX model file "${modelFile}"`);
      sessionBytes.set(0, onnxBytes);
    }

    return new OnnxModel(sessionBytes, manifest, sessionOptions);
  }

  get inputs(): Array<{ name: string; shape?: number[] }> {
    return this.manifest.inputs as Array<{ name: string; shape?: number[] }>;
  }

  get outputs(): Array<{ name: string; shape?: number[] }> {
    return this.manifest.outputs;
  }

  get playback(): PlaybackMeta | undefined {
    return this.manifest.playback;
  }

  get totalFrames(): number | undefined {
    return "total_frames" in this.manifest ? this.manifest.total_frames : undefined;
  }

  /** Releases every ORT session created so far (frees their WASM/GPU-side resources),
   * in one call. Call this when done with a model (e.g. right before loading a
   * different one into the same player) instead of just dropping the reference --
   * sessionCache otherwise keeps every segment's session (and its GPU memory) alive
   * until JS GC gets around to it. Safe even if no session was ever created yet
   * (warmAll()/run() create them lazily) and safe to call more than once. */
  async destroy(): Promise<void> {
    const sessions = Array.from(this.sessionCache.values());
    this.sessionCache.clear();
    this.sessionPending.clear();
    await Promise.all(sessions.map((s) => s.release()));
  }

  async run(inputs: Record<string, Float32Array>): Promise<Record<string, { data: Float32Array; shape: number[] }>> {
    const ort = await getOrt();
    const { manifest } = this;

    if (manifest.format === "onnx-branching") {
      const normTData = inputs["norm_t"];
      if (!normTData) throw new KumaManifestError('Missing input "norm_t"');

      const { total_frames, segment_size, num_segments } = manifest;
      const routes = Array.from(normTData).map((t) => routeFrame(t, total_frames, segment_size, num_segments));

      const bySegment = new Map<number, Array<{ idx: number; localNorm: number }>>();
      for (let i = 0; i < routes.length; i++) {
        const { segId, localNorm } = routes[i]!;
        if (!bySegment.has(segId)) bySegment.set(segId, []);
        bySegment.get(segId)!.push({ idx: i, localNorm });
      }

      const results = new Array<Float32Array>(normTData.length);
      let outShape: number[] = [];

      for (const [segId, items] of bySegment) {
        const session = await this.getSession(segId);
        const localNormT = new Float32Array(items.map((it) => it.localNorm));
        const inputName = session.inputNames[0]!;
        const outputName = session.outputNames[0]!;
        const feeds: Record<string, Tensor> = {
          [inputName]: new ort.Tensor("float32", localNormT, [items.length]),
        };
        const out = await session.run(feeds);
        const outTensor = out[outputName]!;
        // dims is the real shape from ort — e.g. [batch, C, H, W]; slice off batch dim
        outShape = Array.from(outTensor.dims).slice(1) as number[];
        const outData = outTensor.data as Float32Array;
        const elPerFrame = outData.length / items.length;
        for (let j = 0; j < items.length; j++) {
          results[items[j]!.idx] = outData.slice(j * elPerFrame, (j + 1) * elPerFrame);
        }
      }

      const totalElements = results.reduce((sum, r) => sum + r.length, 0);
      const output = new Float32Array(totalElements);
      let offset = 0;
      for (const r of results) {
        output.set(r, offset);
        offset += r.length;
      }
      const outName = manifest.outputs[0]!.name;
      return { [outName]: { data: output, shape: outShape } };
    } else {
      const session = await this.getSession(0);
      const inputName = session.inputNames[0]!;
      const normT = inputs[inputName] ?? inputs["norm_t"];
      if (!normT) throw new KumaManifestError(`Missing input "${inputName}"`);
      const feeds: Record<string, Tensor> = {
        [inputName]: new ort.Tensor("float32", normT, [normT.length]),
      };
      const out = await session.run(feeds);
      const result: Record<string, { data: Float32Array; shape: number[] }> = {};
      for (const name of session.outputNames) {
        result[name] = {
          data: out[name]!.data as Float32Array,
          shape: Array.from(out[name]!.dims).slice(1) as number[],
        };
      }
      return result;
    }
  }
}
