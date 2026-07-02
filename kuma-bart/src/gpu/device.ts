/** Opt-in GPU-buffer leak tracker. Enable by loading the page with `?kumamem` in the URL
 * (or setting `localStorage.kumamem = "1"`). Wraps the device's createBuffer so every
 * allocation increments a live counter and the buffer's own destroy() decrements it, then
 * logs the running totals once a second -- broken down by usage class (storage = weights/
 * activations, uniform = packed params, mapread = readback staging). A healthy steady
 * state plateaus; a per-frame leak shows live count/bytes climbing monotonically. Off (and
 * zero-overhead) unless explicitly enabled, so it's safe to ship. */
function memDebugEnabled(): boolean {
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem("kumamem")) return true;
  } catch {
    /* localStorage can throw in sandboxed contexts -- ignore */
  }
  return typeof location !== "undefined" && /[?&]kumamem\b/.test(location.search);
}

interface MemStats {
  live: number;
  bytes: number;
  liveByUsage: Record<string, number>;
  bytesByUsage: Record<string, number>;
}

function usageClass(usage: number): string {
  if (usage & GPUBufferUsage.MAP_READ) return "mapread";
  if (usage & GPUBufferUsage.UNIFORM) return "uniform";
  return "storage";
}

function instrumentDevice(device: GPUDevice): void {
  const stats: MemStats = { live: 0, bytes: 0, liveByUsage: {}, bytesByUsage: {} };
  (globalThis as unknown as { __kumaMem?: MemStats }).__kumaMem = stats;

  const realCreate = device.createBuffer.bind(device);
  device.createBuffer = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = realCreate(descriptor);
    const cls = usageClass(descriptor.usage);
    const size = descriptor.size;
    stats.live++;
    stats.bytes += size;
    stats.liveByUsage[cls] = (stats.liveByUsage[cls] ?? 0) + 1;
    stats.bytesByUsage[cls] = (stats.bytesByUsage[cls] ?? 0) + size;
    let freed = false;
    const realDestroy = buffer.destroy.bind(buffer);
    buffer.destroy = (): undefined => {
      if (!freed) {
        freed = true;
        stats.live--;
        stats.bytes -= size;
        stats.liveByUsage[cls]! -= 1;
        stats.bytesByUsage[cls]! -= size;
      }
      return realDestroy();
    };
    return buffer;
  };

  setInterval(() => {
    const mb = (stats.bytes / (1024 * 1024)).toFixed(1);
    const byUsage = Object.keys(stats.liveByUsage)
      .map((k) => `${k}=${stats.liveByUsage[k]}(${((stats.bytesByUsage[k] ?? 0) / (1024 * 1024)).toFixed(1)}MB)`)
      .join(" ");
    console.log(`[kuma-mem] live buffers: ${stats.live} (${mb}MB) — ${byUsage}`);
  }, 1000);
}

/** Heuristic device-memory-tier guess, used by model.ts to default new KumaModel loads
 * into low-memory posture (see KumaMemoryOptions.lowMemory) on devices likely to be
 * VRAM/RAM-constrained. WebGPU has no portable "how much free VRAM is there" query, so
 * this combines the two real signals that do exist:
 *  - adapter.info.isFallbackAdapter: a software/emulated adapter -- definitely
 *    constrained, no real GPU memory management to speak of.
 *  - navigator.deviceMemory (Device Memory API, Chromium-only, capped at 8 for privacy):
 *    approximate *system* RAM in GiB. On the integrated-GPU/shared-memory devices this
 *    heuristic actually targets (Chromebooks, low-end laptops), system RAM doubles as the
 *    GPU's memory pool, so a low reading here is a reasonable proxy even though it isn't
 *    literally VRAM. Threshold of 4 GiB is chosen to catch entry-level Chromebooks/tablets
 *    without misclassifying typical 8GiB+ laptops (which the API reports as exactly 8,
 *    its reporting ceiling).
 * Both signals are best-effort and can be wrong in either direction on unusual hardware --
 * this only sets a *default*, always overridable via explicit KumaMemoryOptions fields. */
export function isLikelyLowMemoryDevice(adapter: GPUAdapter): boolean {
  if (adapter.info?.isFallbackAdapter) return true;
  const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof deviceMemory === "number" && deviceMemory <= 4;
}

export async function requestKumaDevice(): Promise<{ device: GPUDevice; isLikelyLowMemory: boolean }> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    throw new Error(
      "WebGPU is not available in this environment (no navigator.gpu). " +
        "kuma-bart requires a browser with WebGPU support (e.g. Chrome/Edge 113+).",
    );
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter request failed — no compatible GPU adapter found.");
  }
  // Opportunistic: only enables GPU-side profiling (engine/profile.ts) when the
  // adapter actually supports it -- most desktop Chrome/Edge GPUs do, but this must be
  // requested at device-creation time (features can't be added afterward), so it's
  // requested here unconditionally rather than only when profiling is first used.
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  if (memDebugEnabled()) instrumentDevice(device);
  return { device, isLikelyLowMemory: isLikelyLowMemoryDevice(adapter) };
}
