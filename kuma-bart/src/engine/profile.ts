import type { GraphNode } from "../types/manifest.js";
import { encodeGraph, type RunGraphParams } from "./scheduler.js";
import { TimestampProfiler } from "webgpu-compute-kit";

export interface OpTiming {
  name: string;
  target: string;
  microseconds: number;
}

export interface TargetTiming {
  target: string;
  totalMicroseconds: number;
  count: number;
  avgMicroseconds: number;
}

export interface ProfileReport {
  totalMicroseconds: number;
  /** One entry per dispatched node, in graph order. */
  perNode: OpTiming[];
  /** Same data grouped by aten target (op "class"), sorted by total time descending --
   * this is the "where do I focus" view. */
  byTarget: TargetTiming[];
}

/**
 * Same graph walk as runGraph (both use encodeGraph), but every dispatched node gets its
 * *own* compute pass with `timestampWrites` recording GPU-side start/end timestamps --
 * still one encoder and one submit, so no per-node CPU/GPU sync stalls. Requires the
 * "timestamp-query" GPU feature; throws a clear error rather than producing zeros when
 * it's missing.
 *
 * Per-dispatch passes are a hard WebGPU constraint, not a tuning choice: timestampWrites
 * attach only at pass boundaries, so per-node timing needs one pass per node. That runs
 * every dispatch in isolation, without the overlap consecutive dispatches get in a shared
 * pass, and has measured 3-4x *higher* than real per-frame GPU time. Treat byTarget's
 * percentages as the signal; KumaModel.profile() reports a comparable real number
 * (realMilliseconds) alongside this.
 */
export async function profileGraph(params: RunGraphParams): Promise<ProfileReport> {
  const { device } = params;
  if (!device.features.has("timestamp-query")) {
    throw new Error(
      "profileGraph requires the WebGPU 'timestamp-query' feature, which this adapter/browser doesn't support.",
    );
  }

  let profiler: TimestampProfiler | undefined;
  const run = await encodeGraph(params, {
    prepare: (dispatchNodes) => {
      profiler = new TimestampProfiler(device, dispatchNodes.length);
    },
    nodePass: (encoder, node) => profiler!.beginPass(encoder, node.name),
    beforeSubmit: (encoder) => profiler!.resolve(encoder),
  });

  // Wait for the frame's GPU work to complete; nothing is read back except timestamps.
  await device.queue.onSubmittedWorkDone();
  const timings = await profiler!.readResults();
  profiler!.destroy();
  // ProfileReport hands no buffer back, so every buffer this call allocated can go.
  run.freeOwned(new Set());

  const perNode: OpTiming[] = run.dispatchedNodes.map((node: GraphNode, i) => ({
    name: node.name,
    target: node.target,
    microseconds: timings[i]!.microseconds,
  }));

  const byTargetMap = new Map<string, { total: number; count: number }>();
  for (const t of perNode) {
    const entry = byTargetMap.get(t.target) ?? { total: 0, count: 0 };
    entry.total += t.microseconds;
    entry.count += 1;
    byTargetMap.set(t.target, entry);
  }
  const byTarget: TargetTiming[] = Array.from(byTargetMap.entries())
    .map(([target, { total, count }]) => ({ target, totalMicroseconds: total, count, avgMicroseconds: total / count }))
    .sort((a, b) => b.totalMicroseconds - a.totalMicroseconds);

  const totalMicroseconds = perNode.reduce((sum, t) => sum + t.microseconds, 0);
  return { totalMicroseconds, perNode, byTarget };
}
