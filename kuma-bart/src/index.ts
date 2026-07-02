export { KumaModel } from "./model.js";
export type { KumaMemoryOptions } from "./model.js";
export { OnnxModel } from "./onnx/model.js";
export { KumaPlayer } from "./player/KumaPlayer.js";
export { KumaManifestError, KumaShapeError, KumaUnsupportedOpError } from "./errors.js";
// Exposed so a caller pipelining multiple in-flight run()/runRaw()/runToGpu() calls
// (e.g. demo/main.ts's MAX_IN_FLIGHT) can size its own concurrency cap against this --
// submitting more calls concurrently than this doesn't get any more frames "in flight"
// in practice, since runGraph's own pooled-buffer reuse internally blocks (awaiting
// the GPU's real completion signal) once exceeded. See context.ts's BufferPoolState.
export { BUFFER_POOL_DEPTH } from "./engine/context.js";
export type { KumaManifest, IOSpec, WeightEntry, GraphNode, NodeRef, ArgValue, PlaybackMeta } from "./types/manifest.js";
export type { GoldenData, GoldenBranch, GoldenNodeStats, GoldenTensorStats } from "./types/golden.js";
export type { VerifyReport, BranchVerifyReport, NodeDiff, FieldDiff } from "./engine/verify.js";
export type { ProfileReport, OpTiming, TargetTiming } from "./engine/profile.js";
export type { RunGraphOutput } from "./engine/scheduler.js";
