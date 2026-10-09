export { KumaModel } from "./model.js";
export type { KumaMemoryOptions, KumaLoadOptions, KumaEncodeOptions, EncodedRun } from "./model.js";
export type { GpuTensor } from "./types/tensor.js";
export { OnnxModel } from "./onnx/model.js";
// <kuma-player> is deliberately not re-exported here: it extends HTMLElement, which would
// break importing this entry under SSR/Node. Import "kuma-bart/player" to register it.
export { KumaManifestError, KumaShapeError, KumaUnsupportedOpError } from "./errors.js";
// How many run()/runRaw()/runToGpu() calls a caller can usefully keep in flight with the
// default memory options (prefer `model.maxInFlight`, which reflects the loaded model's
// real pool depth): past this, runGraph's acquireGenerationSlot blocks on GPU completion.
export { BUFFER_POOL_DEPTH } from "./engine/context.js";
export { releaseOutputs } from "./engine/scheduler.js";
export type { KumaManifest, IOSpec, WeightEntry, GraphNode, NodeRef, ArgValue, PlaybackMeta } from "./types/manifest.js";
export type { GoldenData, GoldenBranch, GoldenNodeStats, GoldenTensorStats } from "./types/golden.js";
export type { VerifyReport, BranchVerifyReport, NodeDiff, FieldDiff } from "./engine/verify.js";
export type { ProfileReport, OpTiming, TargetTiming } from "./engine/profile.js";
export type { RunGraphOutput } from "./engine/scheduler.js";
