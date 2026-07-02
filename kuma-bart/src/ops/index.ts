import type { OpHandler } from "../engine/context.js";
import {
  addHandler,
  mulHandler,
  subHandler,
  divHandler,
  minimumHandler,
  divModeHandler,
  reluHandler,
  geluHandler,
  sqrtHandler,
  cosHandler,
  sinHandler,
  floorHandler,
  erfHandler,
  reshapeHandler,
} from "./elementwise.js";
import { conv2dHandler, conv2dSimpleHandler } from "./conv2d.js";
import { linearHandler, findLinearWeightElisions } from "./linear.js";
import { permuteHandler, transposeHandler } from "./permute.js";
import { concatHandler } from "./concat.js";
import { stackHandler } from "./stack.js";
import { sliceHandler, sliceMultiHandler } from "./slice.js";
import { chunkHandler, getitemHandler, selectHandler } from "./chunk.js";
import { expandHandler } from "./expand.js";
import { complexHandler, realHandler, imagHandler } from "./complex.js";
import { groupNormHandler, layerNormHandler } from "./norm.js";
import { clampHandler, powScalarHandler, rsubScalarHandler } from "./scalar.js";
import { pixelShuffleHandler } from "./pixel_shuffle.js";
import { sumHandler, linalgVectorNormHandler } from "./reduce.js";
import { gatherHandler } from "./gather.js";
import { gatherDimHandler } from "./gather_dim.js";
import { einsumHandler } from "./einsum.js";
import { fftIrfft2Handler } from "./fft.js";
import { passthroughHandler, zerosLikeHandler } from "./passthrough.js";

/** target string -> handler, covering the natural aten aliases documented in each
 * kernel's WGSL header comment. Anything not listed here has no bundled kernel and
 * must fail loudly (see scheduler.ts) rather than be guessed at. */
export const opRegistry: ReadonlyMap<string, OpHandler> = new Map<string, OpHandler>([
  // elementwise
  ["aten.add.Tensor", addHandler],
  ["aten.add_.Tensor", addHandler],
  ["aten.mul.Tensor", mulHandler],
  ["aten.sub.Tensor", subHandler],
  ["aten.div.Tensor", divHandler],
  ["aten.div.Tensor_mode", divModeHandler],
  ["aten.minimum.default", minimumHandler],
  ["aten.relu.default", reluHandler],
  ["aten.gelu.default", geluHandler],
  ["aten.sqrt.default", sqrtHandler],
  ["aten.cos.default", cosHandler],
  ["aten.sin.default", sinHandler],
  ["aten.floor.default", floorHandler],
  ["aten.erf.default", erfHandler],
  ["aten.clamp.default", clampHandler],
  ["aten.pow.Tensor_Scalar", powScalarHandler],
  ["aten.rsub.Scalar", rsubScalarHandler],
  // shape / layout
  ["aten.view.default", reshapeHandler],
  ["aten.reshape.default", reshapeHandler],
  ["aten.permute.default", permuteHandler],
  ["aten.t.default", transposeHandler],
  ["aten.transpose.int", transposeHandler],
  ["aten.cat.default", concatHandler],
  ["aten.stack.default", stackHandler],
  ["aten.slice.Tensor", sliceHandler],
  ["aten.slice_multi.default", sliceMultiHandler],
  ["aten.select.int", selectHandler],
  ["aten.chunk.default", chunkHandler],
  ["getitem", getitemHandler],
  ["aten.expand.default", expandHandler],
  // matmul / conv
  ["aten.convolution.default", conv2dHandler],
  ["aten.conv2d.default", conv2dSimpleHandler],
  ["aten.addmm.default", linearHandler],
  ["aten.matmul.default", linearHandler],
  ["aten.mm.default", linearHandler],
  ["aten.linear.default", linearHandler],
  // normalization
  ["aten.group_norm.default", groupNormHandler],
  ["aten.layer_norm.default", layerNormHandler],
  // reductions
  ["aten.sum.dim_IntList", sumHandler],
  ["aten.linalg_vector_norm.default", linalgVectorNormHandler],
  // upsampling / shuffling
  ["aten.pixel_shuffle.default", pixelShuffleHandler],
  // gather / indexing
  ["aten.index.Tensor", gatherHandler],
  ["aten.gather.default", gatherDimHandler],
  // complex / FFT / Tucker (see ops/complex.ts, ops/einsum.ts, ops/fft.ts)
  ["aten.complex.default", complexHandler],
  ["aten.real.default", realHandler],
  ["aten.imag.default", imagHandler],
  ["aten.einsum.default", einsumHandler],
  ["aten.fft_irfft2.default", fftIrfft2Handler],
  // free / metadata-only passthroughs (see ops/passthrough.ts)
  ["aten.alias.default", passthroughHandler],
  ["aten.contiguous.default", passthroughHandler],
  ["aten.detach.default", passthroughHandler],
  ["aten.detach_.default", passthroughHandler],
  ["aten.lift_fresh_copy.default", passthroughHandler],
  ["aten.to.device", passthroughHandler],
  ["aten.to.dtype", passthroughHandler],
  ["aten.squeeze.dim", passthroughHandler],
  ["aten.unsqueeze.default", passthroughHandler],
  ["aten.zeros_like.default", zerosLikeHandler],
]);

export { findLinearWeightElisions };
