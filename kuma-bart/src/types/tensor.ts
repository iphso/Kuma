/**
 * A GPU-resident tensor, as exchanged with `KumaModel.encode()` -- both what a caller
 * hands in as inputs and what comes back as outputs.
 *
 * Layout contract (the same one the `.iph` compiler packs weights and activations in):
 * contiguous, row-major (C order: the last dimension varies fastest), little-endian
 * float32, with no padding between elements. A 4-D activation is NCHW --
 * `shape = [batch, channels, height, width]`, element `(n, c, y, x)` at float index
 * `((n * C + c) * H + y) * W + x`. Element count is the product of `shape`; the tensor
 * occupies `byteOffset .. byteOffset + 4 * product(shape)` of `buffer`.
 *
 * `buffer` must have been created on the model's own device (`model.gpuDevice`):
 * WebGPU resources are device-scoped, and binding a buffer from another device is a
 * validation error raised asynchronously by WebGPU (an `uncapturederror` event / rejected
 * `popErrorScope`), not something kuma-bart can detect up front -- the API exposes no
 * way to ask a buffer which device it belongs to.
 */
export interface GpuTensor {
  buffer: GPUBuffer;
  shape: readonly number[];
  /** Only float32 for now (see the README's "f16 and fusion" notes for what more would take). */
  dtype: "f32";
  /** Byte offset of element 0 within `buffer`; a multiple of 4. Default 0. Inputs at a
   * non-zero offset are copied into the model's scratch (one `copyBufferToBuffer`, so
   * `buffer` needs COPY_SRC usage) because kuma's kernels bind whole buffers; inputs at
   * offset 0 are bound directly and need STORAGE usage. */
  byteOffset?: number;
}
