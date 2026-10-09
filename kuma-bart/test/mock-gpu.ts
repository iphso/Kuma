/** A minimal fake GPUDevice for exercising the scheduler's structural logic (node
 * resolution, dispatch ordering, buffer ownership, error paths) without a real GPU. It
 * doesn't compute anything — readback buffers are zero-filled — so it can't verify
 * numeric correctness; that needs a real browser (`model.verify()` in the demo).
 *
 * Records every dispatch's workgroup count (`dispatches`) and the label of the pipeline
 * it ran (`kernels` -- pipelines are labeled with their kernel filename, e.g.
 * "relu.wgsl"), plus every buffer created and destroyed, so tests can assert dispatch
 * sequences by kernel name and measure leaks. */
export function createMockDevice(options: { features?: string[] } = {}) {
  const dispatches: number[] = [];
  const kernels: string[] = [];
  /** Every `clearBuffer` recorded, in order (the scheduler's zero-fill of allocate-only nodes). */
  const clears: object[] = [];
  /** Every `copyBufferToBuffer` recorded: [source, sourceOffset, destination, destinationOffset, size]. */
  const copies: Array<[object, number, object, number, number]> = [];
  let submits = 0;
  let deviceDestroyed = false;
  const destroyedBuffers = new Set<object>();
  // A buffer in `createdBuffers` but not `destroyedBuffers` is still live.
  const createdBuffers = new Set<{ usage: number }>();
  let currentPipeline = "";

  const pass = {
    setPipeline: (pipeline: { label?: string }) => {
      currentPipeline = pipeline.label ?? "";
    },
    setBindGroup: () => {},
    dispatchWorkgroups: (count: number) => {
      dispatches.push(count);
      kernels.push(currentPipeline);
    },
    end: () => {},
  };

  function fakeBuffer(size: number, usage = 0) {
    const buffer = {
      size,
      usage,
      mapAsync: async () => {},
      getMappedRange: (_offset = 0, length: number = size) => new ArrayBuffer(length),
      unmap: () => {},
      destroy: () => {
        destroyedBuffers.add(buffer);
      },
    };
    createdBuffers.add(buffer);
    return buffer;
  }

  const device = {
    features: new Set(options.features ?? []),
    destroy: () => {
      deviceDestroyed = true;
    },
    createBuffer: ({ size, usage = 0 }: { size: number; usage?: number }) => fakeBuffer(size, usage),
    createShaderModule: () => ({}),
    createComputePipeline: ({ label }: { label?: string } = {}) => ({ label, getBindGroupLayout: () => ({}) }),
    createQuerySet: () => ({ destroy: () => {} }),
    // Binding a buffer that's already been .destroy()'d is exactly the real WebGPU
    // "[Buffer] used in submit while destroyed" failure (a use-after-destroy the
    // early-free liveness pass must never produce) -- surface it as a hard error.
    createBindGroup: ({ entries }: { entries?: Array<{ resource?: { buffer?: object } }> } = {}) => {
      for (const e of entries ?? []) {
        const buf = e?.resource?.buffer;
        if (buf && destroyedBuffers.has(buf)) {
          throw new Error("mock-gpu: bind group references a destroyed buffer (use-after-destroy)");
        }
      }
      return {};
    },
    createCommandEncoder: () => ({
      beginComputePass: () => pass,
      copyBufferToBuffer: (src: object, srcOffset: number, dst: object, dstOffset: number, size: number) => {
        copies.push([src, srcOffset, dst, dstOffset, size]);
      },
      clearBuffer: (buffer: object) => {
        clears.push(buffer);
      },
      resolveQuerySet: () => {},
      finish: () => ({}),
    }),
    queue: {
      writeBuffer: () => {},
      submit: () => {
        submits++;
      },
      onSubmittedWorkDone: async () => {},
    },
  };

  return {
    device: device as unknown as GPUDevice,
    pass,
    dispatches,
    kernels,
    clears,
    copies,
    destroyedBuffers,
    createdBuffers,
    /** How many times queue.submit() was called so far. */
    get submits() {
      return submits;
    },
    /** Whether device.destroy() has been called. */
    get deviceDestroyed() {
      return deviceDestroyed;
    },
  };
}

export type MockDevice = ReturnType<typeof createMockDevice>;
