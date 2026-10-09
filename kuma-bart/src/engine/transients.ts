import { createStorageBuffer } from "webgpu-compute-kit";
import type { ScratchAllocator } from "./context.js";

/**
 * The scratch memory of one graph encode: every intermediate a `recordGraph` call
 * allocates comes from here, and nothing is ever destroyed mid-encode. A buffer whose last
 * consumer has been encoded goes back on a free list keyed by exact byte size, so a later
 * node of the same size reuses it inside the same command buffer -- safe because passes
 * execute in order within one submission. After the whole encode has been submitted,
 * `reset()` makes every buffer idle again for the next encode that borrows this set.
 *
 * Why not the ring pool (`BufferPool`/`acquireGenerationSlot`)? That design gives one
 * slot per *node* per generation and advances generations on submit, so two encodes
 * recorded into one encoder before submission would either alias each other's buffers
 * (depth exhausted) or need a fence the API doesn't have. A set is held exclusively by
 * one encode from `recordGraph` until the caller's `release()`, so concurrent encodes
 * never share a buffer: a model keeps a free list of sets and hands each encode its own.
 *
 * Memory: a set's footprint is roughly the encode's peak live activation bytes (a free
 * list is a greedy allocator over exact sizes), retained across encodes for reuse.
 */
export class TransientSet implements ScratchAllocator {
  /** Reusable buffers by (padded) byte size. */
  private readonly idle = new Map<number, GPUBuffer[]>();
  private readonly busy = new Set<GPUBuffer>();
  private readonly all = new Set<GPUBuffer>();
  private totalBytes = 0;

  constructor(private readonly device: GPUDevice) {}

  /** `key` is ignored: reuse is by size and liveness, not by node. */
  acquire(_key: string, byteLength: number): GPUBuffer {
    const list = this.idle.get(paddedSize(byteLength));
    let buffer = list?.pop();
    if (!buffer) {
      buffer = createStorageBuffer(this.device, byteLength);
      this.all.add(buffer);
      this.totalBytes += buffer.size;
    }
    this.busy.add(buffer);
    return buffer;
  }

  /** Whether this set allocated `buffer` (weights, inputs and cached constants never are). */
  owns(buffer: GPUBuffer): boolean {
    return this.all.has(buffer);
  }

  /** Marks a buffer of this set reusable by later nodes of the same encode. A no-op for
   * buffers it doesn't own or that are already idle. */
  free(buffer: GPUBuffer): void {
    if (!this.busy.delete(buffer)) return;
    this.pushIdle(buffer);
  }

  /** Every buffer idle again -- call once the encode that used this set has been
   * submitted (destroy/reuse after submit is queue-ordered, no completion wait needed). */
  reset(): void {
    for (const buffer of this.busy) this.pushIdle(buffer);
    this.busy.clear();
  }

  /** Bytes currently allocated in this set, idle or busy. */
  get bytes(): number {
    return this.totalBytes;
  }

  /** Number of buffers currently allocated in this set. */
  get size(): number {
    return this.all.size;
  }

  /** Destroys every buffer. The set must not be in use by an unsubmitted encode. */
  destroy(): void {
    for (const buffer of this.all) buffer.destroy();
    this.all.clear();
    this.idle.clear();
    this.busy.clear();
    this.totalBytes = 0;
  }

  private pushIdle(buffer: GPUBuffer): void {
    let list = this.idle.get(buffer.size);
    if (!list) {
      list = [];
      this.idle.set(buffer.size, list);
    }
    list.push(buffer);
  }
}

/** createStorageBuffer's own rounding, so a request and the buffer it got share a key. */
function paddedSize(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}
