/** Handed to each frame job so it can tell, once its async work resolves, whether its
 * result still matters. */
export interface FrameTicket {
  /** False once `reset()` has run since this frame was submitted (e.g. a new model was
   * loaded); a stale frame must not touch the player's state. */
  isCurrent(): boolean;
  /** True (and records this frame as the newest rendered) iff the frame is current and
   * newer than every frame rendered so far -- an older frame finishing late must not
   * overwrite a newer one. */
  claimRender(): boolean;
}

/** Latest-wins frame submission under an in-flight cap: while `maxInFlight()` frames are
 * running, further requests overwrite a single pending slot instead of queueing, and the
 * newest pending request starts as soon as a running frame finishes. */
export class LatestWinsScheduler {
  private inFlight = 0;
  private pending: number | undefined;
  private generation = 0;
  private nextSequence = 0;
  private latestRendered = -1;
  /** Requests since the last `resetStats()`, and how many hit the in-flight cap. */
  requested = 0;
  capacityHits = 0;

  constructor(
    private readonly maxInFlight: () => number,
    private readonly job: (t: number, ticket: FrameTicket) => Promise<void>,
  ) {}

  submit(t: number): void {
    this.requested++;
    if (this.inFlight >= this.maxInFlight()) {
      this.capacityHits++;
      this.pending = t;
      return;
    }
    this.pending = undefined;
    this.inFlight++;
    const generation = this.generation;
    const sequence = this.nextSequence++;
    const ticket: FrameTicket = {
      isCurrent: () => generation === this.generation,
      claimRender: () => {
        if (generation !== this.generation || sequence <= this.latestRendered) return false;
        this.latestRendered = sequence;
        return true;
      },
    };
    void this.job(t, ticket).finally(() => {
      // A reset() since submission already zeroed the counters for the new generation.
      if (generation !== this.generation) return;
      this.inFlight--;
      if (this.pending !== undefined) this.submit(this.pending);
    });
  }

  /** Abandons every running and pending frame (their tickets go stale). */
  reset(): void {
    this.generation++;
    this.inFlight = 0;
    this.pending = undefined;
  }

  resetStats(): void {
    this.requested = 0;
    this.capacityHits = 0;
  }
}
