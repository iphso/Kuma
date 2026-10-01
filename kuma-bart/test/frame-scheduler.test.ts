import { describe, expect, it } from "vitest";
import { LatestWinsScheduler, type FrameTicket } from "../src/player/frameScheduler.js";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

describe("LatestWinsScheduler", () => {
  it("caps in-flight frames and runs only the newest pending request when one finishes", async () => {
    const started: number[] = [];
    const gates: Array<() => void> = [];
    const scheduler = new LatestWinsScheduler(
      () => 1,
      (t) => {
        started.push(t);
        const d = deferred();
        gates.push(d.resolve);
        return d.promise;
      },
    );
    scheduler.submit(0.1);
    scheduler.submit(0.2); // pending, then overwritten
    scheduler.submit(0.3);
    expect(started).toEqual([0.1]);
    expect(scheduler.capacityHits).toBe(2);
    gates[0]!();
    await Promise.resolve();
    await Promise.resolve();
    expect(started).toEqual([0.1, 0.3]);
  });

  it("lets only the newest frame render, and stales every ticket on reset()", async () => {
    const tickets: FrameTicket[] = [];
    const scheduler = new LatestWinsScheduler(() => 3, async (_t, ticket) => void tickets.push(ticket));
    scheduler.submit(0);
    scheduler.submit(1);
    expect(tickets[1]!.claimRender()).toBe(true);
    expect(tickets[0]!.claimRender()).toBe(false); // older frame finishing late
    scheduler.submit(2);
    scheduler.reset();
    expect(tickets[2]!.isCurrent()).toBe(false);
    expect(tickets[2]!.claimRender()).toBe(false);
  });
});
