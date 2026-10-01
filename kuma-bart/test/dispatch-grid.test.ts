import { describe, expect, it } from "vitest";
import { computeDispatchGrid, MAX_WORKGROUPS_PER_DIMENSION } from "webgpu-compute-kit";

describe("computeDispatchGrid", () => {
  it("stays 1D (y=1) when under the per-dimension limit", () => {
    expect(computeDispatchGrid(1)).toEqual({ x: 1, y: 1 });
    expect(computeDispatchGrid(43200)).toEqual({ x: 43200, y: 1 });
    expect(computeDispatchGrid(MAX_WORKGROUPS_PER_DIMENSION)).toEqual({ x: MAX_WORKGROUPS_PER_DIMENSION, y: 1 });
  });

  it("splits into a 2D grid just above the limit, maxing out x first", () => {
    const grid = computeDispatchGrid(MAX_WORKGROUPS_PER_DIMENSION + 1);
    expect(grid).not.toBeNull();
    expect(grid!.x).toBe(MAX_WORKGROUPS_PER_DIMENSION);
    expect(grid!.y).toBe(2);
    // x*y must cover every requested workgroup.
    expect(grid!.x * grid!.y).toBeGreaterThanOrEqual(MAX_WORKGROUPS_PER_DIMENSION + 1);
  });

  it("covers the real stack/concat case that motivated this (≈86,400 workgroups)", () => {
    const grid = computeDispatchGrid(86_400);
    expect(grid).not.toBeNull();
    expect(grid!.x * grid!.y).toBeGreaterThanOrEqual(86_400);
    expect(grid!.x).toBeLessThanOrEqual(MAX_WORKGROUPS_PER_DIMENSION);
    expect(grid!.y).toBeLessThanOrEqual(MAX_WORKGROUPS_PER_DIMENSION);
  });

  it("returns null once even a full 65535x65535 grid can't cover the request", () => {
    const tooMany = MAX_WORKGROUPS_PER_DIMENSION * MAX_WORKGROUPS_PER_DIMENSION + 1;
    expect(computeDispatchGrid(tooMany)).toBeNull();
  });

  it("always produces a grid whose capacity covers the request, for a range of sizes", () => {
    for (const n of [1, 64, 65535, 65536, 100_000, 5_000_000, 1_000_000_000]) {
      const grid = computeDispatchGrid(n);
      expect(grid).not.toBeNull();
      expect(grid!.x * grid!.y).toBeGreaterThanOrEqual(n);
      expect(grid!.x).toBeLessThanOrEqual(MAX_WORKGROUPS_PER_DIMENSION);
      expect(grid!.y).toBeLessThanOrEqual(MAX_WORKGROUPS_PER_DIMENSION);
    }
  });
});
