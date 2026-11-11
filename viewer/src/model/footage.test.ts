import { describe, expect, it } from "vitest";
import { cameraAt, cropAround, cropAroundPoints, pixelsPerMetre, toPicture } from "./footage";
import type { Camera } from "./types";

/** A camera straight above the centre spot: 10 pixels a metre, the centre spot at (960, 540). */
const overhead = [10, 0, 960, 0, 10, 540, 0, 0, 1];

function camera(rows: (number[] | null)[], every = 1): Camera {
  const matrix = new Float64Array(rows.length * 9).fill(NaN);
  rows.forEach((row, i) => row && matrix.set(row, i * 9));
  return { width: 1920, height: 1080, every, count: rows.length, matrix };
}

describe("the camera of a frame", () => {
  it("is the kept matrix on a kept frame, and nothing where the camera is not known", () => {
    const cam = camera([overhead, null, overhead.map((v) => v * 2)]);
    const out = new Float64Array(9);
    expect(cameraAt(cam, 0, out)).toBe(true);
    expect(Array.from(out)).toEqual(overhead);
    expect(cameraAt(cam, 1, out)).toBe(false);
    expect(cameraAt(cam, 2, out)).toBe(true);
    expect(out[0]).toBe(20);
    expect(cameraAt(cam, 99, out)).toBe(true); // past the end: the last one
    expect(out[0]).toBe(20);
  });

  it("is blended between two kept matrices and borrowed from the nearer one beside a gap", () => {
    const moved = [10, 0, 1060, 0, 10, 540, 0, 0, 1];
    const cam = camera([overhead, moved, null], 5);
    const out = new Float64Array(9);
    expect(cameraAt(cam, 2.5, out)).toBe(true);
    expect(out[2]).toBeCloseTo(1010);
    expect(cameraAt(cam, 6, out)).toBe(true); // a fifth of the way to the gap
    expect(out[2]).toBe(1060);
    expect(cameraAt(cam, 9, out)).toBe(false); // nearer the gap than the known one
  });
});

describe("a pitch position in the picture", () => {
  it("goes through the matrix, perspective included", () => {
    const point = { u: 0, v: 0 };
    expect(toPicture(Float64Array.from(overhead), 5, -3, point)).toBe(true);
    expect(point).toEqual({ u: 1010, v: 510 });
    // the far touchline is half the size of the near one
    const tilted = Float64Array.from([10, 0, 960, 0, 10, 540, 0, -0.02, 1]);
    expect(toPicture(tilted, 10, 25, point)).toBe(true);
    expect(point.u).toBeCloseTo((100 + 960) / 0.5);
    expect(toPicture(tilted, 0, 60, point)).toBe(false); // behind the camera
  });

  it("knows how large a metre looks", () => {
    expect(pixelsPerMetre(Float64Array.from(overhead), 3, 4)).toBeCloseTo(10);
    expect(pixelsPerMetre(Float64Array.from([10, 0, 960, 0, 10, 540, 0, -0.02, 1]), 0, 60)).toBe(0);
  });
});

describe("the part of the picture a tile shows", () => {
  it("keeps the box's proportions and stays inside the picture", () => {
    expect(cropAround(960, 540, 400, 200, 100, 1920, 1080)).toEqual({ x: 760, y: 440, width: 400, height: 200 });
    expect(cropAround(10, 1075, 400, 200, 100, 1920, 1080)).toEqual({ x: 0, y: 880, width: 400, height: 200 });
    // wider than the picture: the whole width, or as much as its height allows
    expect(cropAround(960, 540, 5000, 200, 100, 1920, 1080)).toEqual({ x: 0, y: 60, width: 1920, height: 960 });
    expect(cropAround(960, 540, 5000, 100, 100, 1920, 1080)).toEqual({ x: 420, y: 0, width: 1080, height: 1080 });
  });

  it("holds every player with room over their heads, and ignores points far off the picture", () => {
    const crop = cropAroundPoints([500, 600, 900, 700, NaN, NaN, 9000, 9000], 20, 60, 200, 100, 1920, 1080);
    expect(crop).not.toBeNull();
    expect(crop!.x).toBeLessThanOrEqual(480);
    expect(crop!.x + crop!.width).toBeGreaterThanOrEqual(920);
    expect(crop!.y).toBeLessThanOrEqual(540);
    expect(crop!.y + crop!.height).toBeGreaterThanOrEqual(720);
    expect(crop!.width / crop!.height).toBeCloseTo(2);
    expect(cropAroundPoints([NaN, NaN, 9000, 9000], 20, 60, 200, 100, 1920, 1080)).toBeNull();
  });
});
