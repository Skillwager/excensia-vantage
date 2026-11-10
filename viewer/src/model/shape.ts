import { emptyMoment, sampleMoment, type Moment } from "./moment";
import type { Run, Tracking } from "./types";

/** The shape of one team at one instant, from its outfield players. */
export interface TeamShape {
  /** Outfield players seen. */
  count: number;
  /** Centre of the outfield players. */
  cx: number;
  cy: number;
  /** Spread across the pitch, metres. */
  width: number;
  /** Spread along the pitch, front to back, metres. */
  depth: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  /** x of the defensive line: the average of the four deepest outfield players. */
  lineX: number;
  /** Distance of that line from the team's own goal line, metres. */
  lineHeight: number;
  /** Distance of the team's centre from its own goal line, metres. */
  centreHeight: number;
  /** Convex hull of the outfield players, x0, y0, x1, y1 … */
  hull: number[];
}

interface ShapeIndex {
  keepers: [number, number];
  /** +1 when Home attack towards +x in that minute of the match, -1 the other way, 0 when unknown. */
  homeDir: Int8Array;
  bucketFrames: number;
}

const cache = new WeakMap<Tracking, ShapeIndex>();

/**
 * Works out, from the tracking alone, which slot is each team's goalkeeper (the player who lives by a goal line)
 * and which way each team is attacking minute by minute (a team's players sit on its own side of the other's).
 */
export function shapeIndex(run: Pick<Run, "tracking" | "fps" | "frames" | "pitch">): ShapeIndex | null {
  const tracking = run.tracking;
  if (!tracking) return null;
  const hit = cache.get(tracking);
  if (hit) return hit;

  const stride = Math.max(1, Math.round((run.fps * 2) / tracking.every)); // one look every two seconds
  const keeperOf = (data: Float32Array, slots: number): number => {
    let best = -1;
    let bestMean = 0;
    for (let p = 0; p < slots; p++) {
      let sum = 0;
      let n = 0;
      for (let i = 0; i < tracking.count; i += stride) {
        const x = data[(i * slots + p) * 2];
        if (x === x) {
          sum += Math.abs(x);
          n++;
        }
      }
      const mean = n > 0 ? sum / n : 0;
      if (n > 4 && mean > bestMean) {
        bestMean = mean;
        best = p;
      }
    }
    return bestMean > run.pitch.length * 0.3 ? best : -1;
  };
  const keepers: [number, number] = [keeperOf(tracking.home, tracking.homeSlots), keeperOf(tracking.away, tracking.awaySlots)];

  const bucketFrames = Math.max(1, Math.round(run.fps * 60));
  const buckets = Math.max(1, Math.ceil(run.frames / bucketFrames));
  const diff = new Float32Array(buckets);
  const meanX = (data: Float32Array, slots: number, i: number, skip: number): number => {
    let sum = 0;
    let n = 0;
    for (let p = 0; p < slots; p++) {
      if (p === skip) continue;
      const x = data[(i * slots + p) * 2];
      if (x === x) {
        sum += x;
        n++;
      }
    }
    return n ? sum / n : NaN;
  };
  for (let i = 0; i < tracking.count; i += stride) {
    const b = Math.min(buckets - 1, Math.floor((i * tracking.every) / bucketFrames));
    const h = meanX(tracking.home, tracking.homeSlots, i, keepers[0]);
    const a = meanX(tracking.away, tracking.awaySlots, i, keepers[1]);
    if (h === h && a === a) diff[b] += h - a;
  }
  const homeDir = new Int8Array(buckets);
  for (let b = 0; b < buckets; b++) {
    const around = diff[b] * 2 + (b > 0 ? diff[b - 1] : 0) + (b + 1 < buckets ? diff[b + 1] : 0);
    // Home sitting on the -x side of Away means Home defend that end and attack towards +x.
    homeDir[b] = around < 0 ? 1 : around > 0 ? -1 : 0;
  }
  // a minute with nothing to go on takes its neighbour's answer
  for (let b = 1; b < buckets; b++) if (homeDir[b] === 0) homeDir[b] = homeDir[b - 1];
  for (let b = buckets - 2; b >= 0; b--) if (homeDir[b] === 0) homeDir[b] = homeDir[b + 1];

  const index: ShapeIndex = { keepers, homeDir, bucketFrames };
  cache.set(tracking, index);
  return index;
}

/** +1 when the team attacks towards +x at this frame, -1 the other way, 0 when unknown. */
export function attackDir(index: ShapeIndex | null, team: 0 | 1, frame: number): 1 | -1 | 0 {
  if (!index) return 0;
  const b = Math.min(index.homeDir.length - 1, Math.max(0, Math.floor(frame / index.bucketFrames)));
  const home = index.homeDir[b];
  return (team === 0 ? home : -home) as 1 | -1 | 0;
}

/** Convex hull (monotone chain) of points given as x0, y0, x1, y1 … */
export function convexHull(points: number[]): number[] {
  const n = points.length / 2;
  if (n < 3) return points.slice();
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => points[a * 2] - points[b * 2] || points[a * 2 + 1] - points[b * 2 + 1]);
  const cross = (o: number, a: number, b: number) =>
    (points[a * 2] - points[o * 2]) * (points[b * 2 + 1] - points[o * 2 + 1]) - (points[a * 2 + 1] - points[o * 2 + 1]) * (points[b * 2] - points[o * 2]);
  const lower: number[] = [];
  for (const i of order) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], i) <= 0) lower.pop();
    lower.push(i);
  }
  const upper: number[] = [];
  for (let k = order.length - 1; k >= 0; k--) {
    const i = order[k];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], i) <= 0) upper.pop();
    upper.push(i);
  }
  lower.pop();
  upper.pop();
  const out: number[] = [];
  for (const i of [...lower, ...upper]) out.push(points[i * 2], points[i * 2 + 1]);
  return out;
}

/** The shape of one team in a moment. `dir` is the way it attacks (+1 towards +x); `keeper` is left out. */
export function teamShape(positions: Float32Array, keeper: number, dir: 1 | -1 | 0, pitchLength: number): TeamShape | null {
  const xs: number[] = [];
  const pts: number[] = [];
  let sx = 0;
  let sy = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let p = 0; p < positions.length / 2; p++) {
    if (p === keeper) continue;
    const x = positions[p * 2];
    const y = positions[p * 2 + 1];
    if (x !== x) continue;
    xs.push(x);
    pts.push(x, y);
    sx += x;
    sy += y;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const count = xs.length;
  if (count < 3) return null;
  const facing = dir === 0 ? 1 : dir;
  // deepest first: nearest the team's own goal
  xs.sort((a, b) => (a - b) * facing);
  const back = Math.min(4, count);
  let lineX = 0;
  for (let i = 0; i < back; i++) lineX += xs[i];
  lineX /= back;
  const half = pitchLength / 2;
  const height = (x: number) => (facing > 0 ? x + half : half - x);
  return {
    count,
    cx: sx / count,
    cy: sy / count,
    width: maxY - minY,
    depth: maxX - minX,
    minX,
    maxX,
    minY,
    maxY,
    lineX,
    lineHeight: height(lineX),
    centreHeight: height(sx / count),
    hull: convexHull(pts),
  };
}

export interface ShapePair {
  home: TeamShape | null;
  away: TeamShape | null;
}

/** Both teams' shapes in a moment that has already been read from tracking. */
export function shapesAt(run: Pick<Run, "tracking" | "fps" | "frames" | "pitch">, moment: Moment, frame: number): ShapePair {
  const index = shapeIndex(run);
  if (!index || !moment.has) return { home: null, away: null };
  return {
    home: teamShape(moment.home, index.keepers[0], attackDir(index, 0, frame), run.pitch.length),
    away: teamShape(moment.away, index.keepers[1], attackDir(index, 1, frame), run.pitch.length),
  };
}

export type ShapeMetric = "width" | "depth" | "line";

export interface ShapeSeries {
  /** Seconds of each point. */
  t: Float32Array;
  home: Float32Array;
  away: Float32Array;
}

/** One measure of both teams' shape, second by second, across a window of the match. NaN where there is no tracking. */
export function shapeSeries(
  run: Pick<Run, "tracking" | "fps" | "frames" | "pitch">,
  metric: ShapeMetric,
  fromFrame: number,
  toFrame: number,
  stepSeconds = 1,
): ShapeSeries {
  const step = Math.max(1, Math.round(stepSeconds * run.fps));
  const first = Math.ceil(fromFrame / step) * step;
  const n = Math.max(0, Math.floor((toFrame - first) / step) + 1);
  const t = new Float32Array(n);
  const home = new Float32Array(n).fill(NaN);
  const away = new Float32Array(n).fill(NaN);
  const moment = emptyMoment(run.tracking);
  const read = (s: TeamShape | null) => (s ? (metric === "width" ? s.width : metric === "depth" ? s.depth : s.lineHeight) : NaN);
  for (let k = 0; k < n; k++) {
    const frame = first + k * step;
    t[k] = frame / run.fps;
    if (frame < 0 || frame > run.frames - 1) continue;
    sampleMoment(run.tracking, frame, run.fps, moment);
    const pair = shapesAt(run, moment, frame);
    home[k] = read(pair.home);
    away[k] = read(pair.away);
  }
  return { t, home, away };
}
