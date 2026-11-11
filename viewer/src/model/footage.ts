import type { Camera } from "./types";

/**
 * The camera matrix for a frame, written into `out` (nine values, row-major). Between two kept
 * matrices it is blended; next to a frame with no camera the nearer known one is used. False
 * where the camera is not known.
 */
export function cameraAt(camera: Camera, frame: number, out: Float64Array): boolean {
  if (camera.count === 0) return false;
  const at = Math.min(camera.count - 1, Math.max(0, frame / camera.every));
  const i = Math.floor(at);
  const j = Math.min(camera.count - 1, i + 1);
  const u = at - i;
  const a = camera.matrix;
  const p = i * 9;
  const q = j * 9;
  const hasP = a[p] === a[p];
  const hasQ = a[q] === a[q];
  let from = -1;
  if (u < 1e-6 || i === j) from = hasP ? p : -1;
  else if (hasP && hasQ) {
    for (let k = 0; k < 9; k++) out[k] = a[p + k] + (a[q + k] - a[p + k]) * u;
    return true;
  } else if (hasP && u <= 0.5) from = p;
  else if (hasQ && u > 0.5) from = q;
  if (from < 0) return false;
  for (let k = 0; k < 9; k++) out[k] = a[from + k];
  return true;
}

export interface PicturePoint {
  u: number;
  v: number;
}

/**
 * Where a pitch position (metres from the centre spot) is in the picture, in pixels. False when
 * the position is behind the camera, where the matrix has no meaning.
 */
export function toPicture(m: Float64Array, x: number, y: number, out: PicturePoint): boolean {
  const w = m[6] * x + m[7] * y + m[8];
  if (!(w > 1e-9)) return false;
  out.u = (m[0] * x + m[1] * y + m[2]) / w;
  out.v = (m[3] * x + m[4] * y + m[5]) / w;
  return out.u === out.u && out.v === out.v;
}

const here: PicturePoint = { u: 0, v: 0 };
const along: PicturePoint = { u: 0, v: 0 };
const across: PicturePoint = { u: 0, v: 0 };

/** How many pixels a metre on the ground takes up around a pitch position: the larger of along and across. 0 when it cannot be said. */
export function pixelsPerMetre(m: Float64Array, x: number, y: number): number {
  if (!toPicture(m, x, y, here) || !toPicture(m, x + 1, y, along) || !toPicture(m, x, y + 1, across)) return 0;
  const a = Math.hypot(along.u - here.u, along.v - here.v);
  const b = Math.hypot(across.u - here.u, across.v - here.v);
  return Math.max(a, b);
}

/** A part of the picture, in its pixels. */
export interface Crop {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * The part of a picture `span` pixels wide with the proportions of a box, centred as near to
 * (cu, cv) as the picture's edges allow. Never larger than the picture.
 */
export function cropAround(cu: number, cv: number, span: number, boxW: number, boxH: number, picW: number, picH: number): Crop {
  let width = Math.min(picW, Math.max(16, span));
  let height = (width * boxH) / boxW;
  if (height > picH) {
    height = picH;
    width = (height * boxW) / boxH;
  }
  const x = Math.min(picW - width, Math.max(0, cu - width / 2));
  const y = Math.min(picH - height, Math.max(0, cv - height / 2));
  return { x, y, width, height };
}

/**
 * The smallest part of a picture with the proportions of a box that holds every point (u0, v0,
 * u1, v1 …), with room around them: `side` pixels left, right and below, `above` pixels over
 * them (a player stands up from the point where his feet are). Null when no point is inside
 * the picture.
 */
export function cropAroundPoints(points: ArrayLike<number>, side: number, above: number, boxW: number, boxH: number, picW: number, picH: number): Crop | null {
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i + 1 < points.length; i += 2) {
    const u = points[i];
    const v = points[i + 1];
    if (u !== u || v !== v) continue;
    if (u < -picW * 0.25 || u > picW * 1.25 || v < -picH * 0.25 || v > picH * 1.25) continue; // far outside: tracking noise or off screen
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  if (minU > maxU) return null;
  const left = minU - side;
  const right = maxU + side;
  const top = minV - above;
  const bottom = maxV + side;
  // widen the shorter way so the box's proportions hold
  const span = Math.max(right - left, ((bottom - top) * boxW) / boxH);
  return cropAround((left + right) / 2, (top + bottom) / 2, span, boxW, boxH, picW, picH);
}
