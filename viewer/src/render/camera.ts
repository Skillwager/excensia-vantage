import { ballIndex } from "../model/moment";
import type { Tracking } from "../model/types";

/**
 * A pinhole camera for the drawn match view. World: x along the pitch, y across it (the camera stands on the
 * -y side), z up; metres. Screen: CSS pixels, y down.
 */
export class Projector {
  /** Results of the last project() call. */
  x = 0;
  y = 0;
  /** Pixels per metre at that depth. */
  s = 0;
  /** Distance along the view axis, metres. */
  depth = 0;

  private px = 0;
  private py = 0;
  private pz = 0;
  private rx = 1;
  private ry = 0;
  private rz = 0;
  private ux = 0;
  private uy = 0;
  private uz = 1;
  private fx = 0;
  private fy = 1;
  private fz = 0;
  private focal = 1;
  private cx = 0;
  private cy = 0;

  /** Places the camera at p, looking at l, with a vertical field of view in radians, for a picture w × h. */
  setup(px: number, py: number, pz: number, lx: number, ly: number, lz: number, vfov: number, w: number, h: number): void {
    this.px = px;
    this.py = py;
    this.pz = pz;
    let fx = lx - px;
    let fy = ly - py;
    let fz = lz - pz;
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl;
    fy /= fl;
    fz /= fl;
    // right = forward × up(0,0,1)
    let rx = fy;
    let ry = -fx;
    const rl = Math.hypot(rx, ry) || 1;
    rx /= rl;
    ry /= rl;
    this.fx = fx;
    this.fy = fy;
    this.fz = fz;
    this.rx = rx;
    this.ry = ry;
    this.rz = 0;
    // up = right × forward
    this.ux = ry * fz;
    this.uy = -rx * fz;
    this.uz = rx * fy - ry * fx;
    this.focal = h / 2 / Math.tan(vfov / 2);
    this.cx = w / 2;
    this.cy = h / 2;
  }

  /** Projects a world point. Returns false when it is behind the camera. */
  project(x: number, y: number, z: number): boolean {
    const dx = x - this.px;
    const dy = y - this.py;
    const dz = z - this.pz;
    const depth = dx * this.fx + dy * this.fy + dz * this.fz;
    if (depth < 0.5) return false;
    const s = this.focal / depth;
    this.x = this.cx + (dx * this.rx + dy * this.ry + dz * this.rz) * s;
    this.y = this.cy - (dx * this.ux + dy * this.uy + dz * this.uz) * s;
    this.s = s;
    this.depth = depth;
    return true;
  }
}

const trackCache = new WeakMap<Tracking, Float32Array>();

/**
 * Where the camera looks, sample by sample (x then y for each sample): it follows the ball (or, when the ball
 * is out of sight, the line between where it was last seen and where it is next seen) and the body of the
 * players, eased forwards and backwards in time so it moves like an operator who can see the play coming.
 */
export function cameraTrack(tracking: Tracking, fps: number): Float32Array {
  const hit = trackCache.get(tracking);
  if (hit) return hit;
  const n = tracking.count;
  const out = new Float32Array(n * 2);
  const index = ballIndex(tracking);
  const { ball, home, away, homeSlots, awaySlots } = tracking;
  for (let i = 0; i < n; i++) {
    let sumX = 0;
    let sumY = 0;
    let seen = 0;
    for (let p = 0; p < homeSlots; p++) {
      const x = home[(i * homeSlots + p) * 2];
      if (x === x) {
        sumX += x;
        sumY += home[(i * homeSlots + p) * 2 + 1];
        seen++;
      }
    }
    for (let p = 0; p < awaySlots; p++) {
      const x = away[(i * awaySlots + p) * 2];
      if (x === x) {
        sumX += x;
        sumY += away[(i * awaySlots + p) * 2 + 1];
        seen++;
      }
    }
    const crowdX = seen ? sumX / seen : 0;
    const crowdY = seen ? sumY / seen : 0;
    const before = index.prev[i];
    const after = index.next[i];
    let bx: number;
    let by: number;
    if (before < 0 && after < 0) {
      bx = crowdX;
      by = crowdY;
    } else if (before < 0 || after < 0 || after === before) {
      const k = before < 0 ? after : before;
      bx = ball[k * 2];
      by = ball[k * 2 + 1];
    } else {
      const t = (i - before) / (after - before);
      bx = ball[before * 2] + (ball[after * 2] - ball[before * 2]) * t;
      by = ball[before * 2 + 1] + (ball[after * 2 + 1] - ball[before * 2 + 1]) * t;
    }
    out[i * 2] = 0.62 * bx + 0.38 * crowdX;
    out[i * 2 + 1] = 0.45 * by + 0.55 * crowdY;
  }
  const dt = tracking.every / fps;
  const kx = 1 - Math.exp(-dt / 0.9);
  const ky = 1 - Math.exp(-dt / 1.6);
  for (let i = 1; i < n; i++) {
    out[i * 2] = out[i * 2 - 2] + (out[i * 2] - out[i * 2 - 2]) * kx;
    out[i * 2 + 1] = out[i * 2 - 1] + (out[i * 2 + 1] - out[i * 2 - 1]) * ky;
  }
  for (let i = n - 2; i >= 0; i--) {
    out[i * 2] = out[i * 2 + 2] + (out[i * 2] - out[i * 2 + 2]) * kx;
    out[i * 2 + 1] = out[i * 2 + 3] + (out[i * 2 + 1] - out[i * 2 + 3]) * ky;
  }
  for (let i = 0; i < n; i++) out[i * 2] = Math.max(-33, Math.min(33, out[i * 2]));
  trackCache.set(tracking, out);
  return out;
}

const look = { x: 0, y: 0 };

/** Where play is at a frame, for the camera to look at: x along the pitch and y across it. The object is reused. */
export function cameraLookAt(tracking: Tracking, fps: number, frame: number): { x: number; y: number } {
  const track = cameraTrack(tracking, fps);
  const count = track.length / 2;
  const position = Math.max(0, frame) / tracking.every;
  const i = Math.min(count - 1, Math.floor(position));
  const j = Math.min(count - 1, i + 1);
  const u = position - i;
  look.x = track[i * 2] + (track[j * 2] - track[i * 2]) * u;
  look.y = track[i * 2 + 1] + (track[j * 2 + 1] - track[i * 2 + 1]) * u;
  return look;
}
