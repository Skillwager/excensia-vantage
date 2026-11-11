import type { Tracking } from "./types";

/** Where everyone is at one instant, read from tracking between its samples. */
export interface Moment {
  /** False when the run has no tracking for this frame. */
  has: boolean;
  /** x, y per slot; NaN where a player is not seen. */
  home: Float32Array;
  away: Float32Array;
  ballSeen: boolean;
  /** The ball where it is seen; the best estimate of it where it is hidden; NaN when there is nothing to go on. */
  ballX: number;
  ballY: number;
  /** Seconds since the ball was last seen (0 when it is in view). */
  hiddenFor: number;
  /** Seconds until the ball is seen again (0 when it is in view, Infinity if never). */
  hiddenUntil: number;
  /** How far off the estimate could be, in metres (0 when the ball is in view). */
  doubt: number;
  /** 0..1: how much the estimate is worth showing. Fades when the ball has been gone a long while. */
  trust: number;
}

export function emptyMoment(tracking: Tracking | null): Moment {
  return {
    has: false,
    home: new Float32Array((tracking?.homeSlots ?? 0) * 2).fill(NaN),
    away: new Float32Array((tracking?.awaySlots ?? 0) * 2).fill(NaN),
    ballSeen: false,
    ballX: NaN,
    ballY: NaN,
    hiddenFor: 0,
    hiddenUntil: 0,
    doubt: 0,
    trust: 0,
  };
}

interface BallIndex {
  /** Index of the latest sample at or before i that sees the ball, or -1. */
  prev: Int32Array;
  /** Index of the first sample at or after i that sees the ball, or -1. */
  next: Int32Array;
}

const ballIndexCache = new WeakMap<Tracking, BallIndex>();

export function ballIndex(tracking: Tracking): BallIndex {
  let index = ballIndexCache.get(tracking);
  if (index) return index;
  const n = tracking.count;
  const prev = new Int32Array(n);
  const next = new Int32Array(n);
  let last = -1;
  for (let i = 0; i < n; i++) {
    if (tracking.ball[i * 2] === tracking.ball[i * 2]) last = i;
    prev[i] = last;
  }
  last = -1;
  for (let i = n - 1; i >= 0; i--) {
    if (tracking.ball[i * 2] === tracking.ball[i * 2]) last = i;
    next[i] = last;
  }
  index = { prev, next };
  ballIndexCache.set(tracking, index);
  return index;
}

/** A player moving faster than this between two samples has most likely swapped tracks, so is not blended. */
const MAX_PLAYER_SPEED = 14;
const MAX_BALL_SPEED = 48;

function catmull(p0: number, p1: number, p2: number, p3: number, u: number): number {
  const u2 = u * u;
  const u3 = u2 * u;
  return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (3 * p1 - p0 - 3 * p2 + p3) * u3);
}

function sampleTeam(data: Float32Array, slots: number, count: number, i: number, u: number, maxStep: number, out: Float32Array): void {
  const i0 = i > 0 ? i - 1 : i;
  const i2 = i + 1 < count ? i + 1 : i;
  const i3 = i + 2 < count ? i + 2 : i2;
  for (let p = 0; p < slots; p++) {
    const a = (i * slots + p) * 2;
    const b = (i2 * slots + p) * 2;
    const ax = data[a];
    const ay = data[a + 1];
    const bx = data[b];
    const by = data[b + 1];
    const aSeen = ax === ax;
    const bSeen = bx === bx;
    let x: number;
    let y: number;
    if (aSeen && bSeen) {
      const dx = bx - ax;
      const dy = by - ay;
      if (dx * dx + dy * dy > maxStep * maxStep) {
        // too far to have run: show the nearer sample rather than sliding across the pitch
        x = u < 0.5 ? ax : bx;
        y = u < 0.5 ? ay : by;
      } else {
        const z = (i0 * slots + p) * 2;
        const c = (i3 * slots + p) * 2;
        const zx = data[z];
        const cx = data[c];
        if (zx === zx && cx === cx && Math.abs(zx - ax) < maxStep && Math.abs(cx - bx) < maxStep && Math.abs(data[z + 1] - ay) < maxStep && Math.abs(data[c + 1] - by) < maxStep) {
          x = catmull(zx, ax, bx, cx, u);
          y = catmull(data[z + 1], ay, by, data[c + 1], u);
        } else {
          x = ax + dx * u;
          y = ay + dy * u;
        }
      }
    } else if (aSeen && u < 0.5) {
      x = ax;
      y = ay;
    } else if (bSeen && u >= 0.5) {
      x = bx;
      y = by;
    } else {
      x = NaN;
      y = NaN;
    }
    out[p * 2] = x;
    out[p * 2 + 1] = y;
  }
}

/**
 * Reads tracking at a frame (which may fall between samples) into `out`.
 * Players are eased along their path; the ball moves in straight lines.
 * When the ball is hidden the moment carries an estimate: on the line between where it was last seen
 * and where it is next seen for short gaps, otherwise where it was last seen, with a doubt that grows with time.
 */
export function sampleMoment(tracking: Tracking | null, frame: number, fps: number, out: Moment): Moment {
  if (!tracking || tracking.count === 0) {
    out.has = false;
    out.ballSeen = false;
    out.ballX = NaN;
    out.ballY = NaN;
    out.hiddenFor = 0;
    out.hiddenUntil = 0;
    out.doubt = 0;
    out.trust = 0;
    return out;
  }
  const { every, count } = tracking;
  const position = Math.max(0, frame) / every;
  let i = Math.floor(position);
  let u = position - i;
  if (i >= count - 1) {
    // past the last sample: hold it for one sample's length, then there is no tracking
    out.has = position <= count;
    i = count - 1;
    u = 0;
  } else out.has = true;
  if (!out.has) {
    out.home.fill(NaN);
    out.away.fill(NaN);
    out.ballSeen = false;
    out.ballX = NaN;
    out.ballY = NaN;
    out.hiddenFor = 0;
    out.hiddenUntil = 0;
    out.doubt = 0;
    out.trust = 0;
    return out;
  }

  const dt = every / fps;
  sampleTeam(tracking.home, tracking.homeSlots, count, i, u, MAX_PLAYER_SPEED * dt, out.home);
  sampleTeam(tracking.away, tracking.awaySlots, count, i, u, MAX_PLAYER_SPEED * dt, out.away);

  const ball = tracking.ball;
  const j = i + 1 < count ? i + 1 : i;
  const ax = ball[i * 2];
  const bx = ball[j * 2];
  const aSeen = ax === ax;
  const bSeen = bx === bx;
  const now = position * dt;

  if (aSeen && bSeen) {
    const ay = ball[i * 2 + 1];
    const by = ball[j * 2 + 1];
    const far = Math.hypot(bx - ax, by - ay) > MAX_BALL_SPEED * dt;
    out.ballSeen = true;
    out.ballX = far ? (u < 0.5 ? ax : bx) : ax + (bx - ax) * u;
    out.ballY = far ? (u < 0.5 ? ay : by) : ay + (by - ay) * u;
    out.hiddenFor = 0;
    out.hiddenUntil = 0;
    out.doubt = 0;
    out.trust = 1;
    return out;
  }
  if ((aSeen && u < 0.5) || (bSeen && u >= 0.5)) {
    const k = aSeen && u < 0.5 ? i : j;
    out.ballSeen = true;
    out.ballX = ball[k * 2];
    out.ballY = ball[k * 2 + 1];
    out.hiddenFor = 0;
    out.hiddenUntil = 0;
    out.doubt = 0;
    out.trust = 1;
    return out;
  }

  // hidden: estimate from the sightings either side
  const index = ballIndex(tracking);
  const before = aSeen ? i : index.prev[i];
  const after = bSeen ? j : index.next[j];
  out.ballSeen = false;
  const since = before >= 0 ? now - before * dt : Infinity;
  const until = after >= 0 ? after * dt - now : Infinity;
  out.hiddenFor = since;
  out.hiddenUntil = until;
  if (before < 0 && after < 0) {
    out.ballX = NaN;
    out.ballY = NaN;
    out.doubt = 0;
    out.trust = 0;
    return out;
  }
  const gap = since + until;
  if (before >= 0 && after >= 0 && gap <= 8) {
    const k = gap > 0 ? since / gap : 0;
    out.ballX = ball[before * 2] + (ball[after * 2] - ball[before * 2]) * k;
    out.ballY = ball[before * 2 + 1] + (ball[after * 2 + 1] - ball[before * 2 + 1]) * k;
    out.trust = 1;
  } else if (before >= 0 && (after < 0 || until > 1.2)) {
    out.ballX = ball[before * 2];
    out.ballY = ball[before * 2 + 1];
    out.trust = since <= 2 ? 1 : since >= 7 ? 0.35 : 1 - ((since - 2) / 5) * 0.65;
  } else {
    // about to come back into view: the estimate moves to where it will be seen
    out.ballX = ball[after * 2];
    out.ballY = ball[after * 2 + 1];
    out.trust = 0.6;
  }
  const nearest = Math.min(since, until);
  out.doubt = Math.min(9, 1.6 + 1.1 * nearest);
  return out;
}

/** How far back and ahead the trails go, and in how many steps each way. */
export const TRAIL_SECONDS = 3;
export const TRAIL_STEPS = 9;

/** Positions over the seconds either side of a frame, for drawing where everyone came from and goes next. */
export interface Trails {
  /** Moments kept each side of now. */
  steps: number;
  /** (2 * steps + 1) moments of positions, the middle one being now. */
  home: Float32Array;
  away: Float32Array;
  /** The ball where it was seen, NaN where it was not. */
  ball: Float32Array;
  valid: Uint8Array;
  /** Working space. */
  then: Moment;
}

export function emptyTrails(tracking: Tracking | null, steps = TRAIL_STEPS): Trails {
  const total = steps * 2 + 1;
  return {
    steps,
    home: new Float32Array(total * (tracking?.homeSlots ?? 0) * 2),
    away: new Float32Array(total * (tracking?.awaySlots ?? 0) * 2),
    ball: new Float32Array(total * 2),
    valid: new Uint8Array(total),
    then: emptyMoment(tracking),
  };
}

/** Reads `seconds` either side of frame `at` into `out`. `last` is the last frame of the match. */
export function readTrails(tracking: Tracking, fps: number, last: number, at: number, out: Trails, seconds = TRAIL_SECONDS): void {
  const step = (seconds * fps) / out.steps;
  const total = out.steps * 2 + 1;
  for (let k = 0; k < total; k++) {
    const f = at + (k - out.steps) * step;
    out.valid[k] = 0;
    if (f < 0 || f > last) continue;
    sampleMoment(tracking, f, fps, out.then);
    if (!out.then.has) continue;
    out.valid[k] = 1;
    out.home.set(out.then.home, k * out.then.home.length);
    out.away.set(out.then.away, k * out.then.away.length);
    out.ball[k * 2] = out.then.ballSeen ? out.then.ballX : NaN;
    out.ball[k * 2 + 1] = out.then.ballSeen ? out.then.ballY : NaN;
  }
}
