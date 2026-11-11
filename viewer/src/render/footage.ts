import { cameraAt, cropAround, cropAroundPoints, pixelsPerMetre, toPicture, type Crop, type PicturePoint } from "../model/footage";
import type { EvidenceKind } from "../model/explain";
import type { Moment, Trails } from "../model/moment";
import type { ShapePair, TeamShape } from "../model/shape";
import type { Run } from "../model/types";
import type { Playback } from "../state/playback";
import { C, alpha } from "./palette";

/**
 * Drawing on the match video itself. Where a run file carries the camera of its video, the
 * evidence can be shown on the real picture: the tracking is put through the camera matrix of
 * the frame on screen, so an outline or a trail lands on the players it belongs to.
 */

/** The picture on screen and the camera that took it. */
export interface Footage {
  video: HTMLVideoElement;
  /** Pitch metres to the video's own pixels, for the frame on screen. */
  matrix: Float64Array;
  width: number;
  height: number;
  /** The frame the picture shows: positions are read at this frame, not between frames. */
  frame: number;
}

/**
 * The picture for a frame, or null when there is no video that can be drawn from, the run has no
 * camera, or the camera is not known at this frame. `matrix` is the caller's own nine-value
 * buffer, so nothing is allocated on a repaint.
 */
export function footageAt(playback: Playback, run: Run, frame: number, matrix: Float64Array): Footage | null {
  const camera = run.camera;
  const video = playback.picture();
  if (!camera || !video) return null;
  // the video shows whole frames; a clock reading part-way through one still shows that frame
  const shown = Math.floor(frame + 1e-6);
  if (!cameraAt(camera, shown, matrix)) return null;
  const su = video.videoWidth / camera.width;
  const sv = video.videoHeight / camera.height;
  matrix[0] *= su;
  matrix[1] *= su;
  matrix[2] *= su;
  matrix[3] *= sv;
  matrix[4] *= sv;
  matrix[5] *= sv;
  return { video, matrix, width: video.videoWidth, height: video.videoHeight, frame: shown };
}

/** How a part of the picture sits on a canvas: canvas x = ox + (u - x) * k. */
interface View {
  x: number;
  y: number;
  k: number;
  ox: number;
  oy: number;
}

const point: PicturePoint = { u: 0, v: 0 };

/** Fills the canvas with a part of the picture. */
function drawCrop(ctx: CanvasRenderingContext2D, footage: Footage, crop: Crop, w: number, h: number): View {
  ctx.drawImage(footage.video, crop.x, crop.y, crop.width, crop.height, 0, 0, w, h);
  return { x: crop.x, y: crop.y, k: w / crop.width, ox: 0, oy: 0 };
}

/** Fits the whole picture inside the canvas, with the surround in the pitch's own dark. */
function drawWhole(ctx: CanvasRenderingContext2D, footage: Footage, w: number, h: number): View {
  const k = Math.min(w / footage.width, h / footage.height);
  const ox = (w - footage.width * k) / 2;
  const oy = (h - footage.height * k) / 2;
  ctx.fillStyle = C.pitch;
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(footage.video, 0, 0, footage.width, footage.height, ox, oy, footage.width * k, footage.height * k);
  return { x: 0, y: 0, k, ox, oy };
}

const sx = (view: View, u: number) => view.ox + (u - view.x) * view.k;
const sy = (view: View, v: number) => view.oy + (v - view.y) * view.k;

/** A slight darkening, so outlines in the team colours read against grass in sunlight. */
function dim(ctx: CanvasRenderingContext2D, w: number, h: number, amount: number): void {
  ctx.fillStyle = `rgba(6, 10, 12, ${amount})`;
  ctx.fillRect(0, 0, w, h);
}

function hull(ctx: CanvasRenderingContext2D, view: View, m: Float64Array, shape: TeamShape | null, colour: string, fill: number, width: number): void {
  if (!shape || shape.hull.length < 6) return;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < shape.hull.length; i += 2) {
    if (!toPicture(m, shape.hull[i], shape.hull[i + 1], point)) return; // an outline with a corner behind the camera is not drawn at all
    if (started) ctx.lineTo(sx(view, point.u), sy(view, point.v));
    else ctx.moveTo(sx(view, point.u), sy(view, point.v));
    started = true;
  }
  ctx.closePath();
  ctx.fillStyle = alpha(colour, fill);
  ctx.fill();
  ctx.lineJoin = "round";
  ctx.lineWidth = width + 2;
  ctx.strokeStyle = "rgba(6, 10, 12, 0.55)";
  ctx.stroke();
  ctx.lineWidth = width;
  ctx.strokeStyle = alpha(colour, 0.95);
  ctx.stroke();
}

/** A small mark at each player's feet, in his team's colour. */
function feet(ctx: CanvasRenderingContext2D, view: View, m: Float64Array, data: Float32Array, colour: string, radius: number): void {
  for (let p = 0; p < data.length / 2; p++) {
    const x = data[p * 2];
    if (x !== x || !toPicture(m, x, data[p * 2 + 1], point)) continue;
    const cx = sx(view, point.u);
    const cy = sy(view, point.v);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(6, 10, 12, 0.85)";
    ctx.stroke();
  }
}

/** A ring round the ball where it is tracked; a broken, fainter one round its estimate where it is not. */
function ballRing(ctx: CanvasRenderingContext2D, view: View, m: Float64Array, moment: Moment, radius: number): void {
  if (moment.ballX !== moment.ballX || !toPicture(m, moment.ballX, moment.ballY, point)) return;
  if (!moment.ballSeen && moment.trust < 0.25) return;
  const cx = sx(view, point.u);
  const cy = sy(view, point.v);
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = "rgba(6, 10, 12, 0.6)";
  ctx.stroke();
  ctx.lineWidth = 1.8;
  ctx.setLineDash(moment.ballSeen ? [] : [4, 4]);
  ctx.strokeStyle = alpha(C.lime, moment.ballSeen ? 1 : 0.7);
  ctx.stroke();
  ctx.setLineDash([]);
}

/** Every seen player of both teams in the picture: u0, v0, u1, v1 … */
function playersInPicture(m: Float64Array, moment: Moment, out: number[]): number[] {
  out.length = 0;
  for (const data of [moment.home, moment.away]) {
    for (let p = 0; p < data.length / 2; p++) {
      const x = data[p * 2];
      if (x !== x || !toPicture(m, x, data[p * 2 + 1], point)) continue;
      out.push(point.u, point.v);
    }
  }
  return out;
}

const scratchPoints: number[] = [];

/**
 * 1. The ball, close up, cut out of the real picture: about twenty metres of pitch around the
 * ball (or around `focus` while the ball is not tracked). False when it cannot be drawn.
 */
export function drawBallFootage(ctx: CanvasRenderingContext2D, w: number, h: number, footage: Footage, moment: Moment, focus: { x: number; y: number }): boolean {
  const m = footage.matrix;
  const scale = pixelsPerMetre(m, focus.x, focus.y);
  if (scale <= 0 || !toPicture(m, focus.x, focus.y, point)) return false;
  if (point.u < -footage.width * 0.1 || point.u > footage.width * 1.1 || point.v < -footage.height * 0.1 || point.v > footage.height * 1.1) return false;
  const span = Math.min(footage.width, Math.max(200, 13 * scale));
  // a little above the spot on the ground, so the players standing round it are whole
  const crop = cropAround(point.u, point.v - (span * h) / w / 7, span, w, h, footage.width, footage.height);
  const view = drawCrop(ctx, footage, crop, w, h);
  ballRing(ctx, view, m, moment, Math.max(7, Math.min(14, 0.75 * scale * view.k)));
  return true;
}

/** 2. Both teams' outlines on the real picture, framed on the players. False when it cannot be drawn. */
export function drawShapeFootage(ctx: CanvasRenderingContext2D, w: number, h: number, footage: Footage, moment: Moment, shapes: ShapePair): boolean {
  const m = footage.matrix;
  const points = playersInPicture(m, moment, scratchPoints);
  const crop = cropAroundPoints(points, footage.width * 0.03, footage.height * 0.07, w, h, footage.width, footage.height);
  if (!crop) return false;
  const view = drawCrop(ctx, footage, crop, w, h);
  dim(ctx, w, h, 0.22);
  hull(ctx, view, m, shapes.home, C.home, 0.24, 1.4);
  hull(ctx, view, m, shapes.away, C.away, 0.24, 1.4);
  feet(ctx, view, m, moment.home, C.home, 2);
  feet(ctx, view, m, moment.away, C.away, 2);
  return true;
}

/** The whole picture with both outlines, a mark at every player's feet and a ring round the ball: the team-shape panel on the footage. */
export function drawShapePanelFootage(ctx: CanvasRenderingContext2D, w: number, h: number, footage: Footage, moment: Moment, shapes: ShapePair): boolean {
  const m = footage.matrix;
  const view = drawWhole(ctx, footage, w, h);
  ctx.save();
  ctx.beginPath();
  ctx.rect(view.ox, view.oy, footage.width * view.k, footage.height * view.k);
  ctx.clip();
  dim(ctx, w, h, 0.18);
  hull(ctx, view, m, shapes.home, C.home, 0.2, 1.8);
  hull(ctx, view, m, shapes.away, C.away, 0.2, 1.8);
  const radius = Math.max(2.4, Math.min(4.5, w / 170));
  feet(ctx, view, m, moment.home, C.home, radius);
  feet(ctx, view, m, moment.away, C.away, radius);
  ballRing(ctx, view, m, moment, radius * 2.4);
  ctx.restore();
  return true;
}

/**
 * 3. The seconds before and after, on the real picture: where the ball and the players round it
 * came from (solid) and where they go next (broken). Positions from other moments are put
 * through the camera of the frame on screen, so a trail lies on the grass the player ran over.
 */
export function drawTimingFootage(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  footage: Footage,
  now: Moment,
  focus: { x: number; y: number },
  trails: Trails,
  homeSlots: number,
  awaySlots: number,
): boolean {
  const m = footage.matrix;
  const scale = pixelsPerMetre(m, focus.x, focus.y);
  if (scale <= 0 || !toPicture(m, focus.x, focus.y, point)) return false;
  if (point.u < -footage.width * 0.1 || point.u > footage.width * 1.1 || point.v < -footage.height * 0.1 || point.v > footage.height * 1.1) return false;
  const span = Math.min(footage.width, Math.max(360, 34 * scale));
  const crop = cropAround(point.u, point.v - (span * h) / w / 9, span, w, h, footage.width, footage.height);
  const view = drawCrop(ctx, footage, crop, w, h);
  dim(ctx, w, h, 0.26);
  runs(ctx, view, m, trails, homeSlots, awaySlots, 1);
  feet(ctx, view, m, now.home, C.home, 2.6);
  feet(ctx, view, m, now.away, C.away, 2.6);
  ballRing(ctx, view, m, now, 6);
  return true;
}

/** Where everyone came from (solid) and goes next (broken), and the ball's path where it was tracked. */
function runs(ctx: CanvasRenderingContext2D, view: View, m: Float64Array, trails: Trails, homeSlots: number, awaySlots: number, weight: number): void {
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const steps = trails.steps;
  const team = (data: Float32Array, slots: number, colour: string) => {
    for (const side of [-1, 1]) {
      ctx.beginPath();
      for (let p = 0; p < slots; p++) {
        let started = false;
        for (let j = 0; j <= steps; j++) {
          const k = steps + side * j;
          const x = data[(k * slots + p) * 2];
          if (!trails.valid[k] || x !== x || !toPicture(m, x, data[(k * slots + p) * 2 + 1], point)) break;
          if (started) ctx.lineTo(sx(view, point.u), sy(view, point.v));
          else ctx.moveTo(sx(view, point.u), sy(view, point.v));
          started = true;
        }
      }
      ctx.lineWidth = (side < 0 ? 2 : 1.4) * weight;
      ctx.setLineDash(side < 0 ? [] : [3 * weight, 4 * weight]);
      ctx.strokeStyle = alpha(colour, side < 0 ? 0.95 : 0.8);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  };
  team(trails.home, homeSlots, C.home);
  team(trails.away, awaySlots, C.away);
  for (const side of [-1, 1]) {
    ctx.beginPath();
    let drawing = false;
    for (let j = 0; j <= steps; j++) {
      const k = steps + side * j;
      const x = trails.ball[k * 2];
      if (!trails.valid[k] || x !== x || !toPicture(m, x, trails.ball[k * 2 + 1], point)) {
        drawing = false;
        continue;
      }
      if (drawing) ctx.lineTo(sx(view, point.u), sy(view, point.v));
      else ctx.moveTo(sx(view, point.u), sy(view, point.v));
      drawing = true;
    }
    ctx.lineWidth = 2 * weight;
    ctx.setLineDash(side < 0 ? [] : [3 * weight, 4 * weight]);
    ctx.strokeStyle = alpha("#ffffff", side < 0 ? 0.95 : 0.8);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

/**
 * The evidence chosen in the inspector, drawn over the large match video itself (on a clear
 * canvas lying on the video, which shows through): a ring round the ball, both teams' outlines,
 * or the runs of the seconds before and after.
 */
export function drawEvidenceOnVideo(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  footage: Footage,
  kind: EvidenceKind,
  moment: Moment,
  shapes: ShapePair | null,
  trails: Trails | null,
  homeSlots: number,
  awaySlots: number,
): void {
  const m = footage.matrix;
  // the video sits inside its box whole, centred: the same fit as the element's own
  const k = Math.min(w / footage.width, h / footage.height);
  const view: View = { x: 0, y: 0, k, ox: (w - footage.width * k) / 2, oy: (h - footage.height * k) / 2 };
  ctx.save();
  ctx.beginPath();
  ctx.rect(view.ox, view.oy, footage.width * k, footage.height * k);
  ctx.clip();
  const mark = Math.max(3, Math.min(6, w / 230));
  if (kind === "shape" && shapes) {
    hull(ctx, view, m, shapes.home, C.home, 0.16, 2.2);
    hull(ctx, view, m, shapes.away, C.away, 0.16, 2.2);
    feet(ctx, view, m, moment.home, C.home, mark);
    feet(ctx, view, m, moment.away, C.away, mark);
  } else if (kind === "timing" && trails) {
    runs(ctx, view, m, trails, homeSlots, awaySlots, 1.5);
    feet(ctx, view, m, moment.home, C.home, mark);
    feet(ctx, view, m, moment.away, C.away, mark);
  }
  ballRing(ctx, view, m, moment, mark * 2.6);
  ctx.restore();
}
