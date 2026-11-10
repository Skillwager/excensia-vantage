import type { Moment } from "../model/moment";
import type { TeamShape } from "../model/shape";
import { drawLabel, labelWidth } from "./labels";
import { C, alpha, trackLabel } from "./palette";

/** How pitch metres map onto a top-down drawing. +y (the far touchline in the match view) is at the top. */
export interface TopLayout {
  /** Pixel position of the pitch centre. */
  cx: number;
  cy: number;
  /** Pixels per metre along and across the pitch. */
  sx: number;
  sy: number;
  length: number;
  width: number;
}

export const px = (lay: TopLayout, x: number) => lay.cx + x * lay.sx;
export const py = (lay: TopLayout, y: number) => lay.cy - y * lay.sy;

/**
 * Fits the pitch into a box. `stretch` lets a wide box pull the pitch out sideways (a tactics-board look) up to
 * that ratio before grass fills the rest; 1 keeps true proportions.
 */
export function fitPitch(w: number, h: number, pitch: { length: number; width: number }, pad: number, stretch = 1): TopLayout {
  const availW = Math.max(10, w - pad * 2);
  const availH = Math.max(10, h - pad * 2);
  const sy = Math.min(availH / pitch.width, availW / pitch.length);
  const sx = Math.min(availW / pitch.length, sy * Math.max(1, stretch));
  return { cx: w / 2, cy: h / 2, sx, sy, length: pitch.length, width: pitch.width };
}

/** A layout that shows `metres` of pitch across the box, centred on a point (for the zoomed tiles). */
export function zoomPitch(w: number, h: number, pitch: { length: number; width: number }, x: number, y: number, metres: number): TopLayout {
  const s = w / metres;
  return { cx: w / 2 - x * s, cy: h / 2 + y * s, sx: s, sy: s, length: pitch.length, width: pitch.width };
}

/** A layout that frames a box of the pitch (in metres) inside the drawing, in true proportion. */
export function framePitch(
  w: number,
  h: number,
  pitch: { length: number; width: number },
  box: { minX: number; maxX: number; minY: number; maxY: number },
  pad: number,
): TopLayout {
  const s = Math.min((w - pad * 2) / Math.max(1, box.maxX - box.minX), (h - pad * 2) / Math.max(1, box.maxY - box.minY));
  const midX = (box.minX + box.maxX) / 2;
  const midY = (box.minY + box.maxY) / 2;
  return { cx: w / 2 - midX * s, cy: h / 2 + midY * s, sx: s, sy: s, length: pitch.length, width: pitch.width };
}

const GRASS = "#0f2e1c";
const GRASS_ALT = "#123522";
const GRASS_EDGE = "#0c2616";

/** The pitch from above: grass, mown bands and every marking. */
export function drawPitchTop(ctx: CanvasRenderingContext2D, w: number, h: number, lay: TopLayout, lineAlpha = 0.74): void {
  const hl = lay.length / 2;
  const hw = lay.width / 2;
  ctx.fillStyle = GRASS_EDGE;
  ctx.fillRect(0, 0, w, h);
  const left = px(lay, -hl);
  const top = py(lay, hw);
  const pw = lay.length * lay.sx;
  const ph = lay.width * lay.sy;
  ctx.fillStyle = GRASS;
  ctx.fillRect(left, top, pw, ph);
  const bands = 12;
  ctx.fillStyle = GRASS_ALT;
  for (let i = 0; i < bands; i += 2) ctx.fillRect(left + (i * pw) / bands, top, pw / bands, ph);
  ctx.fillStyle = "rgba(255, 255, 255, 0.018)";
  for (let i = 0; i < 6; i += 2) ctx.fillRect(left, top + (i * ph) / 6, pw, ph / 6);

  ctx.strokeStyle = alpha(C.ink, lineAlpha);
  ctx.fillStyle = alpha(C.ink, lineAlpha);
  ctx.lineWidth = Math.max(1, Math.min(1.5, lay.sy * 0.22));
  ctx.lineJoin = "miter";
  const X = (x: number) => Math.round(px(lay, x)) + 0.5;
  const Y = (y: number) => Math.round(py(lay, y)) + 0.5;
  ctx.strokeRect(X(-hl), Y(hw), X(hl) - X(-hl), Y(-hw) - Y(hw));
  ctx.beginPath();
  ctx.moveTo(X(0), Y(hw));
  ctx.lineTo(X(0), Y(-hw));
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(px(lay, 0), py(lay, 0), 9.15 * lay.sx, 9.15 * lay.sy, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px(lay, 0), py(lay, 0), Math.max(1.6, lay.sy * 0.35), 0, Math.PI * 2);
  ctx.fill();
  for (const side of [-1, 1]) {
    const gx = side * hl;
    const box = (depth: number, half: number) => {
      const x0 = X(gx);
      const x1 = X(gx - side * depth);
      ctx.strokeRect(Math.min(x0, x1), Y(half), Math.abs(x1 - x0), Y(-half) - Y(half));
    };
    box(16.5, 20.16);
    box(5.5, 9.16);
    // the goal, just behind the line
    const g0 = X(gx);
    const g1 = X(gx + side * 1.8);
    ctx.strokeRect(Math.min(g0, g1), Y(3.66), Math.abs(g1 - g0), Y(-3.66) - Y(3.66));
    ctx.beginPath();
    ctx.arc(px(lay, gx - side * 11), py(lay, 0), Math.max(1.3, lay.sy * 0.28), 0, Math.PI * 2);
    ctx.fill();
    const reach = Math.acos(5.5 / 9.15);
    ctx.beginPath();
    // an ellipse arc, so it stays true when the pitch is stretched; the canvas y axis points down
    ctx.ellipse(px(lay, gx - side * 11), py(lay, 0), 9.15 * lay.sx, 9.15 * lay.sy, 0, side > 0 ? Math.PI - reach : -reach, side > 0 ? Math.PI + reach : reach);
    ctx.stroke();
  }
}

export interface TopPlayersOptions {
  labels: boolean;
  keepers: [number, number];
  homeSlots: number;
  /** Dot radius in pixels. */
  radius: number;
  labelSize?: number;
  /** The drawing's width, so labels flip to the left near its right edge. */
  width: number;
}

/** Both teams as dots, with their track labels. */
export function drawPlayersTop(ctx: CanvasRenderingContext2D, lay: TopLayout, moment: Moment, options: TopPlayersOptions): void {
  if (!moment.has) return;
  const density = ctx.getTransform().a;
  const style = { size: options.labelSize ?? 10.5, colour: alpha(C.ink, 0.92), halo: "rgba(8, 20, 13, 0.7)" };
  for (const team of [0, 1] as const) {
    const data = team === 0 ? moment.home : moment.away;
    const colour = team === 0 ? C.home : C.away;
    for (let p = 0; p < data.length / 2; p++) {
      const x = data[p * 2];
      if (x !== x) continue;
      const sx = px(lay, x);
      const sy = py(lay, data[p * 2 + 1]);
      ctx.beginPath();
      ctx.arc(sx, sy, options.radius, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(6, 14, 10, 0.6)";
      ctx.stroke();
      if (!options.labels) continue;
      const text = trackLabel(team, p, options.keepers[team], options.homeSlots);
      const flip = sx + options.radius + 5 + labelWidth(text, density, style) > options.width - 2;
      drawLabel(ctx, text, flip ? sx - options.radius - 4 : sx + options.radius + 4, sy + options.radius + 5, density, {
        ...style,
        align: flip ? "right" : "left",
      });
    }
  }
}

/** The ball, or a dashed ring around where it is thought to be while it is out of sight. */
export function drawBallTop(ctx: CanvasRenderingContext2D, lay: TopLayout, moment: Moment, options: { radius: number; caption: boolean; maxDoubt?: number }): void {
  if (!moment.has || moment.ballX !== moment.ballX) return;
  const sx = px(lay, moment.ballX);
  const sy = py(lay, moment.ballY);
  if (moment.ballSeen) {
    ctx.beginPath();
    ctx.arc(sx, sy, options.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(4, 10, 7, 0.75)";
    ctx.stroke();
    return;
  }
  if (moment.trust <= 0.05) return;
  const doubt = Math.min(options.maxDoubt ?? 9, Math.max(1.6, moment.doubt));
  const rx = Math.max(9, doubt * lay.sx);
  const ry = Math.max(9, doubt * lay.sy);
  ctx.save();
  ctx.setLineDash([4, 3.5]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = alpha(C.lime, 0.9 * moment.trust);
  ctx.beginPath();
  ctx.ellipse(sx, sy, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
  if (options.caption) {
    ctx.save();
    ctx.globalAlpha = Math.max(0.55, moment.trust);
    drawLabel(ctx, "BALL HIDDEN", sx, sy + ry + 11, ctx.getTransform().a, {
      size: 10,
      colour: C.lime,
      halo: "rgba(8, 20, 13, 0.75)",
      align: "center",
      tracking: 0.7,
    });
    ctx.restore();
  }
}

/** A team's outline: the hull of its outfield players, lightly filled. */
export function drawHullTop(ctx: CanvasRenderingContext2D, lay: TopLayout, shape: TeamShape | null, colour: string, fill = 0.13, stroke = 0.7): void {
  if (!shape || shape.hull.length < 6) return;
  ctx.beginPath();
  for (let i = 0; i < shape.hull.length; i += 2) {
    const sx = px(lay, shape.hull[i]);
    const sy = py(lay, shape.hull[i + 1]);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.closePath();
  ctx.fillStyle = alpha(colour, fill);
  ctx.fill();
  ctx.lineWidth = 1.25;
  ctx.lineJoin = "round";
  ctx.strokeStyle = alpha(colour, stroke);
  ctx.stroke();
}

/** A small cross at a team's centre. */
export function drawCentreTop(ctx: CanvasRenderingContext2D, lay: TopLayout, shape: TeamShape | null, colour: string, size = 6): void {
  if (!shape) return;
  const sx = px(lay, shape.cx);
  const sy = py(lay, shape.cy);
  ctx.lineWidth = 3.4;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(6, 14, 10, 0.7)";
  for (const pass of [0, 1]) {
    ctx.beginPath();
    ctx.moveTo(sx - size, sy);
    ctx.lineTo(sx + size, sy);
    ctx.moveTo(sx, sy - size);
    ctx.lineTo(sx, sy + size);
    ctx.stroke();
    if (pass === 0) {
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = colour;
    }
  }
}

/** The defensive line: a dashed line across the pitch at the back four. */
export function drawBackLineTop(ctx: CanvasRenderingContext2D, lay: TopLayout, shape: TeamShape | null, colour: string): void {
  if (!shape) return;
  const sx = Math.round(px(lay, shape.lineX)) + 0.5;
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.25;
  ctx.strokeStyle = alpha(colour, 0.8);
  ctx.beginPath();
  ctx.moveTo(sx, py(lay, lay.width / 2));
  ctx.lineTo(sx, py(lay, -lay.width / 2));
  ctx.stroke();
  ctx.restore();
}
