import { lowerIndex } from "../model/derive";
import { formatTick, tickMinutes } from "../model/time";
import { A, D, H, X, type Run } from "../model/types";
import { C, FONT_SANS, LABEL_COLOR, alpha } from "./palette";

/** The stretch of the match the timeline is showing, in frames. */
export interface TimelineView {
  start: number;
  span: number;
}

/** Vertical layout of the timeline canvas, CSS pixels. Shared with the row labels beside it. */
export const TL = {
  laneTop: 0,
  laneH: 11,
  rowY: [15, 44, 73] as const,
  rowH: 18,
  axisY: 101,
  height: 124,
} as const;

/** Below this confidence a call is drawn hatched. */
export const UNSURE_BELOW = 0.6;

export const frameAtX = (x: number, width: number, view: TimelineView) => view.start + (x / Math.max(1, width)) * view.span;
export const xAtFrame = (frame: number, width: number, view: TimelineView) => ((frame - view.start) / view.span) * width;

export function clampView(view: TimelineView, frames: number, minSpan: number): TimelineView {
  const span = Math.min(frames, Math.max(Math.min(minSpan, frames), view.span));
  const start = Math.min(frames - span, Math.max(0, view.start));
  return { start, span };
}

function addSpan(target: Float32Array, xa: number, xb: number): void {
  const n = target.length;
  if (xb <= 0 || xa >= n) return;
  const a = Math.max(0, xa);
  const b = Math.min(n, xb);
  const i0 = Math.floor(a);
  const i1 = Math.floor(b);
  if (i0 === i1) {
    target[i0] += b - a;
    return;
  }
  target[i0] += i0 + 1 - a;
  for (let i = i0 + 1; i < i1; i++) target[i] += 1;
  if (i1 < n) target[i1] += b - i1;
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Diagonal hatching across a set of column ranges of one row. */
function hatch(ctx: CanvasRenderingContext2D, ranges: number[], y: number, h: number, colour: string): void {
  if (!ranges.length) return;
  ctx.save();
  ctx.beginPath();
  for (let i = 0; i < ranges.length; i += 2) ctx.rect(ranges[i], y, ranges[i + 1] - ranges[i], h);
  ctx.clip();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  const first = ranges[0];
  const last = ranges[ranges.length - 1];
  for (let x = Math.floor((first - h) / 5) * 5; x < last + h; x += 5) {
    ctx.moveTo(x, y + h + 1);
    ctx.lineTo(x + h + 2, y - 1);
  }
  ctx.stroke();
  ctx.restore();
}

/**
 * Paints the rows, the change markers and the time axis for a view of the match.
 * The context is scaled for the screen; `width` is in CSS pixels and `dpr` is the pixel density.
 * Each device pixel column is shaded by how much of its time carries the row's label, so a whole match reads
 * truthfully at any zoom.
 */
export function paintTimeline(ctx: CanvasRenderingContext2D, width: number, dpr: number, run: Run, view: TimelineView): void {
  ctx.clearRect(0, 0, width, TL.height);
  const cols = Math.max(1, Math.round(width * dpr));
  const perFrame = cols / view.span;
  const viewEnd = view.start + view.span;

  // coverage per device column: sure and unsure for each of home, away and dead ball; and no answer
  const sure = [new Float32Array(cols), new Float32Array(cols), new Float32Array(cols)];
  const unsure = [new Float32Array(cols), new Float32Array(cols), new Float32Array(cols)];
  const none = new Float32Array(cols);
  const { segments, segStart } = run;
  for (let i = lowerIndex(segStart, Math.floor(view.start)); i < segments.length; i++) {
    const s = segments[i];
    if (s.start >= viewEnd) break;
    const xa = (Math.max(s.start, view.start) - view.start) * perFrame;
    const xb = (Math.min(s.end + 1, viewEnd) - view.start) * perFrame;
    if (s.label === X) addSpan(none, xa, xb);
    else addSpan(s.confidence !== null && s.confidence < UNSURE_BELOW ? unsure[s.label] : sure[s.label], xa, xb);
  }

  const px1 = 1 / dpr;
  const rows: { label: 0 | 1 | 2; y: number }[] = [
    { label: H, y: TL.rowY[0] },
    { label: A, y: TL.rowY[1] },
    { label: D, y: TL.rowY[2] },
  ];
  for (const row of rows) {
    const colour = LABEL_COLOR[row.label];
    ctx.save();
    roundedRect(ctx, 0, row.y, width, TL.rowH, 3);
    ctx.clip();
    ctx.fillStyle = "rgba(255, 255, 255, 0.058)";
    ctx.fillRect(0, row.y, width, TL.rowH);

    // filled columns, joined into runs of the same strength
    const s = sure[row.label];
    const u = unsure[row.label];
    ctx.fillStyle = colour;
    let from = 0;
    let level = -1;
    for (let c = 0; c <= cols; c++) {
      const value = c < cols ? Math.min(1, s[c] + u[c] * 0.34) : -2;
      const stepped = value <= 0.004 ? 0 : Math.round(Math.pow(value, 0.85) * 32) / 32;
      if (stepped !== level) {
        if (level > 0) {
          ctx.globalAlpha = level;
          ctx.fillRect(from * px1, row.y, (c - from) * px1, TL.rowH);
        }
        from = c;
        level = stepped;
      }
    }
    ctx.globalAlpha = 1;

    // no answer: a quiet band across every row
    ctx.fillStyle = alpha(C.ink, 0.07);
    from = -1;
    for (let c = 0; c <= cols; c++) {
      const on = c < cols && none[c] >= 0.5;
      if (on && from < 0) from = c;
      else if (!on && from >= 0) {
        ctx.fillRect(from * px1, row.y, (c - from) * px1, TL.rowH);
        from = -1;
      }
    }

    // hatching wherever the call is unsure (or there is none), once a stretch is wide enough to read
    const ranges = (data: Float32Array): number[] => {
      const out: number[] = [];
      let start = -1;
      for (let c = 0; c <= cols; c++) {
        const on = c < cols && data[c] >= 0.5;
        if (on && start < 0) start = c;
        else if (!on && start >= 0) {
          if ((c - start) * px1 >= 5) out.push(start * px1, c * px1);
          start = -1;
        }
      }
      return out;
    };
    hatch(ctx, ranges(u), row.y, TL.rowH, alpha(colour, 0.95));
    hatch(ctx, ranges(none), row.y, TL.rowH, alpha(C.ink, 0.3));
    ctx.restore();
  }

  // changes of possession: a tick in the lane above the rows
  const { changes, changeFrame } = run;
  if (changes.length) {
    let first = lowerIndex(changeFrame, Math.floor(view.start));
    if (changeFrame[first] < view.start) first++;
    let last = first;
    while (last < changes.length && changeFrame[last] < viewEnd) last++;
    const shown = last - first;
    const dense = shown > width / 9;
    for (let i = first; i < last; i++) {
      const change = changes[i];
      const x = xAtFrame(change.frame, width, view);
      const colour = change.to === X ? C.dim : LABEL_COLOR[change.to];
      if (dense) {
        // too many to tell apart: a faint tick each, so the lane reads as how busy the match was
        ctx.fillStyle = alpha(colour, 0.5);
        ctx.fillRect(Math.round(x * dpr) / dpr, TL.laneTop + 7, px1 >= 1 ? 1 : px1 * 1.5, TL.laneH - 8);
      } else {
        ctx.fillStyle = colour;
        roundedRect(ctx, x - 1.25, TL.laneTop + 2, 2.5, TL.laneH - 3, 1.2);
        ctx.fill();
        if (change.confidence !== null && change.confidence < UNSURE_BELOW) {
          ctx.beginPath();
          ctx.arc(x, TL.laneTop + 1.2, 1.6, 0, Math.PI * 2);
          ctx.fillStyle = C.ember;
          ctx.fill();
        }
      }
    }
  }

  // the time axis
  ctx.fillStyle = alpha(C.ink, 0.14);
  ctx.fillRect(0, TL.axisY, width, 1);
  const seconds = view.span / run.fps;
  const step = tickMinutes(seconds, width);
  const stepSeconds = step * 60;
  const minor = step >= 15 ? 5 : step >= 5 ? 1 : step >= 1 ? step / 2 : step / 3;
  const startSeconds = view.start / run.fps;
  const endSeconds = viewEnd / run.fps;
  ctx.fillStyle = alpha(C.ink, 0.2);
  for (let t = Math.ceil(startSeconds / (minor * 60) - 1e-9) * minor * 60; t <= endSeconds + 1e-6; t += minor * 60) {
    const x = xAtFrame(t * run.fps, width, view);
    ctx.fillRect(Math.min(width - 1, Math.round(x)), TL.axisY, 1, 3);
  }
  ctx.font = `500 11px ${FONT_SANS}`;
  ctx.textBaseline = "top";
  for (let t = Math.ceil(startSeconds / stepSeconds - 1e-9) * stepSeconds; t <= endSeconds + 1e-6; t += stepSeconds) {
    const x = xAtFrame(t * run.fps, width, view);
    ctx.fillStyle = alpha(C.ink, 0.34);
    ctx.fillRect(Math.min(width - 1, Math.round(x)), TL.axisY, 1, 5);
    const text = formatTick(t, step);
    const tw = ctx.measureText(text).width;
    ctx.textAlign = "left";
    ctx.fillStyle = C.mist;
    ctx.fillText(text, Math.min(width - tw, Math.max(0, x - tw / 2)), TL.axisY + 9);
  }
}

/** The playhead (lime, with its marker on top) and the hover line. Drawn over the painted timeline every frame. */
export function paintPlayhead(ctx: CanvasRenderingContext2D, width: number, view: TimelineView, frame: number, hoverX: number | null): void {
  if (hoverX !== null && hoverX >= 0 && hoverX <= width) {
    ctx.fillStyle = alpha(C.ink, 0.4);
    ctx.fillRect(Math.round(hoverX), TL.rowY[0] - 3, 1, TL.rowY[2] + TL.rowH + 3 - (TL.rowY[0] - 3));
  }
  const x = xAtFrame(frame, width, view);
  if (x < -6 || x > width + 6) return;
  const cx = Math.min(width - 1, Math.max(1, x));
  ctx.fillStyle = "rgba(8, 10, 11, 0.55)";
  ctx.fillRect(cx - 2, TL.laneTop + 4, 4, TL.axisY - TL.laneTop - 2);
  ctx.fillStyle = C.lime;
  ctx.fillRect(cx - 1, TL.laneTop + 4, 2, TL.axisY - TL.laneTop - 2);
  ctx.beginPath();
  ctx.moveTo(cx - 5.5, TL.laneTop);
  ctx.lineTo(cx + 5.5, TL.laneTop);
  ctx.lineTo(cx, TL.laneTop + 8);
  ctx.closePath();
  ctx.fill();
}

/** A strip of the match's labels in one line, for small previews: each pixel takes the colour of whoever has most of it. */
export function paintMiniStrip(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  dpr: number,
  run: Run,
  from: number,
  to: number,
): void {
  const cols = Math.max(1, Math.round(w * dpr));
  const span = Math.max(1, to - from);
  const cover = [new Float32Array(cols), new Float32Array(cols), new Float32Array(cols), new Float32Array(cols)];
  const perFrame = cols / span;
  for (let r = lowerIndex(run.runStart, Math.max(0, Math.floor(from))); r < run.runStart.length; r++) {
    const s = run.runStart[r];
    if (s >= to) break;
    const e = r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames;
    addSpan(cover[run.runLabel[r]], (Math.max(s, from) - from) * perFrame, (Math.min(e, to) - from) * perFrame);
  }
  ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
  ctx.fillRect(x, y, w, h);
  const px1 = 1 / dpr;
  let start = 0;
  let current = -1;
  for (let c = 0; c <= cols; c++) {
    let best = -1;
    if (c < cols) {
      let top = 0.2;
      for (let l = 0; l < 4; l++) {
        if (cover[l][c] > top) {
          top = cover[l][c];
          best = l;
        }
      }
    } else best = -2;
    if (best !== current) {
      if (current >= 0) {
        ctx.fillStyle = current === X ? alpha(C.ink, 0.16) : LABEL_COLOR[current];
        ctx.fillRect(x + start * px1, y, (c - start) * px1, h);
      }
      start = c;
      current = best;
    }
  }
}
