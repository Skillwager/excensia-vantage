import { FONT_SANS } from "./palette";

/**
 * Small text with a dark halo, for labels on the pitch. Stroking text every frame is the most expensive thing a
 * canvas of labels does, so each label is drawn once into a little sprite and stamped from then on.
 */

interface Sprite {
  canvas: HTMLCanvasElement;
  /** Size in CSS pixels. */
  w: number;
  h: number;
  /** Padding around the text, CSS pixels. */
  pad: number;
}

const sprites = new Map<string, Sprite>();
let epoch = 0;

/** Forgets every sprite (the web fonts have arrived, so they must be drawn again). */
export function resetLabelSprites(): void {
  sprites.clear();
  epoch++;
}

function sprite(text: string, size: number, colour: string, halo: string, scale: number, tracking: number): Sprite {
  const key = `${epoch}|${text}|${size}|${colour}|${halo}|${scale}|${tracking}`;
  let hit = sprites.get(key);
  if (hit) return hit;
  if (sprites.size > 900) sprites.clear();
  const canvas = document.createElement("canvas");
  const g = canvas.getContext("2d")!;
  const font = `600 ${size}px ${FONT_SANS}`;
  g.font = font;
  const letters = tracking > 0 ? [...text] : [text];
  const widths = letters.map((ch) => g.measureText(ch).width + tracking);
  const textW = widths.reduce((sum, v) => sum + v, 0) - tracking;
  const pad = 3;
  const w = Math.ceil(textW + pad * 2);
  const h = Math.ceil(size * 1.3 + pad * 2);
  canvas.width = Math.max(1, Math.ceil(w * scale));
  canvas.height = Math.max(1, Math.ceil(h * scale));
  g.setTransform(scale, 0, 0, scale, 0, 0);
  g.font = font;
  g.textBaseline = "middle";
  g.textAlign = "left";
  g.lineJoin = "round";
  g.lineWidth = 3;
  g.strokeStyle = halo;
  g.fillStyle = colour;
  let x = pad;
  for (let i = 0; i < letters.length; i++) {
    g.strokeText(letters[i], x, h / 2 + 0.5);
    x += widths[i];
  }
  x = pad;
  for (let i = 0; i < letters.length; i++) {
    g.fillText(letters[i], x, h / 2 + 0.5);
    x += widths[i];
  }
  hit = { canvas, w, h, pad };
  sprites.set(key, hit);
  return hit;
}

export interface LabelStyle {
  /** Font size in CSS pixels; rounded to the nearest half pixel. */
  size: number;
  colour: string;
  halo: string;
  align?: "left" | "right" | "center";
  /** Extra space between letters, CSS pixels. */
  tracking?: number;
}

/**
 * Stamps a label with its text's left (or right, or centre) at x and its middle at y.
 * `scale` is the canvas's pixel density, so the sprite is as sharp as the canvas.
 */
export function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, scale: number, style: LabelStyle): number {
  const size = Math.round(style.size * 2) / 2;
  const s = sprite(text, size, style.colour, style.halo, Math.round(scale * 4) / 4, style.tracking ?? 0);
  const textW = s.w - s.pad * 2;
  const left = style.align === "right" ? x - textW : style.align === "center" ? x - textW / 2 : x;
  ctx.drawImage(s.canvas, left - s.pad, y - s.h / 2, s.w, s.h);
  return textW;
}

/** The width a label's text takes, CSS pixels. */
export function labelWidth(text: string, scale: number, style: LabelStyle): number {
  const s = sprite(text, Math.round(style.size * 2) / 2, style.colour, style.halo, Math.round(scale * 4) / 4, style.tracking ?? 0);
  return s.w - s.pad * 2;
}
