import type { Moment } from "../model/moment";
import { Projector } from "./camera";
import { drawLabel } from "./labels";
import { C, FONT_SANS, alpha, trackLabel } from "./palette";

/**
 * The drawn match view: the simulated match (or any run's tracking) seen from a high camera on the halfway line,
 * the way a tactical feed looks. Everything here is drawn from tracking; no footage is involved.
 */

export interface MatchViewScene {
  pitch: { length: number; width: number };
  moment: Moment;
  /** Where play is, for the camera to look at: metres along the pitch and across it. */
  lookX: number;
  lookY: number;
  /** Goalkeeper slot for each team, or -1. */
  keepers: [number, number];
  homeSlots: number;
  /**
   * The same match a moment earlier (and how much earlier, in seconds). With it players stride as they run and
   * a fast ball leaves a short streak; without it everyone simply stands.
   */
  before?: Moment | null;
  beforeSeconds?: number;
}

export interface MatchViewOptions {
  labels: boolean;
  /** Zoom in on a point of the pitch (the ball tile). `metres` is how much of the pitch fits across the picture. */
  zoom?: { x: number; y: number; metres: number };
  /** Skip the far stand and the boards (small tiles). */
  plain?: boolean;
}

/* ───────────────────────── textures, drawn once ───────────────────────── */

const STAND_FROM = -92;
const STAND_TO = 92;
const CROWD_PX_PER_M = 9;
const BOARD_FROM = -66;
const BOARD_TO = 66;
const BOARD_PX_PER_M = 22;
const BOARD_H = 1.15;

let crowdTex: HTMLCanvasElement | null = null;
let boardTex: HTMLCanvasElement | null = null;
let wordmark: HTMLImageElement | null = null;

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildCrowd(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round((STAND_TO - STAND_FROM) * CROWD_PX_PER_M);
  canvas.height = 170;
  const g = canvas.getContext("2d")!;
  const { width: w, height: h } = canvas;
  g.fillStyle = "#0b0d10";
  g.fillRect(0, 0, w, h);
  const rand = seeded(41);
  const tones = ["#262b33", "#333944", "#444b57", "#565e6b", "#6c7480", "#8a919b", "#3b4658", "#4d3f47", "#2f3a33", "#b9bec6"];
  for (let y = 6; y < h - 2; y += 4.4) {
    const tier = y > h * 0.47 && y < h * 0.53;
    if (tier) continue;
    for (let x = (y * 7) % 3; x < w; x += 3.3) {
      const r = rand();
      if (r < 0.2) continue;
      g.globalAlpha = 0.55 + rand() * 0.45;
      g.fillStyle = r > 0.985 ? "#a7c93a" : tones[Math.floor(rand() * tones.length)];
      g.fillRect(x, y, 2.1, 2.7);
    }
  }
  g.globalAlpha = 1;
  // aisles and the break between tiers
  g.fillStyle = "#090b0d";
  for (let x = 60; x < w; x += 150) g.fillRect(x, 0, 5, h);
  g.fillRect(0, h * 0.47, w, h * 0.06);
  // the roof's shadow over the back rows, and a general dimming: this is a night game
  const shade = g.createLinearGradient(0, 0, 0, h);
  shade.addColorStop(0, "rgba(6, 8, 10, 0.92)");
  shade.addColorStop(0.45, "rgba(6, 8, 10, 0.5)");
  shade.addColorStop(1, "rgba(6, 8, 10, 0.3)");
  g.fillStyle = shade;
  g.fillRect(0, 0, w, h);
  return canvas;
}

function buildBoards(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round((BOARD_TO - BOARD_FROM) * BOARD_PX_PER_M);
  canvas.height = Math.round(BOARD_H * BOARD_PX_PER_M);
  const g = canvas.getContext("2d")!;
  const { width: w, height: h } = canvas;
  g.fillStyle = "#07090a";
  g.fillRect(0, 0, w, h);
  g.fillStyle = "rgba(255, 255, 255, 0.14)";
  g.fillRect(0, 0, w, 1);
  const chevrons = (x: number) => {
    g.fillStyle = C.lime;
    for (let i = 0; i < 4; i++) {
      const x0 = x + i * 13;
      g.beginPath();
      g.moveTo(x0, h - 5);
      g.lineTo(x0 + 7, 5);
      g.lineTo(x0 + 14, 5);
      g.lineTo(x0 + 7, h - 5);
      g.closePath();
      g.fill();
    }
  };
  const panel = 30 * BOARD_PX_PER_M;
  for (let x = 0; x < w; x += panel) {
    chevrons(x + 26);
    if (wordmark) {
      const wh = h - 12;
      const ww = (wordmark.naturalWidth / wordmark.naturalHeight) * wh;
      g.drawImage(wordmark, x + 118, 6, ww, wh);
    }
    chevrons(x + 318);
    g.fillStyle = "rgba(245, 245, 242, 0.9)";
    g.font = `600 ${Math.round(h * 0.44)}px ${FONT_SANS}`;
    g.textBaseline = "middle";
    g.textAlign = "left";
    const words = "TRAIN.  MEASURE.  ASCEND.";
    let at = x + 410;
    for (const ch of words) {
      g.fillText(ch, at, h / 2 + 0.5);
      at += g.measureText(ch).width + 2.2;
    }
  }
  return canvas;
}

/** Hands the renderer the Excensia wordmark for the pitch-side boards. Call once the image has loaded. */
export function setBoardWordmark(image: HTMLImageElement | null): void {
  wordmark = image;
  boardTex = null;
}

/** Rebuilds the board lettering (after the web fonts arrive). */
export function refreshBoards(): void {
  boardTex = null;
}

/* ───────────────────────── drawing ───────────────────────── */

const cam = new Projector();
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const CAM_BACK = 46; // metres behind the near touchline
const CAM_HEIGHT = 22;
const CAM_FOV = (18.4 * Math.PI) / 180;
/** Players are drawn a little larger than life, so they read at a distance. */
const BODY = 1.16;

const GRASS_A = "#3a8748";
const GRASS_B = "#327a40";
const APRON = "#2d6f3b";
const TRACK = "#141a18";
const LINE = "rgba(238, 244, 236, 0.9)";

interface Sprite {
  depth: number;
  team: 0 | 1 | 2; // 2 is the ball
  slot: number;
  x: number;
  y: number;
}
const sprites: Sprite[] = [];

function poly(ctx: CanvasRenderingContext2D, pts: number[]): boolean {
  ctx.beginPath();
  for (let i = 0; i < pts.length; i += 3) {
    if (!cam.project(pts[i], pts[i + 1], pts[i + 2])) return false;
    if (i === 0) ctx.moveTo(cam.x, cam.y);
    else ctx.lineTo(cam.x, cam.y);
  }
  ctx.closePath();
  return true;
}

function ground(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, fill: string): void {
  if (poly(ctx, [x0, y0, 0, x1, y0, 0, x1, y1, 0, x0, y1, 0])) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
}

/** A straight pitch line as a band on the grass, never thinner than a pixel. */
function line(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, width = 0.14): void {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy) || 1;
  const nx = (-dy / len) * (width / 2);
  const ny = (dx / len) * (width / 2);
  if (poly(ctx, [x0 + nx, y0 + ny, 0, x1 + nx, y1 + ny, 0, x1 - nx, y1 - ny, 0, x0 - nx, y0 - ny, 0])) {
    ctx.fill();
    ctx.stroke();
  }
}

/** An arc of a pitch marking, as a band. */
function arc(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, a0: number, a1: number, width = 0.14): void {
  const steps = Math.max(8, Math.ceil((Math.abs(a1 - a0) * r) / 0.9));
  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const a = a0 + ((a1 - a0) * i) / steps;
    if (!cam.project(cx + Math.cos(a) * (r + width / 2), cy + Math.sin(a) * (r + width / 2), 0)) return;
    if (i === 0) ctx.moveTo(cam.x, cam.y);
    else ctx.lineTo(cam.x, cam.y);
  }
  for (let i = steps; i >= 0; i--) {
    const a = a0 + ((a1 - a0) * i) / steps;
    if (!cam.project(cx + Math.cos(a) * (r - width / 2), cy + Math.sin(a) * (r - width / 2), 0)) return;
    ctx.lineTo(cam.x, cam.y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

/** Width of the picture being drawn, so slices that fall outside it can be skipped. */
let viewW = 0;

/**
 * A textured surface (a stand, a row of boards) drawn in slices so the perspective holds.
 * `at(u, v)` gives the world point at u along the surface (0 … 1) and v up it (0 its foot, 1 its top);
 * the texture's [texFrom, texTo] columns are spread along u.
 */
function surface(
  ctx: CanvasRenderingContext2D,
  tex: HTMLCanvasElement,
  texFrom: number,
  texTo: number,
  slices: number,
  at: (u: number, v: 0 | 1, out: [number, number, number]) => void,
): void {
  const point: [number, number, number] = [0, 0, 0];
  const texW = texTo - texFrom;
  for (let i = 0; i < slices; i++) {
    const u0 = i / slices;
    const u1 = (i + 1) / slices;
    at(u0, 1, point);
    if (!cam.project(point[0], point[1], point[2])) continue;
    const tlx = cam.x;
    const tly = cam.y;
    at(u1, 1, point);
    if (!cam.project(point[0], point[1], point[2])) continue;
    const trx = cam.x;
    const tryy = cam.y;
    at(u0, 0, point);
    if (!cam.project(point[0], point[1], point[2])) continue;
    const blx = cam.x;
    const bly = cam.y;
    if (Math.max(tlx, trx, blx) < -80 || Math.min(tlx, trx, blx) > viewW + 80) continue;
    const sx = texFrom + u0 * texW;
    const sw = texW / slices;
    ctx.save();
    // overlap each slice a touch so no seams show
    ctx.transform((trx - tlx) / sw, (tryy - tly) / sw, (blx - tlx) / tex.height, (bly - tly) / tex.height, tlx, tly);
    ctx.drawImage(tex, sx, 0, sw, tex.height, 0, 0, sw + 0.7, tex.height);
    ctx.restore();
  }
}

/** The ground and the stadium beyond the pitch: a strip of track, the boards and the stands on three sides. */
function drawSurroundings(ctx: CanvasRenderingContext2D, hl: number, hw: number, plain: boolean): void {
  ground(ctx, -hl - 40, hw + 3.4, hl + 40, hw + 9, TRACK);
  ground(ctx, -hl - 40, -hw - 16, hl + 40, -hw - 3.6, TRACK);
  ground(ctx, -hl - 40, -hw - 3.6, -hl - 4.6, hw + 3.4, TRACK);
  ground(ctx, hl + 4.6, -hw - 3.6, hl + 40, hw + 3.4, TRACK);
  if (plain) return;
  crowdTex ??= buildCrowd();
  boardTex ??= buildBoards();
  const crowd = crowdTex;
  const boards = boardTex;
  const wall = "#0a0c0e";

  // the stand along the far touchline
  surface(ctx, crowd, 0, crowd.width, 64, (u, v, out) => {
    out[0] = lerp(STAND_FROM, STAND_TO, u);
    out[1] = v ? hw + 27 : hw + 9;
    out[2] = v ? 12.5 : 1.3;
  });
  if (poly(ctx, [STAND_FROM, hw + 9, 0, STAND_TO, hw + 9, 0, STAND_TO, hw + 9, 1.3, STAND_FROM, hw + 9, 1.3])) {
    ctx.fillStyle = wall;
    ctx.fill();
  }
  // a stand behind each goal, and the boards in front of it
  const endLength = hw * 2 + 44;
  const endTex = Math.min(crowd.width, endLength * CROWD_PX_PER_M);
  const boardsTex = Math.min(boards.width, (hw * 2 + 10) * BOARD_PX_PER_M);
  for (const side of [-1, 1]) {
    surface(ctx, crowd, crowd.width - endTex, crowd.width, 40, (u, v, out) => {
      out[0] = side * (hl + (v ? 27.5 : 9.5));
      out[1] = lerp(-hw - 17, hw + 27, u);
      out[2] = v ? 12.5 : 1.3;
    });
    const ex = side * (hl + 9.5);
    if (poly(ctx, [ex, -hw - 17, 0, ex, hw + 27, 0, ex, hw + 27, 1.3, ex, -hw - 17, 1.3])) {
      ctx.fillStyle = wall;
      ctx.fill();
    }
    // lettering reads left to right for someone on the pitch
    surface(ctx, boards, 0, boardsTex, 36, (u, v, out) => {
      out[0] = side * (hl + 6.6);
      out[1] = side > 0 ? lerp(hw + 6.4, -hw - 3.6, u) : lerp(-hw - 3.6, hw + 6.4, u);
      out[2] = v ? BOARD_H : 0;
    });
  }
  surface(ctx, boards, 0, boards.width, 66, (u, v, out) => {
    out[0] = lerp(BOARD_FROM, BOARD_TO, u);
    out[1] = hw + 6.4;
    out[2] = v ? BOARD_H : 0;
  });
}

function drawPitch(ctx: CanvasRenderingContext2D, hl: number, hw: number): void {
  ground(ctx, -hl - 4.6, -hw - 3.6, hl + 4.6, hw + 3.4, APRON);
  // mown bands across the pitch
  const bands = 18;
  const step = (hl * 2) / bands;
  for (let i = 0; i < bands; i++) ground(ctx, -hl + i * step, -hw, -hl + (i + 1) * step, hw, i % 2 ? GRASS_B : GRASS_A);
  // and a fainter set along it, for the chequer a groundsman leaves
  const along = 8;
  const stepY = (hw * 2) / along;
  for (let i = 0; i < along; i += 2) ground(ctx, -hl, -hw + i * stepY, hl, -hw + (i + 1) * stepY, "rgba(255, 255, 255, 0.028)");

  ctx.fillStyle = LINE;
  ctx.strokeStyle = "rgba(238, 244, 236, 0.32)";
  ctx.lineWidth = 0.9;
  ctx.lineJoin = "round";
  // boundary and halfway line
  line(ctx, -hl, -hw, hl, -hw);
  line(ctx, -hl, hw, hl, hw);
  line(ctx, -hl, -hw, -hl, hw);
  line(ctx, hl, -hw, hl, hw);
  line(ctx, 0, -hw, 0, hw);
  arc(ctx, 0, 0, 9.15, 0, Math.PI * 2);
  arc(ctx, 0, 0, 0.22, 0, Math.PI * 2, 0.44);
  for (const side of [-1, 1]) {
    const gx = side * hl;
    // penalty area, goal area, spot and the arc outside the area
    line(ctx, gx, -20.16, gx - side * 16.5, -20.16);
    line(ctx, gx, 20.16, gx - side * 16.5, 20.16);
    line(ctx, gx - side * 16.5, -20.16, gx - side * 16.5, 20.16);
    line(ctx, gx, -9.16, gx - side * 5.5, -9.16);
    line(ctx, gx, 9.16, gx - side * 5.5, 9.16);
    line(ctx, gx - side * 5.5, -9.16, gx - side * 5.5, 9.16);
    arc(ctx, gx - side * 11, 0, 0.2, 0, Math.PI * 2, 0.4);
    const reach = Math.acos(5.5 / 9.15);
    if (side > 0) arc(ctx, gx - 11, 0, 9.15, Math.PI - reach, Math.PI + reach);
    else arc(ctx, gx + 11, 0, 9.15, -reach, reach);
    // corner arcs
    arc(ctx, gx, -hw, 1, side > 0 ? Math.PI / 2 : 0, side > 0 ? Math.PI : Math.PI / 2);
    arc(ctx, gx, hw, 1, side > 0 ? Math.PI : -Math.PI / 2, side > 0 ? Math.PI * 1.5 : 0);
  }
}

function drawGoal(ctx: CanvasRenderingContext2D, hl: number, side: number): void {
  const gx = side * hl;
  const bx = gx + side * 2;
  const post = 3.66;
  const bar = 2.44;
  // netting: back, roof and the two sides
  const nets = [
    [bx, -post, 0, bx, post, 0, bx, post, bar * 0.82, bx, -post, bar * 0.82],
    [gx, -post, bar, gx, post, bar, bx, post, bar * 0.82, bx, -post, bar * 0.82],
    [gx, -post, 0, bx, -post, 0, bx, -post, bar * 0.82, gx, -post, bar],
    [gx, post, 0, bx, post, 0, bx, post, bar * 0.82, gx, post, bar],
  ];
  ctx.lineWidth = 0.7;
  for (const net of nets) {
    if (!poly(ctx, net)) continue;
    ctx.fillStyle = "rgba(255, 255, 255, 0.07)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.stroke();
  }
  // the frame
  if (!cam.project(gx, -post, 0)) return;
  const ax = cam.x;
  const ay = cam.y;
  const s = cam.s;
  cam.project(gx, -post, bar);
  const bx1 = cam.x;
  const by1 = cam.y;
  cam.project(gx, post, bar);
  const cx = cam.x;
  const cy = cam.y;
  cam.project(gx, post, 0);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx1, by1);
  ctx.lineTo(cx, cy);
  ctx.lineTo(cam.x, cam.y);
  ctx.strokeStyle = "#f2f4f1";
  ctx.lineWidth = Math.max(1.4, 0.13 * s);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

const KITS = [
  { shirt: "#1d3164", shorts: "#0f1830", socks: "#16244a", trim: C.home, keeper: "#4a5668" },
  { shirt: "#eef1f4", shorts: "#cfd4da", socks: "#dfe3e8", trim: C.away, keeper: "#30353d" },
] as const;

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** How much a circle on the grass is squashed at a point, seen from the camera (0 edge-on … 1 from above). */
function squashAt(x: number, y: number): number {
  cam.project(x, y - 0.5, 0);
  const near = cam.y;
  cam.project(x, y + 0.5, 0);
  const far = cam.y;
  cam.project(x, y, 0);
  return Math.max(0.12, Math.min(1, (near - far) / cam.s));
}

function drawRing(ctx: CanvasRenderingContext2D, x: number, y: number, colour: string): void {
  const squash = squashAt(x, y);
  const s = cam.s;
  const rx = 0.95 * s;
  const ry = rx * squash;
  // a soft shadow where the player stands, then the ring
  ctx.beginPath();
  ctx.ellipse(cam.x + 0.22 * s, cam.y + 0.04 * s, 0.62 * s, 0.62 * s * squash * 0.8, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 0, 0, 0.24)";
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cam.x, cam.y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
  ctx.fill();
  ctx.lineWidth = Math.max(1.5, Math.min(2.8, 0.09 * s)) + 1.4;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
  ctx.stroke();
  ctx.lineWidth = Math.max(1.5, Math.min(2.8, 0.09 * s));
  ctx.strokeStyle = colour;
  ctx.stroke();
}

/**
 * A player: a small figure standing in its ring. It strides when it runs and stands when it stops; `vx`, `vy`
 * are its speed over the grass in metres a second.
 */
function drawPlayer(ctx: CanvasRenderingContext2D, x: number, y: number, team: 0 | 1, keeper: boolean, vx: number, vy: number): void {
  if (!cam.project(x, y, 0)) return;
  const s = cam.s * BODY;
  const fx = cam.x;
  const fy = cam.y;
  const kit = KITS[team];
  const shirt = keeper ? kit.keeper : kit.shirt;
  const legs = keeper ? "#191c21" : kit.socks;

  // how fast, and how much of the run goes across the picture (a run towards the camera shows no stride)
  const speed = Math.hypot(vx, vy);
  let across = 0;
  let phase = 0;
  if (speed > 0.5) {
    cam.project(x + vx / speed, y + vy / speed, 0);
    across = Math.max(-1, Math.min(1, (cam.x - fx) / cam.s));
    // a stride every 1.2 m, wherever the player is: the same frame always draws the same pose
    phase = 2.6 * ((x * vx + y * vy) / speed);
  }
  const go = Math.min(1, speed / 5.5);
  const swing = Math.sin(phase) * go;
  const reach = 0.3 * s * across;
  const lean = 0.09 * s * across * go;
  const stance = 0.085 * s * (1 - 0.55 * go);
  const hipY = fy - 0.86 * s;
  const shoulderY = fy - 1.46 * s;
  const hipX = fx + lean * 0.4;
  const shoulderX = fx + lean;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // legs: one goes forward as the other goes back, and the back foot lifts
  ctx.strokeStyle = legs;
  ctx.lineWidth = Math.max(1.3, 0.15 * s);
  ctx.beginPath();
  ctx.moveTo(hipX - 0.07 * s, hipY);
  ctx.lineTo(fx - stance + reach * swing, fy - Math.max(0, -swing) * 0.13 * s);
  ctx.moveTo(hipX + 0.07 * s, hipY);
  ctx.lineTo(fx + stance - reach * swing, fy - Math.max(0, swing) * 0.13 * s);
  ctx.stroke();
  // shorts
  roundRect(ctx, hipX - 0.19 * s, fy - 1.0 * s, 0.38 * s, 0.27 * s, 0.09 * s);
  ctx.fillStyle = keeper ? "#16191d" : kit.shorts;
  ctx.fill();
  // arms swing against the legs
  ctx.strokeStyle = shirt;
  ctx.lineWidth = Math.max(1.1, 0.11 * s);
  ctx.beginPath();
  ctx.moveTo(shoulderX - 0.22 * s, shoulderY + 0.06 * s);
  ctx.lineTo(shoulderX - 0.27 * s - reach * 0.55 * swing, fy - 0.98 * s - Math.abs(swing) * 0.07 * s);
  ctx.moveTo(shoulderX + 0.22 * s, shoulderY + 0.06 * s);
  ctx.lineTo(shoulderX + 0.27 * s + reach * 0.55 * swing, fy - 0.98 * s - Math.abs(swing) * 0.07 * s);
  ctx.stroke();
  // the shirt: wider at the shoulders than at the waist
  ctx.beginPath();
  ctx.moveTo(hipX - 0.17 * s, fy - 0.92 * s);
  ctx.lineTo(shoulderX - 0.2 * s, shoulderY);
  ctx.lineTo(shoulderX + 0.2 * s, shoulderY);
  ctx.lineTo(hipX + 0.17 * s, fy - 0.92 * s);
  ctx.closePath();
  ctx.fillStyle = shirt;
  ctx.fill();
  ctx.lineWidth = Math.max(1.2, 0.1 * s);
  ctx.strokeStyle = shirt;
  ctx.stroke();
  // a band of the team colour across the chest
  ctx.fillStyle = kit.trim;
  ctx.fillRect(shoulderX - 0.235 * s, fy - 1.33 * s, 0.47 * s, Math.max(1.2, 0.11 * s));
  // light from the floodlights down one side
  ctx.fillStyle = team === 0 ? "rgba(190, 215, 255, 0.2)" : "rgba(255, 255, 255, 0.22)";
  ctx.fillRect(shoulderX - 0.24 * s, shoulderY - 0.03 * s, 0.1 * s, 0.5 * s);
  // head
  ctx.beginPath();
  ctx.arc(shoulderX + lean * 0.35, fy - 1.7 * s, 0.145 * s, 0, Math.PI * 2);
  ctx.fillStyle = "#e7d8c6";
  ctx.fill();
  ctx.lineWidth = 0.6;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
  ctx.stroke();
}

/**
 * Paints the match view into a canvas area of w × h CSS pixels.
 * The context must already be scaled for the screen's pixel density.
 */
export function drawMatchView(ctx: CanvasRenderingContext2D, w: number, h: number, scene: MatchViewScene, options: MatchViewOptions): void {
  const hl = scene.pitch.length / 2;
  const hw = scene.pitch.width / 2;
  const { moment } = scene;

  const density = ctx.getTransform().a;
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, w, h);
  ctx.clip();
  ctx.fillStyle = "#090b0c";
  ctx.fillRect(0, 0, w, h);

  // The camera stands on the halfway line and turns to follow play. A picture taller than 16:9 sees more of the
  // near side rather than a narrower slice of pitch.
  const aspect = w / h;
  const fov = aspect >= 1.6 ? CAM_FOV : 2 * Math.atan(Math.tan(CAM_FOV / 2) * (1.6 / aspect) * 0.94);
  const scale = hl / 52.5;
  // it tilts a little with play: further up the picture when the game is on the far side, down when it is near
  const tilt = Math.max(-hw * 0.5, Math.min(hw * 0.3, scene.lookY * 0.7 - hw * 0.14));
  cam.setup(0, -(hw + CAM_BACK * scale), CAM_HEIGHT * scale, scene.lookX, tilt, 0, fov, w, h);
  viewW = w;

  if (options.zoom) {
    // magnify around a point of the pitch, like a zoomed crop of the same picture
    cam.project(options.zoom.x, options.zoom.y, 0.6);
    const k = w / (options.zoom.metres * cam.s);
    ctx.translate(w / 2, h / 2);
    ctx.scale(k, k);
    ctx.translate(-cam.x, -cam.y);
  }

  drawSurroundings(ctx, hl, hw, options.plain === true);
  drawPitch(ctx, hl, hw);
  drawGoal(ctx, hl, -1);
  drawGoal(ctx, hl, 1);

  if (moment.has) {
    sprites.length = 0;
    for (const team of [0, 1] as const) {
      const data = team === 0 ? moment.home : moment.away;
      for (let p = 0; p < data.length / 2; p++) {
        const x = data[p * 2];
        if (x !== x) continue;
        if (!cam.project(x, data[p * 2 + 1], 0)) continue;
        sprites.push({ depth: cam.depth, team, slot: p, x, y: data[p * 2 + 1] });
      }
    }
    for (const sprite of sprites) drawRing(ctx, sprite.x, sprite.y, sprite.team === 0 ? C.home : C.away);

    // where the ball is thought to be while it is out of sight
    if (!moment.ballSeen && moment.ballX === moment.ballX && moment.trust > 0.05) {
      const squash = squashAt(moment.ballX, moment.ballY);
      const r = Math.min(2.8, Math.max(1.4, moment.doubt)) * cam.s;
      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = alpha(C.lime, 0.85 * moment.trust);
      ctx.beginPath();
      ctx.ellipse(cam.x, cam.y, r, r * squash, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    if (moment.ballSeen && cam.project(moment.ballX, moment.ballY, 0)) {
      sprites.push({ depth: cam.depth - 0.4, team: 2, slot: 0, x: moment.ballX, y: moment.ballY });
      // a ball travelling fast leaves a short streak on the grass, as it does on film
      const earlier = scene.before;
      const before = earlier && earlier.has && earlier.ballSeen ? { x: earlier.ballX, y: earlier.ballY } : null;
      if (before) {
        const travelled = Math.hypot(moment.ballX - before.x, moment.ballY - before.y);
        if (travelled > 0.9 && travelled < 9) {
          const toX = cam.x;
          const toY = cam.y;
          const width = Math.max(1.6, 0.2 * cam.s);
          if (cam.project(before.x, before.y, 0)) {
            const streak = ctx.createLinearGradient(cam.x, cam.y, toX, toY);
            streak.addColorStop(0, "rgba(255, 255, 255, 0)");
            streak.addColorStop(1, `rgba(255, 255, 255, ${Math.min(0.42, 0.12 + travelled * 0.07)})`);
            ctx.beginPath();
            ctx.moveTo(cam.x, cam.y - width * 0.4);
            ctx.lineTo(toX, toY - width * 0.4);
            ctx.lineWidth = width;
            ctx.lineCap = "round";
            ctx.strokeStyle = streak;
            ctx.stroke();
          }
        }
      }
    }
    sprites.sort((a, b) => b.depth - a.depth);
    for (const sprite of sprites) {
      if (sprite.team === 2) {
        cam.project(sprite.x, sprite.y, 0);
        const r = Math.max(2.3, 0.15 * cam.s);
        ctx.beginPath();
        ctx.ellipse(cam.x + r * 0.4, cam.y + r * 0.1, r * 1.15, r * 0.5, 0, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cam.x, cam.y - r * 0.85, r, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.lineWidth = 0.9;
        ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
        ctx.stroke();
      } else {
        // speed over the grass, from where the player was a moment ago
        let vx = 0;
        let vy = 0;
        const was = scene.before ? (sprite.team === 0 ? scene.before.home : scene.before.away) : null;
        if (was && scene.beforeSeconds && was[sprite.slot * 2] === was[sprite.slot * 2]) {
          vx = (sprite.x - was[sprite.slot * 2]) / scene.beforeSeconds;
          vy = (sprite.y - was[sprite.slot * 2 + 1]) / scene.beforeSeconds;
          if (vx * vx + vy * vy > 144) vx = vy = 0; // a jump between tracks, not a run
        }
        drawPlayer(ctx, sprite.x, sprite.y, sprite.team, sprite.slot === scene.keepers[sprite.team], vx, vy);
      }
    }

    if (options.labels) {
      for (const sprite of sprites) {
        if (sprite.team === 2) continue;
        cam.project(sprite.x, sprite.y, 0);
        const text = trackLabel(sprite.team, sprite.slot, scene.keepers[sprite.team], scene.homeSlots);
        drawLabel(ctx, text, cam.x + 0.62 * cam.s + 4, cam.y - 1.02 * cam.s, density, {
          size: Math.max(10, Math.min(12.5, 0.4 * cam.s + 4.5)),
          colour: C.ink,
          halo: "rgba(8, 14, 10, 0.72)",
        });
      }
    }
  }
  ctx.restore();

  // the night: darker at the edges and towards the stand
  ctx.save();
  const shade = ctx.createRadialGradient(w / 2, h * 0.56, Math.min(w, h) * 0.35, w / 2, h * 0.56, Math.max(w, h) * 0.78);
  shade.addColorStop(0, "rgba(0, 0, 0, 0)");
  shade.addColorStop(1, "rgba(0, 0, 0, 0.32)");
  ctx.fillStyle = shade;
  ctx.fillRect(0, 0, w, h);
  const haze = ctx.createLinearGradient(0, 0, 0, h);
  haze.addColorStop(0, "rgba(9, 11, 12, 0.2)");
  haze.addColorStop(0.3, "rgba(9, 11, 12, 0)");
  haze.addColorStop(1, "rgba(0, 0, 0, 0.16)");
  ctx.fillStyle = haze;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}
