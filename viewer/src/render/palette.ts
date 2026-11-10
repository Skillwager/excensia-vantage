/**
 * The Excensia Night palette for canvases (a canvas cannot read CSS custom properties).
 * These mirror src/styles/tokens.css; keep the two the same.
 */
export const C = {
  pitch: "#0f1113",
  pitchMid: "#1e2125",
  pitchLift: "#26292e",
  lime: "#c6ff00",
  ink: "#f5f5f2",
  mist: "#a9afab",
  dim: "#7a817c",
  home: "#35c8f5",
  away: "#ff3db4",
  dead: "#8b9299",
  ember: "#ffb35c",
  coral: "#d97868",
  line: "rgba(255, 255, 255, 0.08)",
  lineStrong: "rgba(255, 255, 255, 0.14)",
} as const;

/** Colour for each label: home, away, dead ball, no answer. */
export const LABEL_COLOR = [C.home, C.away, C.dead, C.dim] as const;

export const FONT_SANS = '"Inter", ui-sans-serif, system-ui, sans-serif';
export const FONT_MONO = '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace';
export const FONT_DISPLAY = '"Exo 2", "Barlow Condensed", ui-sans-serif, sans-serif';

/** "#rrggbb" with an alpha, as rgba(). */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Track label for a player: the goalkeeper is "GK", everyone else "T" and a number that is unique across both teams. */
export function trackLabel(team: 0 | 1, slot: number, keeper: number, homeSlots: number): string {
  if (slot === keeper) return "GK";
  return `T${(team === 0 ? 0 : homeSlots) + slot + 1}`;
}
