/** Time and frame formatting. Every clock in the viewer is footage time: frame / fps. */

export function frameToSeconds(frame: number, fps: number): number {
  return frame / fps;
}

/** The frame showing at a time in seconds, clamped to the match. */
export function secondsToFrame(seconds: number, fps: number, frames: number): number {
  const frame = Math.floor(seconds * fps + 1e-6);
  if (!(frame > 0)) return 0;
  return frame > frames - 1 ? frames - 1 : frame;
}

function pad2(value: number): string {
  return value < 10 ? `0${value}` : `${value}`;
}

/** "72:14". Minutes run past 59; seconds are floored, never rounded up. */
export function formatClock(seconds: number): string {
  const whole = Math.floor(Math.max(0, Number.isFinite(seconds) ? seconds : 0) + 1e-6);
  return `${pad2(Math.floor(whole / 60))}:${pad2(whole % 60)}`;
}

/** The clock for a frame: "72:14". */
export function formatFrameClock(frame: number, fps: number): string {
  return formatClock(Math.floor(Math.max(0, frame)) / fps);
}

/** "72:14.36", for stepping one frame at a time. */
export function formatClockPrecise(frame: number, fps: number): string {
  const f = Math.floor(Math.max(0, frame));
  const wholeFps = Math.max(1, Math.round(fps));
  const hundredths = Math.floor(((f % wholeFps) / wholeFps) * 100 + 1e-6);
  return `${formatClock(f / fps)}.${pad2(hundredths)}`;
}

/** The whole-match clock, e.g. "95:00" for 142,500 frames at 25 fps. */
export function formatTotalClock(frames: number, fps: number): string {
  return formatClock(frames / fps);
}

/** A length of time in plain words: "9 s", "1 min 12 s", "3 min". */
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  if (s < 9.95) {
    const tenths = Math.round(s * 10) / 10;
    return Number.isInteger(tenths) ? `${tenths} s` : `${tenths.toFixed(1)} s`;
  }
  const whole = Math.round(s);
  if (whole < 60) return `${whole} s`;
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} s`;
}

/** A length of time as a clock, "1:12", for tables. */
export function formatSpan(seconds: number): string {
  const whole = Math.round(Math.max(0, Number.isFinite(seconds) ? seconds : 0));
  return `${Math.floor(whole / 60)}:${pad2(whole % 60)}`;
}

/** "15'" */
export function formatMinuteMark(minutes: number): string {
  return `${Math.round(minutes)}'`;
}

/** "142,500" */
export function formatCount(value: number): string {
  return Math.round(value).toLocaleString("en-GB");
}

/** A share as a whole percentage: 0.531 → "53%". */
export function formatPercent(share: number): string {
  if (!Number.isFinite(share)) return "–";
  return `${Math.round(share * 100)}%`;
}

/**
 * A pair of shares as whole percentages that add up to 100, so a split bar never reads 53% / 48%.
 */
export function formatSplit(a: number, b: number): [string, string] {
  const total = a + b;
  if (!(total > 0)) return ["–", "–"];
  const left = Math.round((a / total) * 100);
  return [`${left}%`, `${100 - left}%`];
}

/**
 * Several parts of a whole as whole percentages that add up to exactly 100 (largest remainders get the spare
 * points), so a row of shares never reads 31% + 27% + 41% + 2% = 101%.
 */
export function formatShares(parts: readonly number[]): string[] {
  const total = parts.reduce((sum, v) => sum + (v > 0 ? v : 0), 0);
  if (!(total > 0)) return parts.map(() => "–");
  const exact = parts.map((v) => ((v > 0 ? v : 0) / total) * 100);
  const whole = exact.map(Math.floor);
  let spare = 100 - whole.reduce((sum, v) => sum + v, 0);
  const order = exact.map((_, i) => i).sort((a, b) => exact[b] - whole[b] - (exact[a] - whole[a]) || a - b);
  for (const i of order) {
    if (spare <= 0) break;
    whole[i]++;
    spare--;
  }
  return whole.map((v) => `${v}%`);
}

/** Tick spacing in minutes for a timeline showing this many seconds across this many pixels. */
export function tickMinutes(visibleSeconds: number, widthPx: number): number {
  const steps = [0.25, 0.5, 1, 2, 5, 15, 30];
  const minutesPerPx = visibleSeconds / 60 / Math.max(1, widthPx);
  for (const step of steps) if (step / minutesPerPx >= 64) return step;
  return 30;
}

/** Tick label for a time in seconds: "15'" on whole minutes, "72:30" between them. */
export function formatTick(seconds: number, stepMinutes: number): string {
  if (stepMinutes >= 1) return `${Math.round(seconds / 60)}'`;
  return formatClock(seconds);
}
