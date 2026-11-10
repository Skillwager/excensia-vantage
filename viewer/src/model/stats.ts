import { countLabels } from "./derive";
import { A, D, H, type Label, type Run } from "./types";

/** One unbroken stretch of a label. */
export interface SpellRef {
  label: Label;
  /** First frame. */
  start: number;
  /** Last frame (inclusive). */
  end: number;
  seconds: number;
}

export interface TeamPossession {
  /** Number of spells on the ball. */
  spells: number;
  frames: number;
  meanSeconds: number;
  medianSeconds: number;
  longest: SpellRef | null;
}

export interface Band {
  fromMinute: number;
  toMinute: number;
  /** First frame, and the frame after the last. */
  start: number;
  end: number;
  /** Frames per label: [home, away, dead ball, no answer]. */
  counts: [number, number, number, number];
}

export interface PeriodStats {
  name: string;
  start: number;
  end: number;
  counts: [number, number, number, number];
}

export interface MatchStats {
  /** Frames per label: [home, away, dead ball, no answer]. */
  frames: [number, number, number, number];
  /** Frames with the ball in play: home + away. */
  liveFrames: number;
  /** Home's share of the time the ball was in play, 0..1 (NaN when it never was). */
  homeShare: number;
  /** Share of answered time with the ball in play, 0..1. */
  inPlayShare: number;
  home: TeamPossession;
  away: TeamPossession;
  dead: { spells: number; frames: number; meanSeconds: number; longest: SpellRef | null };
  /** The longest spells on the ball, either team, longest first. */
  longestSpells: SpellRef[];
  bands: Band[];
  periods: PeriodStats[];
  /** Times the ball went straight from one team to the other. */
  turnovers: number;
  /** Every change of label. */
  changes: number;
}

function median(sorted: number[]): number {
  if (!sorted.length) return 0;
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Every unbroken stretch of each label, in match order. */
export function spellsOf(run: Pick<Run, "runStart" | "runLabel" | "frames" | "fps">): SpellRef[] {
  const out: SpellRef[] = [];
  for (let r = 0; r < run.runStart.length; r++) {
    const start = run.runStart[r];
    const end = (r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames) - 1;
    out.push({ label: run.runLabel[r] as Label, start, end, seconds: (end - start + 1) / run.fps });
  }
  return out;
}

function teamPossession(spells: SpellRef[], label: Label): TeamPossession {
  const mine = spells.filter((s) => s.label === label);
  const lengths = mine.map((s) => s.seconds).sort((a, b) => a - b);
  const total = lengths.reduce((sum, v) => sum + v, 0);
  let longest: SpellRef | null = null;
  let frames = 0;
  for (const s of mine) {
    frames += s.end - s.start + 1;
    if (!longest || s.seconds > longest.seconds) longest = s;
  }
  return { spells: mine.length, frames, meanSeconds: mine.length ? total / mine.length : 0, medianSeconds: median(lengths), longest };
}

/** The match in numbers: who had the ball, for how long, and when. */
export function matchStats(run: Run, bandMinutes = 15): MatchStats {
  const spells = spellsOf(run);
  const frames = countLabels(run, 0, run.frames);
  const liveFrames = frames[H] + frames[A];
  const answered = liveFrames + frames[D];

  const deadSpells = spells.filter((s) => s.label === D);
  let longestDead: SpellRef | null = null;
  for (const s of deadSpells) if (!longestDead || s.seconds > longestDead.seconds) longestDead = s;

  const bandFrames = Math.max(1, Math.round(bandMinutes * 60 * run.fps));
  const bands: Band[] = [];
  for (let start = 0, i = 0; start < run.frames; start += bandFrames, i++) {
    const end = Math.min(run.frames, start + bandFrames);
    // a short sliver at the very end belongs to the band before it
    if (bands.length && end - start < bandFrames * 0.4) {
      const last = bands[bands.length - 1];
      last.end = end;
      last.toMinute = end / run.fps / 60;
      last.counts = countLabels(run, last.start, end);
      break;
    }
    bands.push({ fromMinute: i * bandMinutes, toMinute: end / run.fps / 60, start, end, counts: countLabels(run, start, end) });
  }

  let turnovers = 0;
  for (const c of run.changes) if ((c.from === H && c.to === A) || (c.from === A && c.to === H)) turnovers++;

  return {
    frames,
    liveFrames,
    homeShare: liveFrames > 0 ? frames[H] / liveFrames : NaN,
    inPlayShare: answered > 0 ? liveFrames / answered : NaN,
    home: teamPossession(spells, H),
    away: teamPossession(spells, A),
    dead: {
      spells: deadSpells.length,
      frames: frames[D],
      meanSeconds: deadSpells.length ? frames[D] / run.fps / deadSpells.length : 0,
      longest: longestDead,
    },
    longestSpells: spells
      .filter((s) => s.label === H || s.label === A)
      .sort((a, b) => b.seconds - a.seconds || a.start - b.start)
      .slice(0, 6),
    bands,
    periods: run.periods.map((p) => ({ name: p.name, start: p.start, end: p.end, counts: countLabels(run, p.start, p.end + 1) })),
    turnovers,
    changes: run.changes.length,
  };
}
