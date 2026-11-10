import { lowerIndex, nextChange, segmentAt } from "./derive";
import type { Moment } from "./moment";
import type { ShapePair } from "./shape";
import { formatClock, formatDuration } from "./time";
import { A, D, H, X, type Change, type Label, type Run } from "./types";

export type EvidenceKind = "ball" | "shape" | "timing";

export interface Explanation {
  /** The call in words: a team name, "Dead ball" or "No answer". */
  estimate: string;
  label: Label;
  /** One plain sentence (two at most) for each kind of evidence. */
  ball: string;
  shape: string;
  timing: string;
}

export function labelWords(run: Pick<Run, "home" | "away">, label: Label): string {
  return label === H ? run.home : label === A ? run.away : label === D ? "Dead ball" : "No answer";
}

const metres = (v: number) => `${Math.round(v)} m`;

function inPeriod(run: Pick<Run, "periods">, frame: number): boolean {
  if (!run.periods.length) return true;
  return run.periods.some((p) => frame >= p.start && frame <= p.end);
}

/** The change that opened the unbroken run of the label at this frame, or null at the very start. */
function changeInto(run: Pick<Run, "changes" | "changeFrame">, frame: number): Change | null {
  if (!run.changes.length || run.changeFrame[0] > frame) return null;
  return run.changes[lowerIndex(run.changeFrame, frame)];
}

/**
 * Says, in plain words, why the model made its call at this frame.
 * `earlier` is the team shapes a few seconds before, used to say whether a team is moving up or dropping back.
 */
export function explain(run: Run, frame: number, moment: Moment, shapes: ShapePair, earlier: ShapePair | null): Explanation {
  const f = Math.floor(frame);
  const segment = segmentAt(run, f);
  const label = segment.label;
  const team = label === H ? run.home : label === A ? run.away : null;
  const foe = label === H ? run.away : label === A ? run.home : null;
  const mine = label === H ? shapes.home : label === A ? shapes.away : null;
  const theirs = label === H ? shapes.away : label === A ? shapes.home : null;
  const mineBefore = earlier ? (label === H ? earlier.home : label === A ? earlier.away : null) : null;
  const moved = mine && mineBefore ? mine.centreHeight - mineBefore.centreHeight : 0;
  const hasTracking = run.tracking !== null && moment.has;

  /* ── the ball ── */
  let ball: string;
  if (!hasTracking) {
    ball =
      label === X
        ? "There is no answer for this moment, and this run has no tracking to look at."
        : `The call here is ${team ?? "dead ball"}. This run has no tracking, so the ball itself cannot be shown.`;
  } else if (label === X) {
    ball = inPeriod(run, f)
      ? "No clear answer here: the ball is loose and neither side has it under control."
      : "This is outside the periods of play, so there is no call to make.";
  } else if (label === D) {
    ball = moment.ballSeen
      ? "The ball is in view but not in play. Both teams are waiting for the restart."
      : "The ball is out of play and out of sight. Play has stopped.";
  } else if (moment.ballSeen) {
    ball = `The ball is in clear view and ${team} have it under control.`;
  } else {
    const trend = moved > 1.5 ? "their shape is still advancing" : moved < -1.5 ? "they are dropping back with it" : "their shape is holding";
    const gone = moment.hiddenFor >= 3 && Number.isFinite(moment.hiddenFor) ? ` It has been out of sight for ${formatDuration(moment.hiddenFor)}.` : "";
    ball = `The ball is hidden behind players. ${team} keep it: ${trend} and nobody from ${foe} has controlled it.${gone}`;
  }

  /* ── team shape ── */
  let shape: string;
  if (!hasTracking || !shapes.home || !shapes.away) {
    shape = "This run has no player tracking for this moment, so there is no team shape to show.";
  } else if (label === D) {
    shape = `Play has stopped and both teams are setting up for the restart. ${run.home} are ${metres(shapes.home.width)} wide, ${run.away} ${metres(shapes.away.width)}.`;
  } else if (label === X || !mine || !theirs) {
    shape = `Neither side is set. ${run.home} are ${metres(shapes.home.width)} wide and ${run.away} ${metres(shapes.away.width)}, both drawn towards the ball.`;
  } else {
    const trend = moved > 1.5 ? " and moving up the pitch" : moved < -1.5 ? " and dropping back" : "";
    const compare =
      theirs.width < mine.width - 3
        ? `${foe} are tighter, ${metres(theirs.width)} wide, as a side without the ball would be.`
        : theirs.width > mine.width + 3
          ? `${foe} are wider, ${metres(theirs.width)}, which is unusual for the side without the ball.`
          : `${foe} are about as wide, ${metres(theirs.width)}.`;
    shape = `${team} are spread ${metres(mine.width)} wide with their back line ${metres(mine.lineHeight)} from their own goal${trend}. ${compare}`;
  }

  /* ── timing ── */
  // the whole unbroken run of this label, not just the segment
  const before = changeInto(run, f);
  const after = nextChange(run, f);
  const runStart = before ? before.frame : 0;
  const runEnd = after ? after.frame : run.frames;
  const elapsed = (f - runStart) / run.fps;
  const total = (runEnd - runStart) / run.fps;
  const who = team ? `${team} are` : label === D ? "The ball has been dead for" : "There has been no answer for";
  let timing =
    team !== null
      ? `${who} ${formatDuration(elapsed)} into a spell of ${formatDuration(total)}.`
      : `${who} ${formatDuration(elapsed)} of ${formatDuration(total)}.`;
  if (before) timing += before.why ? ` It began at ${formatClock(before.frame / run.fps)}: ${before.why}` : ` It began at ${formatClock(before.frame / run.fps)}.`;
  if (after) timing += ` The next change is at ${formatClock(after.frame / run.fps)}.`;

  return { estimate: labelWords(run, label), label, ball, shape, timing };
}
