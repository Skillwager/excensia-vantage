import { emptyMoment, sampleMoment } from "./moment";
import { attackDir, shapeIndex } from "./shape";
import { spellsOf, type MatchStats } from "./stats";
import { formatClock, formatDuration, formatPercent } from "./time";
import { A, H, X, type Run } from "./types";

export type InsightVisual =
  | { kind: "split"; rows: { label: string; home: number; away: number }[] }
  | { kind: "thirds"; team: 0 | 1; counts: [number, number, number]; pick: number }
  | { kind: "strip"; from: number; to: number; markFrom: number; markTo: number }
  | { kind: "bands"; values: number[]; labels: string[]; pick: number };

/** One finding, written from the run, that points at a moment on the timeline. */
export interface Insight {
  id: string;
  /** A short label for the kind of finding. */
  kicker: string;
  headline: string;
  detail: string;
  /** The frame the finding points at. */
  frame: number;
  /** Words for the button that goes there. */
  linkLabel: string;
  visual: InsightVisual;
}

const THIRD_WORDS = ["their own third", "the middle third", "the final third"] as const;

/** Where the ball changed sides in open play, by third of the pitch, for the team that lost it. */
export function turnoverThirds(run: Run): { counts: [[number, number, number], [number, number, number]]; frames: [number[][], number[][]] } | null {
  const index = shapeIndex(run);
  if (!index || !run.tracking) return null;
  const counts: [[number, number, number], [number, number, number]] = [
    [0, 0, 0],
    [0, 0, 0],
  ];
  const frames: [number[][], number[][]] = [
    [[], [], []],
    [[], [], []],
  ];
  const moment = emptyMoment(run.tracking);
  const length = run.pitch.length;
  for (const change of run.changes) {
    let lost: 0 | 1;
    if (change.from === H && change.to === A) lost = 0;
    else if (change.from === A && change.to === H) lost = 1;
    else continue;
    sampleMoment(run.tracking, change.frame, run.fps, moment);
    if (!moment.has || moment.ballX !== moment.ballX) continue;
    const dir = attackDir(index, lost, change.frame);
    if (dir === 0) continue;
    const fromOwnGoal = dir > 0 ? moment.ballX + length / 2 : length / 2 - moment.ballX;
    const third = fromOwnGoal < length / 3 ? 0 : fromOwnGoal < (2 * length) / 3 ? 1 : 2;
    counts[lost][third]++;
    frames[lost][third].push(change.frame);
  }
  return { counts, frames };
}

/** Up to six findings, written from the run. Fewer when the run does not hold enough to say something. */
export function buildInsights(run: Run, stats: MatchStats): Insight[] {
  const out: Insight[] = [];
  const names = [run.home, run.away] as const;
  const fps = run.fps;
  const clock = (frame: number) => formatClock(frame / fps);
  if (stats.liveFrames === 0) return out;

  // 1. who had more of the ball
  {
    const home = stats.homeShare;
    const lead: 0 | 1 = home >= 0.5 ? 0 : 1;
    const share = lead === 0 ? home : 1 - home;
    const level = Math.abs(home - 0.5) < 0.015;
    const longest = (lead === 0 ? stats.home : stats.away).longest;
    out.push({
      id: "share",
      kicker: "Share of the ball",
      headline: level ? "The two sides shared the ball evenly." : `${names[lead]} had more of the ball: ${formatPercent(share)} of the time it was in play.`,
      detail: `The ball was in play for ${formatDuration(stats.liveFrames / fps)}. ${names[0]} had it for ${formatDuration(stats.frames[H] / fps)}, ${names[1]} for ${formatDuration(stats.frames[A] / fps)}.`,
      frame: longest ? longest.start : 0,
      linkLabel: longest ? `See ${names[lead]}'s longest spell` : "Go to the start",
      visual: { kind: "split", rows: [{ label: "Whole match", home: stats.frames[H], away: stats.frames[A] }] },
    });
  }

  // 2. before and after the break
  if (stats.periods.length >= 2) {
    const first = stats.periods[0];
    const last = stats.periods[stats.periods.length - 1];
    const before = first.counts[H] + first.counts[A] > 0 ? first.counts[H] / (first.counts[H] + first.counts[A]) : NaN;
    const after = last.counts[H] + last.counts[A] > 0 ? last.counts[H] / (last.counts[H] + last.counts[A]) : NaN;
    if (before === before && after === after) {
      const lead: 0 | 1 = after >= 0.5 ? 0 : 1;
      const now = lead === 0 ? after : 1 - after;
      const was = lead === 0 ? before : 1 - before;
      const swing = now - was;
      const headline =
        Math.abs(swing) < 0.03
          ? `Little changed after the break: ${names[lead]} kept ${formatPercent(now)} of the ball.`
          : swing > 0
            ? `${names[lead]} held the ball longer after the break: ${formatPercent(now)}, up from ${formatPercent(was)}.`
            : `${names[lead]} still had more of the ball after the break, but less of it: ${formatPercent(now)}, down from ${formatPercent(was)}.`;
      out.push({
        id: "break",
        kicker: "After the break",
        headline,
        detail: `${first.name}: ${names[0]} ${formatPercent(before)}, ${names[1]} ${formatPercent(1 - before)}. ${last.name}: ${names[0]} ${formatPercent(after)}, ${names[1]} ${formatPercent(1 - after)}.`,
        frame: last.start,
        linkLabel: `Go to the ${last.name.toLowerCase()}`,
        visual: {
          kind: "split",
          rows: [
            { label: first.name, home: first.counts[H], away: first.counts[A] },
            { label: last.name, home: last.counts[H], away: last.counts[A] },
          ],
        },
      });
    }
  }

  // 3. where the ball was lost
  const thirds = turnoverThirds(run);
  if (thirds && thirds.counts[0].concat(thirds.counts[1]).reduce((s, v) => s + v, 0) >= 6) {
    const totals = thirds.counts.map((c) => c[0] + c[1] + c[2]);
    const team: 0 | 1 = totals[1] > totals[0] ? 1 : 0;
    const mine = thirds.counts[team];
    let pick = 0;
    for (let i = 1; i < 3; i++) if (mine[i] > mine[pick]) pick = i;
    const otherTeam = team === 0 ? 1 : 0;
    const theirs = thirds.counts[otherTeam];
    out.push({
      id: "turnovers",
      kicker: "Where the ball was lost",
      headline: `${names[team]} lost the ball most often in ${THIRD_WORDS[pick]}: ${mine[pick]} of ${totals[team]} times in open play.`,
      detail: `${names[otherTeam]} lost it ${totals[otherTeam]} times: ${theirs[0]} in their own third, ${theirs[1]} in the middle and ${theirs[2]} in the final third.`,
      frame: Math.max(0, (thirds.frames[team][pick][0] ?? 0) - Math.round(3 * fps)),
      linkLabel: "Watch the first of them",
      visual: { kind: "thirds", team, counts: mine, pick },
    });
  } else if (stats.turnovers >= 6) {
    // no tracking to place them: say when they bunched instead
    const windowFrames = Math.round(5 * 60 * fps);
    let bestStart = 0;
    let best = 0;
    const turns = run.changes.filter((c) => (c.from === H && c.to === A) || (c.from === A && c.to === H)).map((c) => c.frame);
    for (let start = 0; start < run.frames; start += Math.round(60 * fps)) {
      const n = turns.filter((f) => f >= start && f < start + windowFrames).length;
      if (n > best) {
        best = n;
        bestStart = start;
      }
    }
    out.push({
      id: "turnovers",
      kicker: "When the ball changed sides",
      headline: `Possession changed hands most between ${clock(bestStart)} and ${clock(bestStart + windowFrames)}: ${best} times in five minutes.`,
      detail: `Across the match the ball went straight from one side to the other ${stats.turnovers} times.`,
      frame: bestStart,
      linkLabel: "Go to that passage",
      visual: { kind: "strip", from: bestStart, to: Math.min(run.frames, bestStart + windowFrames), markFrom: bestStart, markTo: Math.min(run.frames, bestStart + windowFrames) },
    });
  }

  // 4. the longest spell on the ball
  const longest = stats.longestSpells[0];
  if (longest) {
    const team = longest.label === H ? 0 : 1;
    const pad = Math.round(Math.max(20, longest.seconds * 0.6) * fps);
    out.push({
      id: "longest",
      kicker: "Longest spell",
      headline: `${names[team]}'s longest spell on the ball lasted ${formatDuration(longest.seconds)}, from ${clock(longest.start)}.`,
      detail: `A typical spell lasted ${formatDuration((stats.home.meanSeconds * stats.home.spells + stats.away.meanSeconds * stats.away.spells) / Math.max(1, stats.home.spells + stats.away.spells))}.`,
      frame: longest.start,
      linkLabel: "Watch it",
      visual: { kind: "strip", from: Math.max(0, longest.start - pad), to: Math.min(run.frames, longest.end + pad), markFrom: longest.start, markTo: longest.end + 1 },
    });
  }

  // 5. the longest stoppage
  const stop = stats.dead.longest;
  if (stop) {
    const pad = Math.round(Math.max(20, stop.seconds * 0.6) * fps);
    const change = run.changes.find((c) => c.frame === stop.start);
    out.push({
      id: "stoppage",
      kicker: "Longest stoppage",
      headline: `The longest stoppage lasted ${formatDuration(stop.seconds)}, from ${clock(stop.start)}.`,
      detail: `${change?.why ? `${change.why} ` : ""}The ball was dead ${stats.dead.spells} times, for ${formatDuration(stats.dead.meanSeconds)} on average.`,
      frame: Math.max(0, stop.start - Math.round(3 * fps)),
      linkLabel: "Go to the stoppage",
      visual: { kind: "strip", from: Math.max(0, stop.start - pad), to: Math.min(run.frames, stop.end + pad), markFrom: stop.start, markTo: stop.end + 1 },
    });
  }

  // 6. where the model was least sure
  {
    let pick: (typeof run.segments)[number] | null = null;
    for (const s of run.segments) {
      if (s.label === X || s.confidence === null) continue;
      if ((s.end - s.start + 1) / fps < 4) continue;
      if (!pick || s.confidence < pick.confidence!) pick = s;
    }
    if (pick && pick.confidence! < 0.75) {
      const who = pick.label === H ? names[0] : pick.label === A ? names[1] : "dead-ball";
      const seconds = (pick.end - pick.start + 1) / fps;
      const pad = Math.round(Math.max(20, seconds * 0.6) * fps);
      const hiddenShare = pick.ballSeen !== null ? 1 - pick.ballSeen : null;
      out.push({
        id: "unsure",
        kicker: "Least certain call",
        headline: `The model was least sure at ${clock(pick.start)}: ${formatPercent(pick.confidence!)} confident of a ${formatDuration(seconds)} ${who} spell.`,
        detail:
          hiddenShare !== null
            ? `The ball was out of sight for ${formatPercent(hiddenShare)} of that spell, so the call rests on team shape and timing. Worth a look.`
            : "Worth a look before you rely on it.",
        frame: pick.start,
        linkLabel: "Check the evidence",
        visual: { kind: "strip", from: Math.max(0, pick.start - pad), to: Math.min(run.frames, pick.end + pad), markFrom: pick.start, markTo: pick.end + 1 },
      });
    }
  }

  // 7. the most stop-start passage, when there is room for it
  if (out.length < 6 && stats.bands.length >= 3) {
    const shares = stats.bands.map((b) => {
      const answered = b.counts[0] + b.counts[1] + b.counts[2];
      return answered > 0 ? (b.counts[0] + b.counts[1]) / answered : NaN;
    });
    let pick = -1;
    for (let i = 0; i < shares.length; i++) if (shares[i] === shares[i] && (pick < 0 || shares[i] < shares[pick])) pick = i;
    if (pick >= 0) {
      const band = stats.bands[pick];
      out.push({
        id: "tempo",
        kicker: "Stop-start",
        headline: `The game was most stop-start between ${Math.round(band.fromMinute)}' and ${Math.round(band.toMinute)}': the ball was in play for ${formatPercent(shares[pick])} of it.`,
        detail: `Across the match the ball was in play ${formatPercent(stats.inPlayShare)} of the time.`,
        frame: band.start,
        linkLabel: "Go to that passage",
        visual: { kind: "bands", values: shares, labels: stats.bands.map((b) => `${Math.round(b.fromMinute)}'`), pick },
      });
    }
  }

  return out.slice(0, 6);
}

/** Every spell, for look-ups by the cards. */
export { spellsOf };
