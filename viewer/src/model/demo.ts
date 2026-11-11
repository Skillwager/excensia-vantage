/**
 * The simulated match.
 *
 * A seeded generator writes a full run in the `excensia.vantage.run/1` shape: about 95 minutes at 25 fps,
 * possession spells with a long tail (mean about 14 s), about 40% dead-ball time, and tracking in which the
 * ball goes unseen on about 40% of samples. Nothing here is real match data.
 *
 * It works in two passes:
 *   1. a script of phases at frame level (who has the ball, when it goes dead and why);
 *   2. a small football simulation at 5 samples a second that acts the script out: two teams holding their
 *      shape, advancing with the ball, tightening without it, and setting up for every restart.
 */
import { ballSeenShare, buildRun, toRunFile, type RunParts } from "./loader";
import { Rng } from "./rng";
import type { DemoScene, LabelCode, LabelRun, Run, RunFile, RunFileSpell, RunFileTransition, Tracking } from "./types";

export const DEMO_SEED = 7214;
export const DEMO_FPS = 25;
export const DEMO_EVERY = 5;
export const DEMO_FRAMES = 142_500;

const FPS = DEMO_FPS;
const EVERY = DEMO_EVERY;
const DT = EVERY / FPS;
const H1_END = 70_500; // first frame after the first half
const H2_START = 72_000;
const LEAD_IN = 120;
const HL = 52.5;
const HW = 34;

type Team = 0 | 1;
type EndKind = "tackle" | "intercept" | "throw" | "goalkick" | "corner" | "foul" | "offside" | "goal" | "whistle" | "loose" | "continue";
type Restart = "throw" | "goalkick" | "corner" | "freekick" | "kickoff";
type Mode = "carried" | "pass" | "exit" | "out" | "placed" | "still" | "loose";

interface LivePhase {
  kind: "live";
  team: Team;
  start: number;
  /** Exclusive. */
  end: number;
  endKind: EndKind;
  /** Frames in which the ball is certainly hidden (a scripted moment). */
  hide?: [number, number];
  unsure?: boolean;
}
interface DeadPhase {
  kind: "dead";
  start: number;
  end: number;
  restartTeam: Team;
  planned: EndKind;
}
interface NonePhase {
  kind: "none";
  start: number;
  end: number;
  reason: "pre" | "halftime" | "loose";
  nextTeam: Team;
}
type Phase = LivePhase | DeadPhase | NonePhase;

const sec = (s: number) => Math.round(s * FPS);
const other = (t: Team): Team => (t === 0 ? 1 : 0);
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const sign = (v: number) => (v < 0 ? -1 : 1);
const hyp = (x: number, y: number) => Math.sqrt(x * x + y * y);

/** Scripted moments, so the match has a story: a goal each way and the hidden-ball passage at 72:14. */
interface Anchor {
  start: number;
  end: number;
  team: Team;
  endKind: EndKind;
  hide?: [number, number];
  unsure?: boolean;
  /** The spell must begin at `start`, not carry on from the same side's possession before it. */
  fresh?: boolean;
}
const ANCHORS: Anchor[] = [
  { start: sec(31 * 60 + 40 - 19), end: sec(31 * 60 + 40), team: 0, endKind: "goal" },
  { start: sec(72 * 60 + 1), end: sec(72 * 60 + 33), team: 0, endKind: "tackle", hide: [sec(72 * 60 + 11.5), sec(72 * 60 + 33)], unsure: true, fresh: true },
  { start: sec(84 * 60 + 10 - 21), end: sec(84 * 60 + 10), team: 1, endKind: "goal" },
];

/* ───────────────────────── pass 1: the script ───────────────────────── */

function writeScript(rng: Rng): Phase[] {
  const phases: Phase[] = [];
  let anchorAt = 0;
  const MIN_LIVE = sec(2.5);
  const MIN_DEAD = sec(6);

  const possession = (team: Team, half: 0 | 1) => {
    // Home see more of the ball before the break, Away after it.
    const lean = half === 0 ? (team === 0 ? 1.1 : 0.92) : team === 0 ? 0.84 : 1.2;
    return sec(rng.logNormal(9.7 * lean, 0.9, 1.2, 150));
  };
  const turnover = (): EndKind => (rng.chance(0.55) ? "tackle" : "intercept");
  const pickEnd = (length: number): EndKind => {
    const u = rng.next();
    if (u < 0.425) {
      const kind = (["throw", "foul", "goalkick", "corner", "offside"] as const)[rng.pick([0.4, 0.25, 0.15, 0.14, 0.06])];
      // an ending by a goal line needs a possession long enough to get there
      if (length < sec(6) && kind !== "throw" && kind !== "foul") return rng.chance(0.62) ? "throw" : "foul";
      return kind;
    }
    if (u < 0.45) return "loose";
    return turnover();
  };
  const deadFrames = (kind: EndKind): number => {
    switch (kind) {
      case "throw":
        return sec(rng.logNormal(12.5, 0.4, 5, 32));
      case "goalkick":
        return sec(rng.logNormal(25, 0.3, 14, 45));
      case "corner":
        return sec(rng.logNormal(30, 0.3, 18, 55));
      case "offside":
        return sec(rng.logNormal(20, 0.3, 12, 35));
      case "goal":
        return sec(rng.range(54, 70));
      default: {
        // a foul; now and then a long stoppage for treatment or a change
        const base = rng.logNormal(27, 0.55, 9, 95);
        return sec(rng.chance(0.07) ? base + rng.range(45, 80) : base);
      }
    }
  };
  const restartFor = (kind: EndKind, team: Team): Team => {
    switch (kind) {
      case "throw":
        return rng.chance(0.82) ? other(team) : team;
      case "corner":
        return team;
      case "foul":
        return rng.chance(0.72) ? team : other(team);
      default:
        return other(team); // goal kick, offside, goal
    }
  };
  /** Rewrites the end of the last phase so that this team has the ball next. */
  const giveTo = (want: Team) => {
    const last = phases[phases.length - 1];
    if (!last) return;
    if (last.kind === "live") last.endKind = last.team === want ? "continue" : last.endKind === "intercept" ? "intercept" : "tackle";
    else if (last.kind === "dead") last.restartTeam = want;
    else last.nextTeam = want;
  };

  for (const half of [0, 1] as const) {
    const halfEnd = half === 0 ? H1_END : DEMO_FRAMES;
    let f: number;
    let team: Team = half === 0 ? 0 : 1;
    if (half === 0) {
      phases.push({ kind: "none", start: 0, end: LEAD_IN, reason: "pre", nextTeam: team });
      f = LEAD_IN;
    } else {
      phases.push({ kind: "none", start: H1_END, end: H2_START, reason: "halftime", nextTeam: team });
      f = H2_START;
    }

    while (f < halfEnd) {
      const anchor = anchorAt < ANCHORS.length && ANCHORS[anchorAt].start < halfEnd ? ANCHORS[anchorAt] : null;
      const wall = anchor ? anchor.start : halfEnd;

      if (!anchor && wall - f < MIN_LIVE && phases.length) {
        // too little time left before the whistle for anything new: the last phase runs to it
        const last = phases[phases.length - 1];
        last.end = wall;
        if (last.kind === "live") last.endKind = "whistle";
        break;
      }

      let live: LivePhase;
      if (anchor && wall - f < MIN_LIVE) {
        // whatever came before lasts right up to the scripted moment
        const last = phases[phases.length - 1];
        if (f < wall && last) {
          last.end = wall;
          f = wall;
        }
        giveTo(anchor.team);
        live = { kind: "live", team: anchor.team, start: f, end: anchor.end, endKind: anchor.endKind, hide: anchor.hide, unsure: anchor.unsure };
        anchorAt++;
      } else {
        const length = possession(team, half);
        let end = f + length;
        let endKind = pickEnd(length);
        if (end > wall - MIN_LIVE) {
          end = wall;
          if (!anchor) endKind = "whistle";
          else if (anchor.team !== team) endKind = turnover();
          else if (!anchor.fresh) endKind = "continue";
          else {
            // The scripted spell has to start on its own, so the same side cannot simply carry on into it:
            // the ball goes out for a quick throw-in first.
            const gap = sec(8);
            const last = phases[phases.length - 1];
            if (wall - f >= MIN_LIVE + gap) {
              phases.push({ kind: "live", team, start: f, end: wall - gap, endKind: "throw" });
              phases.push({ kind: "dead", start: wall - gap, end: wall, restartTeam: anchor.team, planned: "throw" });
            } else if (last) {
              // no room for that: whatever came before lasts until the scripted spell
              last.end = wall;
            }
            f = wall;
            continue;
          }
        }
        live = { kind: "live", team, start: f, end, endKind };
      }
      phases.push(live);
      f = live.end;
      team = live.team;
      if (f >= halfEnd) break;

      const nextAnchor = anchorAt < ANCHORS.length && ANCHORS[anchorAt].start < halfEnd ? ANCHORS[anchorAt] : null;
      const nextWall = nextAnchor ? nextAnchor.start : halfEnd;
      switch (live.endKind) {
        case "continue":
          break;
        case "tackle":
        case "intercept":
          team = other(team);
          break;
        case "loose": {
          const length = sec(rng.range(2, 6.5));
          if (f + length > nextWall - MIN_LIVE) {
            live.endKind = "tackle";
            team = other(team);
            break;
          }
          const nextTeam: Team = rng.chance(0.5) ? 0 : 1;
          phases.push({ kind: "none", start: f, end: f + length, reason: "loose", nextTeam });
          f += length;
          team = nextTeam;
          break;
        }
        default: {
          let length = deadFrames(live.endKind);
          let restartTeam = restartFor(live.endKind, team);
          if (f + length > nextWall - MIN_LIVE) {
            if (nextWall - f < MIN_DEAD) {
              // No room for a stoppage before the next scripted moment or the whistle.
              if (nextAnchor && nextAnchor.fresh && nextAnchor.team === team && nextWall - f >= sec(2.5)) {
                // a quick restart, so the scripted spell still starts on its own
                phases.push({ kind: "dead", start: f, end: nextWall, restartTeam: team, planned: live.endKind });
                f = nextWall;
              } else if (nextAnchor) {
                live.endKind = nextAnchor.team === team ? "continue" : "tackle";
                team = nextAnchor.team;
              } else {
                live.end = halfEnd;
                live.endKind = "whistle";
                f = halfEnd;
              }
              break;
            }
            length = nextWall - f;
            if (nextAnchor) restartTeam = nextAnchor.team;
          }
          phases.push({ kind: "dead", start: f, end: f + length, restartTeam, planned: live.endKind });
          f += length;
          team = restartTeam;
        }
      }
    }
  }
  return phases;
}

/* ───────────────────────── pass 2: the simulation ───────────────────────── */

interface Slot {
  /** Lateral place, -1 … 1. */
  l: number;
  /** Depth within the team's block, 0 back line … 1 front line. */
  r: number;
  gk?: boolean;
}

// Slot 0 is the goalkeeper. Home play 4-3-3, Away 4-4-2.
const HOME_SHAPE: Slot[] = [
  { l: 0, r: 0, gk: true },
  { l: -0.74, r: 0.07 },
  { l: -0.26, r: 0 },
  { l: 0.26, r: 0 },
  { l: 0.74, r: 0.07 },
  { l: 0, r: 0.36 },
  { l: -0.4, r: 0.54 },
  { l: 0.4, r: 0.54 },
  { l: -0.78, r: 0.92 },
  { l: 0, r: 1 },
  { l: 0.78, r: 0.92 },
];
const AWAY_SHAPE: Slot[] = [
  { l: 0, r: 0, gk: true },
  { l: -0.74, r: 0.06 },
  { l: -0.25, r: 0 },
  { l: 0.25, r: 0 },
  { l: 0.74, r: 0.06 },
  { l: -0.78, r: 0.54 },
  { l: -0.26, r: 0.47 },
  { l: 0.26, r: 0.47 },
  { l: 0.78, r: 0.54 },
  { l: -0.2, r: 0.97 },
  { l: 0.22, r: 1 },
];
const SHAPES = [HOME_SHAPE, AWAY_SHAPE] as const;

interface Body {
  x: number;
  y: number;
  vx: number;
  vy: number;
  pace: number;
  /** Phases and rates of the slow drift that keeps a player from standing like a statue. */
  p1: number;
  p2: number;
  w1: number;
  w2: number;
}
interface Aim {
  x: number;
  y: number;
  v: number;
}

interface Note {
  why: string;
  confidence: number;
}

interface SimResult {
  tracking: Tracking;
  /** notes[k] describes the change from phase k to phase k + 1. */
  notes: (Note | null)[];
  goals: DemoScene["goals"];
}

function simulate(phases: Phase[], rng: Rng, names: [string, string]): SimResult {
  const count = Math.ceil(DEMO_FRAMES / EVERY);
  const ballOut = new Float32Array(count * 2);
  const teamOut = [new Float32Array(count * 22), new Float32Array(count * 22)];
  const hideScore = new Float32Array(count);
  /** 1 certainly hidden, 2 certainly seen, 0 decided by the score. */
  const forced = new Uint8Array(count);
  /** 0 live, 1 dead, 2 none. */
  const klass = new Uint8Array(count);
  const notes: (Note | null)[] = new Array(phases.length).fill(null);
  const goals: DemoScene["goals"] = [];

  const dirOf = (team: Team, frame: number): 1 | -1 => ((team === 0) === frame < H1_END ? 1 : -1);
  const depth = (team: Team, frame: number, x: number) => (dirOf(team, frame) > 0 ? x + HL : HL - x);
  const fromDepth = (team: Team, frame: number, s: number) => (dirOf(team, frame) > 0 ? s - HL : HL - s);

  // ── players ──
  const P: Body[][] = [[], []];
  for (const team of [0, 1] as const) {
    for (let i = 0; i < 11; i++) {
      P[team].push({
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        pace: rng.range(0.92, 1.08),
        p1: rng.range(0, 6.28),
        p2: rng.range(0, 6.28),
        w1: rng.range(0.32, 0.85),
        w2: rng.range(0.35, 0.9),
      });
    }
  }
  const aims: Aim[][] = [[], []];
  for (const team of [0, 1] as const) for (let i = 0; i < 11; i++) aims[team].push({ x: 0, y: 0, v: 0 });
  const override: (Aim | null)[][] = [new Array<Aim | null>(11).fill(null), new Array<Aim | null>(11).fill(null)];
  const clearOverrides = () => {
    override[0].fill(null);
    override[1].fill(null);
  };

  // ── team shape, smoothed so the block moves like a block ──
  const shape = [
    { sRef: HL, yRef: 0, poss: 0.5 },
    { sRef: HL, yRef: 0, poss: 0.5 },
  ];

  // ── ball ──
  let bx = 0;
  let by = 0;
  let mode = "placed" as Mode;
  /** Reads the ball's state afresh (the closures above change it). */
  const is = (...modes: Mode[]) => modes.includes(mode);
  let carrier = { team: 0 as Team, i: 9 };
  let pass = { fx: 0, fy: 0, toTeam: 0 as Team, to: 0, t0: 0, t1: 1 };
  let exit = { fx: 0, fy: 0, tx: 0, ty: 0, t0: 0, t1: 1 };
  let looseAim = { x: 0, y: 0, until: 0 };
  let holdUntil = 0;
  let carrySpeed = 3.5;
  let carryAngle = 0;
  let firstPass: number | null = null;
  let advance = false;

  // ── what the dead ball or the kick-off is waiting for ──
  let kickoffMode = true as boolean;
  let kickTeam = 0 as Team;
  let taker = { team: 0 as Team, i: 9 };
  let spot = { x: 0, y: 0 };
  let refPoint = { x: 0, y: 0 };
  let possTeam = -1 as Team | -1;
  let prep = 3;
  let restart = "kickoff" as Restart;
  let presser: [number, number] = [-1, -1];

  let ending = { launched: false, winner: -1, actual: null as EndKind | null, exitX: 0, exitY: 0, side: 1, rx: 6, ry: 10, lead: 1.2 };

  const nearest = (team: Team, x: number, y: number, skipGk: boolean, skip = -1): number => {
    let best = -1;
    let bestD = Infinity;
    for (let i = skipGk ? 1 : 0; i < 11; i++) {
      if (i === skip) continue;
      const d = hyp(P[team][i].x - x, P[team][i].y - y);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };
  const distTo = (team: Team, i: number, x: number, y: number) => hyp(P[team][i].x - x, P[team][i].y - y);

  const kickoffSpot = (team: Team, i: number, frame: number): Aim => {
    const slot = SHAPES[team][i];
    const dir = dirOf(team, frame);
    let s = slot.gk ? 5 : 18 + slot.r * 27;
    let y = slot.gk ? 0 : slot.l * HW * 0.82 * dir;
    if (team === kickTeam) {
      if (i === 9) {
        s = HL;
        y = 0;
      } else if (i === 6) {
        s = HL - 1.2;
        y = 2.6 * dir;
      }
    }
    return { x: fromDepth(team, frame, s), y, v: 3.2 };
  };

  const shapeSpot = (team: Team, i: number, frame: number, out: Aim) => {
    const slot = SHAPES[team][i];
    const st = shape[team];
    const dir = dirOf(team, frame);
    const p = st.poss;
    const backIn = clamp(st.sRef - 30, 17, 62);
    const backOut = clamp(st.sRef - 15, 13, 50);
    const back = lerp(backOut, backIn, p);
    if (slot.gk) {
      out.x = fromDepth(team, frame, clamp(back * 0.3, 3, 15));
      out.y = st.yRef * 0.1;
      return;
    }
    const length = lerp(31, 42, p);
    const width = lerp(0.62, 1, p);
    const s = clamp(back + slot.r * length, 3, 101);
    out.x = fromDepth(team, frame, s);
    out.y = clamp(slot.l * HW * width * dir + st.yRef * lerp(0.5, 0.22, p), -32.5, 32.5);
  };

  const zoneWords = (team: Team, frame: number, x: number) => {
    const s = depth(team, frame, x);
    return s < 35 ? "in their own third" : s < 70 ? "in midfield" : "high up the pitch";
  };
  const sideWords = (y: number) => (y < 0 ? "near" : "far");

  const newHold = (t: number, quick: boolean) => {
    const pressed = presser[other(carrier.team)] >= 0 && distTo(other(carrier.team), presser[other(carrier.team)], bx, by) < 3.2;
    const base = quick ? rng.range(0.3, 0.7) : rng.chance(0.16) ? rng.range(0.25, 0.6) : rng.range(0.9, 3.3);
    holdUntil = t + base * (pressed ? 0.7 : 1);
    carrySpeed = rng.range(2.4, 5.2);
    carryAngle = rng.normal() * (advance ? 0.35 : 0.62);
  };

  const pickReceiver = (team: Team, frame: number): number => {
    if (firstPass !== null) {
      const forcedTo = firstPass;
      firstPass = null;
      if (forcedTo !== carrier.i) return forcedTo;
    }
    const me = P[team][carrier.i];
    const myDepth = depth(team, frame, me.x);
    let best = -1;
    let bestScore = -Infinity;
    for (let j = 0; j < 11; j++) {
      if (j === carrier.i) continue;
      if (j === 0 && !(myDepth < 30 && rng.chance(0.08))) continue;
      const q = P[team][j];
      const d = hyp(q.x - me.x, q.y - me.y);
      if (d < 5 || d > 42) continue;
      const gain = depth(team, frame, q.x) - myDepth;
      const marker = nearest(other(team), q.x, q.y, true);
      const marked = marker >= 0 && distTo(other(team), marker, q.x, q.y) < 3 ? 0.5 : 0;
      const score = (advance ? 1.35 : 0.45) * clamp(gain / 18, -1, 1.4) - Math.abs(d - 15) / 22 - marked + rng.range(0, 0.9);
      if (score > bestScore) {
        bestScore = score;
        best = j;
      }
    }
    return best >= 0 ? best : nearest(team, me.x, me.y, true, carrier.i);
  };

  const launchPass = (toTeam: Team, to: number, t: number, arrive: number | null) => {
    const q = P[toTeam][to];
    const d = hyp(q.x - bx, q.y - by);
    const speed = clamp(10 + d * 0.33, 11, 23);
    pass = { fx: bx, fy: by, toTeam, to, t0: t, t1: arrive ?? t + Math.max(0.4, d / speed) };
    if (pass.t1 <= pass.t0) pass.t1 = pass.t0 + DT;
    mode = "pass";
  };

  /** Sets up the restart that follows a stoppage: where it is taken, who takes it and where everyone goes. */
  const setupRestart = (kind: Restart, team: Team, frame: number, length: number) => {
    restart = kind;
    clearOverrides();
    kickoffMode = false;
    possTeam = team;
    const dir = dirOf(team, frame);
    const foe = other(team);
    prep = Math.min(rng.range(2.6, 5), length * 0.45);
    switch (kind) {
      case "kickoff": {
        kickoffMode = true;
        kickTeam = team;
        spot = { x: 0, y: 0 };
        refPoint = { x: 0, y: 0 };
        taker = { team, i: 9 };
        firstPass = 5;
        prep = Math.min(5, length * 0.3);
        break;
      }
      case "throw": {
        refPoint = { x: spot.x, y: spot.y };
        const i = nearest(team, spot.x, spot.y, true);
        taker = { team, i };
        override[team][i] = { x: spot.x, y: spot.y + sign(spot.y) * 0.3, v: 4.5 };
        const inward = -sign(spot.y);
        const offers: [number, number][] = [
          [dir * 7, inward * 6],
          [-dir * 5, inward * 9],
        ];
        const used = new Set<number>([i]);
        offers.forEach(([ox, oy], n) => {
          let best = -1;
          let bestD = Infinity;
          for (let j = 1; j < 11; j++) {
            if (used.has(j)) continue;
            const d = distTo(team, j, spot.x + ox, spot.y + oy);
            if (d < bestD) {
              bestD = d;
              best = j;
            }
          }
          if (best < 0) return;
          used.add(best);
          const ax = clamp(spot.x + ox, -50, 50);
          const ay = spot.y + oy;
          override[team][best] = { x: ax, y: ay, v: 3.6 };
          if (n === 0) firstPass = best;
          const marker = nearest(foe, ax, ay, true);
          if (marker >= 0 && !override[foe][marker]) override[foe][marker] = { x: ax - dir * 1.6, y: ay + inward * 1, v: 3.6 };
        });
        break;
      }
      case "goalkick": {
        const goalX = -dir * HL;
        const sideY = sign(spot.y || rng.sign());
        spot = { x: goalX + dir * 5.5, y: sideY * 5 };
        refPoint = { x: goalX + dir * 30, y: 0 };
        taker = { team, i: 0 };
        override[team][0] = { x: spot.x - dir * 0.6, y: spot.y, v: 3.4 };
        override[team][2] = { x: goalX + dir * 13, y: -16 * dir, v: 3.4 };
        override[team][3] = { x: goalX + dir * 13, y: 16 * dir, v: 3.4 };
        firstPass = rng.chance(0.6) ? (rng.chance(0.5) ? 2 : 3) : rng.chance(0.5) ? 6 : 7;
        break;
      }
      case "corner": {
        const goalX = dir * HL;
        const sideY = sign(spot.y || rng.sign());
        spot = { x: goalX - dir * 0.5, y: sideY * (HW - 0.4) };
        refPoint = { x: goalX - dir * 25, y: sideY * 6 };
        let i = -1;
        let bestD = Infinity;
        for (let j = 5; j < 11; j++) {
          const d = distTo(team, j, spot.x, spot.y);
          if (d < bestD) {
            bestD = d;
            i = j;
          }
        }
        taker = { team, i };
        override[team][i] = { x: spot.x, y: spot.y, v: 4.6 };
        const places: [number, number][] = [
          [6, -3],
          [8, 2.5],
          [10.5, -0.5],
          [7, 6.5],
          [13, -5.5],
          [19, 3],
        ];
        const attackers = [5, 6, 7, 8, 9, 10, 2].filter((j) => j !== i).slice(0, places.length);
        const markers = [1, 2, 3, 4, 5, 6, 7, 8];
        attackers.forEach((j, n) => {
          const ax = goalX - dir * places[n][0];
          const ay = places[n][1] * sideY;
          override[team][j] = { x: ax, y: ay, v: 4 };
          const m = markers[n];
          if (m !== undefined) override[foe][m] = { x: ax + dir * 1.3, y: ay * 0.9, v: 4 };
        });
        override[foe][0] = { x: goalX - dir * 0.9, y: 0, v: 3 };
        firstPass = attackers[rng.int(0, Math.min(3, attackers.length - 1))];
        break;
      }
      case "freekick": {
        spot = { x: clamp(bx, -HL + 1, HL - 1), y: clamp(by, -HW + 1, HW - 1) };
        refPoint = { x: spot.x, y: spot.y };
        const i = nearest(team, spot.x, spot.y, true);
        taker = { team, i };
        override[team][i] = { x: spot.x - dir * 0.7, y: spot.y, v: 4 };
        break;
      }
    }
  };

  const enterPhase = (k: number, frame: number, t: number) => {
    const ph = phases[k];
    const prev: Phase | undefined = phases[k - 1];
    const say = (why: string, lo: number, hi: number) => {
      if (k > 0) notes[k - 1] = { why, confidence: rng.range(lo, hi) };
    };

    if (ph.kind === "none") {
      possTeam = -1;
      if (ph.reason === "pre") {
        kickoffMode = true;
        kickTeam = ph.nextTeam;
        mode = "placed";
        bx = 0;
        by = 0;
        spot = { x: 0, y: 0 };
        refPoint = { x: 0, y: 0 };
      } else if (ph.reason === "halftime") {
        clearOverrides();
        kickoffMode = true;
        kickTeam = ph.nextTeam;
        mode = "out";
        spot = { x: 0, y: 0 };
        refPoint = { x: 0, y: 0 };
        say("Half-time whistle.", 0.95, 0.99);
      } else {
        mode = "loose";
        looseAim = { x: bx, y: by, until: t };
        refPoint = { x: bx, y: by };
        say("Loose ball: neither side had it under control.", 0.5, 0.62);
      }
      return;
    }

    if (ph.kind === "dead") {
      const before = prev as LivePhase;
      const T = before.team;
      const R = ph.restartTeam;
      const kind = ending.actual ?? before.endKind;
      const length = (ph.end - ph.start) / FPS;
      if (kind === "goal") {
        goals.push({ frame: ph.start, side: T });
        mode = "out";
        setupRestart("kickoff", R, frame, length);
        say(`Goal for ${names[T]}.`, 0.95, 0.99);
      } else if (kind === "throw") {
        spot = { x: clamp(ending.exitX, -HL + 1, HL - 1), y: sign(ending.exitY) * HW };
        mode = "out";
        setupRestart("throw", R, frame, length);
        say(`The ball went out on the ${sideWords(spot.y)} touchline. Throw-in to ${names[R]}.`, 0.88, 0.98);
      } else if (kind === "goalkick" || kind === "corner") {
        // Over a goal line: a goal kick if the team defending that end restarts, a corner if the attackers do.
        const end = sign(ending.exitX);
        const defending: Team = dirOf(0, frame) === -end ? 0 : 1;
        spot = { x: end * HL, y: ending.exitY };
        mode = "out";
        if (R === defending) {
          setupRestart("goalkick", R, frame, length);
          say(`${names[other(R)]} put the ball behind the goal. Goal kick to ${names[R]}.`, 0.88, 0.98);
        } else {
          setupRestart("corner", R, frame, length);
          say(`${names[other(R)]} turned the ball behind. Corner to ${names[R]}.`, 0.86, 0.97);
        }
      } else {
        // a foul or an offside flag: the ball stops where it is
        mode = "still";
        setupRestart("freekick", R, frame, length);
        if (kind === "offside" && R !== T) say(`Offside against ${names[T]}. Free kick to ${names[R]}.`, 0.8, 0.94);
        else if (R === T) say(`Foul on a ${names[T]} player. Free kick to ${names[T]}.`, 0.78, 0.93);
        else say(`Foul by ${names[T]}. Free kick to ${names[R]}.`, 0.78, 0.93);
      }
      return;
    }

    // a team has the ball
    const T = ph.team;
    advance = ph.endKind === "goalkick" || ph.endKind === "corner" || ph.endKind === "goal" || ph.endKind === "offside";
    if (!prev || prev.kind === "none") {
      if (prev && prev.reason === "loose") {
        const i = nearest(T, bx, by, true);
        carrier = { team: T, i };
        launchPass(T, i, t, t + clamp(distTo(T, i, bx, by) / 10, 0.2, 0.6));
        say(`${names[T]} came away with the loose ball.`, 0.62, 0.84);
      } else {
        carrier = { team: T, i: 9 };
        mode = "carried";
        firstPass = 5;
        newHold(t, true);
        say(prev && prev.reason === "halftime" ? `${names[T]} kicked off the second half.` : `${names[T]} kicked off.`, 0.95, 0.99);
      }
    } else if (prev.kind === "dead") {
      // whoever is on the ball takes it; a taker still walking over gets it rolled to them
      let i = taker.i;
      if (distTo(T, i, bx, by) > 1.5) {
        const near = nearest(T, bx, by, restart !== "goalkick");
        if (near >= 0 && distTo(T, near, bx, by) < distTo(T, i, bx, by)) i = near;
      }
      carrier = { team: T, i };
      const d = distTo(T, i, bx, by);
      if (d > 1.5) launchPass(T, i, t, t + clamp(d / 9, 0.3, 1.2));
      else {
        mode = "carried";
        newHold(t, true);
      }
      const what = restart === "throw" ? "the throw-in" : restart === "goalkick" ? "the goal kick" : restart === "corner" ? "the corner" : restart === "freekick" ? "the free kick" : null;
      say(what ? `${names[T]} took ${what}.` : `${names[T]} kicked off.`, 0.86, 0.98);
    } else if (prev.endKind !== "continue") {
      const cutOut = mode === "pass";
      const i = ending.winner >= 0 ? ending.winner : nearest(T, bx, by, true);
      carrier = { team: T, i };
      const d = distTo(T, i, bx, by);
      if (d > 1.2) launchPass(T, i, t, t + clamp(d / 12, 0.2, 0.6));
      else {
        mode = "carried";
        newHold(t, false);
      }
      const where = zoneWords(T, frame, bx);
      if (cutOut) say(`${names[T]} cut out a pass ${where}.`, 0.72, 0.93);
      else say(`${names[T]} won the ball with a tackle ${where}.`, 0.66, 0.9);
    }
    clearOverrides();
    kickoffMode = false;
    possTeam = T;
    presser = [-1, -1];
    ending = { launched: false, winner: -1, actual: null, exitX: 0, exitY: 0, side: rng.sign(), rx: rng.range(3, 12), ry: rng.range(0, 1), lead: rng.range(1, 1.7) };
  };

  // ── first positions: both teams lined up for the kick-off ──
  kickTeam = 0;
  for (const team of [0, 1] as const) {
    for (let i = 0; i < 11; i++) {
      const at = kickoffSpot(team, i, 0);
      P[team][i].x = at.x;
      P[team][i].y = at.y;
    }
  }

  let k = -1;
  let lastBallSpeed = 0;
  for (let n = 0; n < count; n++) {
    const frame = n * EVERY;
    const t = n * DT;
    while (k + 1 < phases.length && phases[k + 1].start <= frame) {
      k++;
      enterPhase(k, frame, t);
    }
    const ph = phases[k];
    const tEnd = ph.end / FPS;
    const remain = (ph.end - frame) / FPS;
    const open = ph.kind === "live" || (ph.kind === "none" && ph.reason === "loose");

    /* ── what the ball is at this moment ── */
    if (ph.kind === "live") {
      if (is("pass") && t >= pass.t1 - 1e-6) {
        carrier = { team: pass.toTeam, i: pass.to };
        mode = "carried";
        newHold(t, false);
      }
    } else if (ph.kind === "dead") {
      if (is("exit")) mode = "out";
      if (is("out", "still") && remain <= prep) {
        mode = "placed";
        bx = spot.x;
        by = spot.y;
      }
    } else if (ph.reason === "halftime" && !is("placed") && remain <= 6) {
      mode = "placed";
      bx = 0;
      by = 0;
    }

    /* ── write the sample: everything as it stands at this frame ── */
    ballOut[n * 2] = bx;
    ballOut[n * 2 + 1] = by;
    klass[n] = ph.kind === "live" ? 0 : ph.kind === "dead" ? 1 : 2;
    if (is("out") || Math.abs(bx) > HL + 0.6 || Math.abs(by) > HW + 0.6) forced[n] = 1;
    else if (ph.kind === "none" && ph.reason === "pre") forced[n] = 2;
    {
      let score: number;
      if (is("carried")) {
        const foe = other(carrier.team);
        const m = nearest(foe, bx, by, true);
        const d = m >= 0 ? distTo(foe, m, bx, by) : 9;
        score = 0.3 + 0.45 * clamp(1 - (d - 0.9) / 3.6, 0, 1);
      } else if (is("pass")) score = 0.12 + clamp((lastBallSpeed - 14) / 14, 0, 0.45);
      else if (is("exit")) score = 0.3;
      else if (is("loose")) score = 0.92;
      else score = 0.5;
      // occlusion comes in passages, not flickers: slow waves decide most of it
      score += 0.1 * Math.sin(t * 1.7 + 0.4) + 0.2 * Math.sin(t * 0.69 + 2.1) + 0.24 * Math.sin(t * 0.27 + 4.4) + 0.2 * Math.sin(t * 0.103 + 1.3) + rng.range(-0.05, 0.05);
      hideScore[n] = score;
    }
    for (const team of [0, 1] as const) {
      const out = teamOut[team];
      for (let i = 0; i < 11; i++) {
        out[(n * 11 + i) * 2] = P[team][i].x;
        out[(n * 11 + i) * 2 + 1] = P[team][i].y;
      }
    }

    /* ── decisions: how this possession ends, and the next pass ── */
    if (ph.kind === "live") {
      const T = ph.team;
      const O = other(T);
      const dir = dirOf(T, frame);

      if (!ending.launched) {
        const tryExit = (kind: "throw" | "goalkick" | "corner" | "goal") => {
          let ex: number;
          let ey: number;
          let speed: number;
          if (kind === "throw") {
            const side = Math.abs(by) > 5 ? sign(by) : ending.side;
            ex = clamp(bx + dir * ending.rx, -51, 51);
            ey = side * (HW + 1.4);
            speed = 17;
          } else if (kind === "goal") {
            ex = dir * (HL + 1.7);
            ey = lerp(-2.5, 2.5, ending.ry);
            speed = 25;
          } else {
            ex = dir * (HL + 1.5);
            ey = ending.side * (kind === "goalkick" ? lerp(7, 20, ending.ry) : lerp(4.5, 15, ending.ry));
            speed = 21;
          }
          const d = hyp(ex - bx, ey - by);
          if (remain > d / speed + DT * 0.5) return; // not time yet
          const needed = d / Math.max(remain, DT);
          const reachable = needed <= 31 && (kind === "throw" || kind === "goal" || d <= 45);
          if (!reachable && kind !== "goal") {
            if (kind !== "throw") {
              // too far from the goal line: the ball goes out for a throw-in instead
              ending.actual = "throw";
              tryExit("throw");
            } else {
              // too far from any line: the whistle goes for a foul
              ending.actual = "foul";
              ending.launched = true;
            }
            return;
          }
          exit = { fx: bx, fy: by, tx: ex, ty: ey, t0: t, t1: Math.max(tEnd, t + DT) };
          mode = "exit";
          ending.launched = true;
          ending.actual = kind;
          ending.exitX = ex;
          ending.exitY = ey;
        };

        const wanted = ending.actual ?? ph.endKind;
        switch (wanted) {
          case "throw":
          case "goalkick":
          case "corner":
          case "goal":
            tryExit(wanted);
            break;
          case "offside":
            if (remain <= 1.3) {
              let far = -1;
              let farS = -Infinity;
              for (let j = 1; j < 11; j++) {
                if (j === carrier.i && is("carried")) continue;
                const s = depth(T, frame, P[T][j].x);
                if (s > farS) {
                  farS = s;
                  far = j;
                }
              }
              if (far >= 0) launchPass(T, far, t, Math.max(tEnd, t + DT));
              ending.launched = true;
            }
            break;
          case "tackle":
            if (remain <= 1.4) {
              ending.winner = nearest(O, bx, by, true);
              ending.launched = true;
              if (is("pass")) launchPass(O, ending.winner, t, Math.max(tEnd, t + DT));
            }
            break;
          case "intercept":
            if (remain <= ending.lead) {
              const aimAt = is("pass") && pass.toTeam === T ? pass.to : pickReceiver(T, frame);
              const q = P[T][aimAt];
              ending.winner = nearest(O, lerp(bx, q.x, 0.7), lerp(by, q.y, 0.7), true);
              launchPass(O, ending.winner, t, Math.max(tEnd, t + DT));
              ending.launched = true;
            }
            break;
          default:
            break;
        }
      }

      if (is("carried") && !ending.launched && carrier.team === T && t >= holdUntil && remain > 0.5) {
        launchPass(T, pickReceiver(T, frame), t, null);
      }
    } else if (ph.kind === "none" && ph.reason === "loose" && t >= looseAim.until) {
      const team: Team = rng.chance(0.5) ? 0 : 1;
      const i = nearest(team, bx + rng.range(-4, 4), by + rng.range(-4, 4), true);
      looseAim = { x: P[team][i].x + rng.range(-1.5, 1.5), y: P[team][i].y + rng.range(-1.5, 1.5), until: t + rng.range(0.4, 0.9) };
    }

    /* ── where everyone wants to be ── */
    const rx = open ? bx : refPoint.x;
    const ry = open ? by : refPoint.y;
    for (const team of [0, 1] as const) {
      const st = shape[team];
      const ease = 1 - Math.exp(-DT / 1.6);
      st.sRef += (depth(team, frame, clamp(rx, -HL, HL)) - st.sRef) * ease;
      st.yRef += (clamp(ry, -HW, HW) - st.yRef) * ease;
      const want = possTeam === -1 ? 0.5 : possTeam === team ? 1 : 0;
      st.poss += (want - st.poss) * (1 - Math.exp(-DT / 2.2));
      for (let i = 0; i < 11; i++) {
        const aim = aims[team][i];
        const body = P[team][i];
        if (kickoffMode) {
          const at = kickoffSpot(team, i, frame);
          aim.x = at.x;
          aim.y = at.y;
          aim.v = ph.kind === "none" && ph.reason === "pre" ? 0.8 : ph.kind === "none" ? 2.7 : 3.3;
        } else {
          shapeSpot(team, i, frame, aim);
          aim.v = open ? 5.6 * body.pace : 3.1;
        }
        // a slow personal drift, smaller when the ball is dead
        const drift = open ? 1.25 : 0.35;
        aim.x += Math.sin(t * body.w1 + body.p1) * drift;
        aim.y += Math.cos(t * body.w2 + body.p2) * drift;
        const forcedAim = override[team][i];
        if (forcedAim) {
          aim.x = forcedAim.x;
          aim.y = forcedAim.y;
          aim.v = forcedAim.v;
        }
      }
    }

    if (ph.kind === "live") {
      const T = ph.team;
      const O = other(T);
      const dir = dirOf(T, frame);
      if (is("carried") && carrier.team === T) {
        const c = P[T][carrier.i];
        let ax = dir * Math.cos(carryAngle);
        let ay = Math.sin(carryAngle);
        const s = depth(T, frame, c.x);
        if (Math.abs(c.y) > 27) ay = -sign(c.y) * 0.85;
        if (s > 90) {
          ax = -dir * 0.25;
          ay = -sign(c.y || 1);
        } else if (s < 6) ax = dir;
        const aim = aims[T][carrier.i];
        aim.x = c.x + ax * 5;
        aim.y = c.y + ay * 5;
        aim.v = carrySpeed;
      } else if (is("pass") && pass.toTeam === T) {
        // the receiver checks towards the ball
        const q = P[T][pass.to];
        const aim = aims[T][pass.to];
        aim.x = lerp(aim.x, lerp(q.x, bx, 0.25), 0.6);
        aim.y = lerp(aim.y, lerp(q.y, by, 0.25), 0.6);
      }
      // the defending side: one presses the ball, one covers behind
      const oDir = dirOf(O, frame);
      let first = nearest(O, bx, by, true);
      if (presser[O] >= 0 && presser[O] !== first && distTo(O, presser[O], bx, by) < distTo(O, first, bx, by) + 2.5) first = presser[O];
      presser[O] = first;
      const a1 = aims[O][first];
      a1.x = bx - oDir * 1.4;
      a1.y = by;
      a1.v = 7.2;
      const second = nearest(O, bx - oDir * 6, by, true, first);
      if (second >= 0) {
        const a2 = aims[O][second];
        a2.x = lerp(a2.x, bx - oDir * 6.5, 0.7);
        a2.y = lerp(a2.y, by + sign(P[O][second].y - by) * 3, 0.7);
        a2.v = 6.2;
      }
      if (ending.winner >= 0) {
        const w = aims[O][ending.winner];
        const meet = is("pass") && pass.toTeam === O ? 0.5 : 0;
        w.x = lerp(bx, P[O][ending.winner].x, meet);
        w.y = lerp(by, P[O][ending.winner].y, meet);
        w.v = 8.2;
      }
    } else if (ph.kind === "none" && ph.reason === "loose") {
      for (const team of [0, 1] as const) {
        const first = nearest(team, bx, by, true);
        const second = nearest(team, bx, by, true, first);
        for (const i of [first, second]) {
          if (i < 0) continue;
          aims[team][i].x = bx + rng.range(-0.8, 0.8);
          aims[team][i].y = by + rng.range(-0.8, 0.8);
          aims[team][i].v = 6.6;
        }
      }
    } else if (ph.kind === "dead" && restart === "freekick") {
      // the other side keep their distance from the free kick
      const foe = other(taker.team);
      const oDir = dirOf(foe, frame);
      for (let i = 1; i < 11; i++) {
        const aim = aims[foe][i];
        const d = hyp(aim.x - spot.x, aim.y - spot.y);
        if (d < 9.5) {
          const ux = d > 0.1 ? (aim.x - spot.x) / d : -oDir;
          const uy = d > 0.1 ? (aim.y - spot.y) / d : 0;
          aim.x = spot.x + ux * 9.6;
          aim.y = spot.y + uy * 9.6;
        }
      }
    }

    /* ── move ── */
    for (const team of [0, 1] as const) {
      for (let i = 0; i < 11; i++) {
        const body = P[team][i];
        const aim = aims[team][i];
        const dx = aim.x - body.x;
        const dy = aim.y - body.y;
        const d = hyp(dx, dy);
        const want = Math.min(aim.v, d * 1.6);
        const wx = d > 1e-6 ? (dx / d) * want : 0;
        const wy = d > 1e-6 ? (dy / d) * want : 0;
        let ax = wx - body.vx;
        let ay = wy - body.vy;
        const a = hyp(ax, ay);
        const maxStep = 4.6 * DT;
        if (a > maxStep) {
          ax = (ax / a) * maxStep;
          ay = (ay / a) * maxStep;
        }
        body.vx += ax;
        body.vy += ay;
        body.x = clamp(body.x + body.vx * DT, -HL - 2.5, HL + 2.5);
        body.y = clamp(body.y + body.vy * DT, -HW - 2.5, HW + 2.5);
      }
    }
    // nobody stands inside anybody else
    for (let a = 0; a < 22; a++) {
      const pa = P[a < 11 ? 0 : 1][a % 11];
      for (let b = a + 1; b < 22; b++) {
        const pb = P[b < 11 ? 0 : 1][b % 11];
        const dx = pb.x - pa.x;
        const dy = pb.y - pa.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 0.81 && d2 > 1e-8) {
          const d = Math.sqrt(d2);
          const push = (0.9 - d) / 2;
          pa.x -= (dx / d) * push;
          pa.y -= (dy / d) * push;
          pb.x += (dx / d) * push;
          pb.y += (dy / d) * push;
        }
      }
    }

    /* ── the ball moves on to the next sample ── */
    const px = bx;
    const py = by;
    if (is("carried")) {
      const c = P[carrier.team][carrier.i];
      const sp = hyp(c.vx, c.vy);
      if (sp > 0.6) {
        bx = c.x + (c.vx / sp) * 0.7;
        by = c.y + (c.vy / sp) * 0.7;
      } else {
        bx = c.x + dirOf(carrier.team, frame) * 0.5;
        by = c.y;
      }
    } else if (is("pass")) {
      const kk = clamp((t + DT - pass.t0) / (pass.t1 - pass.t0), 0, 1);
      const q = P[pass.toTeam][pass.to];
      bx = lerp(pass.fx, q.x, kk);
      by = lerp(pass.fy, q.y, kk);
    } else if (is("exit")) {
      const kk = clamp((t + DT - exit.t0) / (exit.t1 - exit.t0), 0, 1);
      bx = lerp(exit.fx, exit.tx, kk);
      by = lerp(exit.fy, exit.ty, kk);
    } else if (is("loose")) {
      const dx = looseAim.x - bx;
      const dy = looseAim.y - by;
      const d = hyp(dx, dy);
      const step = Math.min(d, 8.5 * DT);
      if (d > 1e-6) {
        bx += (dx / d) * step;
        by += (dy / d) * step;
      }
    }
    lastBallSpeed = hyp(bx - px, by - py) / DT;
  }

  /* ── which samples see the ball ── */
  const hidden = new Uint8Array(count);
  const liveScores: number[] = [];
  const deadScores: number[] = [];
  const noneScores: number[] = [];
  let forcedHidden = 0;
  for (let n = 0; n < count; n++) {
    if (forced[n] === 1) {
      hidden[n] = 1;
      forcedHidden++;
    } else if (forced[n] === 0) (klass[n] === 0 ? liveScores : klass[n] === 1 ? deadScores : noneScores).push(hideScore[n]);
  }
  const cutFor = (scores: number[], hideShare: number): number => {
    if (!scores.length || hideShare <= 0) return Infinity;
    const sorted = Float32Array.from(scores).sort();
    return sorted[clamp(Math.floor(sorted.length * (1 - hideShare)), 0, sorted.length - 1)];
  };
  const targetHidden = 0.4 * count;
  const liveShare = 0.27;
  const noneShare = 0.7;
  const fromDead = targetHidden - forcedHidden - liveShare * liveScores.length - noneShare * noneScores.length;
  const deadShare = clamp(fromDead / Math.max(1, deadScores.length), 0.05, 0.9);
  const liveCut = cutFor(liveScores, liveShare);
  const deadCut = cutFor(deadScores, deadShare);
  const noneCut = cutFor(noneScores, noneShare);
  for (let n = 0; n < count; n++) {
    if (forced[n] !== 0) continue;
    const cut = klass[n] === 0 ? liveCut : klass[n] === 1 ? deadCut : noneCut;
    if (hideScore[n] >= cut) hidden[n] = 1;
  }
  // no one-sample flickers
  for (let n = 1; n < count - 1; n++) {
    if (forced[n] === 0 && hidden[n - 1] === hidden[n + 1] && hidden[n] !== hidden[n - 1]) hidden[n] = hidden[n - 1];
  }
  for (const ph of phases) {
    if (ph.kind === "live" && ph.hide) {
      // in clear view for the three seconds before, so the moment it goes out of sight is plain
      for (let n = Math.ceil((ph.hide[0] - 3 * FPS) / EVERY); n * EVERY < ph.hide[0] && n < count; n++) if (n >= 0 && forced[n] !== 1) hidden[n] = 0;
      for (let n = Math.ceil(ph.hide[0] / EVERY); n * EVERY < ph.hide[1] && n < count; n++) hidden[n] = 1;
    }
  }

  const round1 = (v: number) => Math.round(v * 10) / 10 + 0; // "+ 0" turns a negative zero into zero
  const ball = new Float32Array(count * 2);
  for (let n = 0; n < count; n++) {
    if (hidden[n]) {
      ball[n * 2] = NaN;
      ball[n * 2 + 1] = NaN;
    } else {
      ball[n * 2] = round1(ballOut[n * 2]);
      ball[n * 2 + 1] = round1(ballOut[n * 2 + 1]);
    }
  }
  for (const out of teamOut) for (let i = 0; i < out.length; i++) out[i] = round1(out[i]);

  return {
    tracking: { every: EVERY, count, homeSlots: 11, awaySlots: 11, ball, home: teamOut[0], away: teamOut[1] },
    notes,
    goals,
  };
}

/* ───────────────────────── putting the run together ───────────────────────── */

export interface DemoMatch {
  parts: RunParts;
  scene: DemoScene;
}

const codeFor = (ph: Phase): LabelCode => (ph.kind === "live" ? (ph.team === 0 ? "H" : "A") : ph.kind === "dead" ? "D" : "X");
const round2 = (v: number) => Math.round(v * 100) / 100 + 0;

export function generateDemo(seed: number = DEMO_SEED): DemoMatch {
  const names: [string, string] = ["Home", "Away"];
  const phases = writeScript(new Rng(seed));
  const sim = simulate(phases, new Rng(seed ^ 0x9e3779b9), names);
  const rng = new Rng(seed ^ 0x51ed270b);
  const tracking = sim.tracking;

  // labels, and the phases behind each run
  const labels: LabelRun[] = [];
  const runPhases: Phase[][] = [];
  for (const ph of phases) {
    const code = codeFor(ph);
    const last = labels[labels.length - 1];
    if (last && last[0] === code) {
      last[1] += ph.end - ph.start;
      runPhases[runPhases.length - 1].push(ph);
    } else {
      labels.push([code, ph.end - ph.start]);
      runPhases.push([ph]);
    }
  }

  const spells: RunFileSpell[] = [];
  let at = 0;
  labels.forEach(([code, length], r) => {
    const start = at;
    const end = at + length - 1;
    at += length;
    if (code === "X") return;
    const seen = ballSeenShare(tracking, start, end) ?? 0;
    const seconds = length / FPS;
    let confidence: number;
    let ball: number;
    let shape: number;
    if (code === "D") {
      confidence = clamp(0.84 + 0.1 * rng.next() + (seconds > 8 ? 0.03 : -0.05), 0.6, 0.99);
      ball = 0.2 + 0.35 * seen;
      shape = (1 - ball) * rng.range(0.4, 0.55);
    } else {
      confidence = clamp(0.38 + 0.56 * Math.pow(seen, 1.2) + 0.02 * Math.log2(1 + seconds) + rng.range(-0.05, 0.05), 0.36, 0.99);
      ball = 0.12 + 0.68 * seen;
      shape = (1 - ball) * rng.range(0.55, 0.72);
    }
    if (runPhases[r].some((ph) => ph.kind === "live" && ph.unsure)) {
      confidence = 0.58;
      ball = 0.14;
      shape = 0.52;
    }
    ball = round2(ball);
    shape = round2(shape);
    spells.push({
      start,
      end,
      label: code,
      confidence: round2(confidence),
      evidence: { ball, shape, timing: round2(1 - ball - shape) },
      ball_seen: round2(seen),
    });
  });

  const transitions: RunFileTransition[] = [];
  for (let k = 0; k + 1 < phases.length; k++) {
    const from = codeFor(phases[k]);
    const to = codeFor(phases[k + 1]);
    if (from === to) continue;
    const frame = phases[k + 1].start;
    const note = sim.notes[k];
    // a change the ball was not seen making is a less certain change
    const n = Math.min(tracking.count - 1, Math.round(frame / EVERY));
    const unseen = tracking.ball[n * 2] !== tracking.ball[n * 2] && to !== "D" && to !== "X";
    const confidence = clamp((note?.confidence ?? 0.8) - (unseen ? 0.14 : 0), 0.4, 0.99);
    transitions.push({ frame, from, to, confidence: round2(confidence), ...(note ? { why: note.why } : {}) });
  }

  const parts: RunParts = {
    match: {
      title: "Home vs Away",
      home: names[0],
      away: names[1],
      fps: FPS,
      frames: DEMO_FRAMES,
      pitch: { length: 105, width: 68 },
      periods: [
        { name: "1st half", start: 0, end: H1_END - 1 },
        { name: "2nd half", start: H2_START, end: DEMO_FRAMES - 1 },
      ],
    },
    labels,
    spells,
    transitions,
    tracking,
    videoOffset: 0,
    videoWithheld: null,
    camera: null,
    generator: { name: "excensia-vantage", version: "0.1.0" },
  };
  return { parts, scene: { goals: sim.goals } };
}

/** The simulated match, ready for the viewer. */
export function demoRun(seed: number = DEMO_SEED): { run: Run; scene: DemoScene } {
  const { parts, scene } = generateDemo(seed);
  return { run: buildRun(parts, "demo", null), scene };
}

/** The simulated match as a run file, in the exact shape the engine writes. */
export function demoRunFile(seed: number = DEMO_SEED): RunFile {
  return toRunFile(demoRun(seed).run);
}
