import { buildChanges, buildSegments, indexRuns, isLabelCode, labelOf, lowerIndex, mergeRuns } from "./derive";
import {
  RUN_SCHEMA,
  type Label,
  type LabelRun,
  type Period,
  type Run,
  type RunFile,
  type RunFileSpell,
  type RunFileTransition,
  type RunSource,
  type Segment,
  type Tracking,
} from "./types";

/** A run file that cannot be opened. The message is written for the person who dropped the file. */
export class RunFileError extends Error {
  override name = "RunFileError";
}

const LABEL_WORDS = ["Home", "Away", "Dead ball", "No answer"] as const;
const fmt = (n: number) => (Number.isFinite(n) ? Math.round(n).toLocaleString("en-GB") : String(n));

function fail(message: string): never {
  throw new RunFileError(message);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function show(value: unknown): string {
  if (value === undefined) return "nothing";
  if (typeof value === "string") return `"${value.length > 40 ? `${value.slice(0, 40)}…` : value}"`;
  if (typeof value === "number" || typeof value === "boolean" || value === null) return String(value);
  if (Array.isArray(value)) return "a list";
  return "an object";
}

function isWhole(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function share(value: unknown, what: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isFiniteNumber(value) || value < 0 || value > 1) fail(`${what} is ${show(value)}. It should be a number from 0 to 1.`);
  return value;
}

/* ───────────────────────── validation ───────────────────────── */

interface CheckedMatch {
  title: string;
  home: string;
  away: string;
  fps: number;
  frames: number;
  pitch: { length: number; width: number };
  periods: Period[];
}

function checkTeamName(match: Record<string, unknown>, key: "home" | "away"): string {
  const team = match[key];
  const name = isObject(team) ? team.name : undefined;
  if (typeof name !== "string" || !name.trim()) {
    fail(`match.${key}.name should be the ${key} team's name. Got ${show(name)}.`);
  }
  return name.trim();
}

function checkMatch(raw: unknown): CheckedMatch {
  if (!isObject(raw)) fail('The run file has no "match" section.');
  const home = checkTeamName(raw, "home");
  const away = checkTeamName(raw, "away");

  const fps = raw.fps;
  if (!isFiniteNumber(fps) || fps <= 0 || fps > 1000) fail(`match.fps should be the frames per second, a number above 0. Got ${show(fps)}.`);

  const frames = raw.frames;
  if (!isWhole(frames) || frames <= 0) fail(`match.frames should be a whole number above 0. Got ${show(frames)}.`);
  if (frames > 20_000_000) fail(`match.frames is ${fmt(frames)}, which is more than this viewer can open.`);

  let pitch = { length: 105, width: 68 };
  if (raw.pitch !== undefined) {
    const p = raw.pitch;
    if (!isObject(p) || !isFiniteNumber(p.length) || !isFiniteNumber(p.width) || p.length <= 0 || p.width <= 0) {
      fail("match.pitch should give a length and a width in metres, both above 0.");
    }
    pitch = { length: p.length, width: p.width };
  }

  const periods: Period[] = [];
  if (raw.periods !== undefined) {
    if (!Array.isArray(raw.periods)) fail("match.periods should be a list of periods.");
    raw.periods.forEach((entry: unknown, i: number) => {
      const n = i + 1;
      if (!isObject(entry)) fail(`Period ${n} should be an object with a name, a start and an end.`);
      const name = typeof entry.name === "string" && entry.name.trim() ? entry.name.trim() : `Period ${n}`;
      const { start, end } = entry;
      if (!isWhole(start) || !isWhole(end)) fail(`Period ${n} ("${name}") needs a whole-number start and end frame.`);
      if (start < 0) fail(`Period ${n} ("${name}") starts at frame ${fmt(start)}. Frames start at 0.`);
      if (end < start) fail(`Period ${n} ("${name}") starts at frame ${fmt(start)} but ends at frame ${fmt(end)}.`);
      if (end > frames - 1) {
        fail(`Period ${n} ("${name}") ends at frame ${fmt(end)}, after the last frame of the match (${fmt(frames - 1)}).`);
      }
      const before = periods[periods.length - 1];
      if (before && start <= before.end) fail(`Period ${n} ("${name}") starts at frame ${fmt(start)}, before period ${n - 1} ends (${fmt(before.end)}).`);
      periods.push({ name, start, end });
    });
  }

  const title = typeof raw.title === "string" && raw.title.trim() ? raw.title.trim() : `${home} vs ${away}`;
  return { title, home, away, fps, frames, pitch, periods };
}

function checkLabels(raw: unknown, frames: number): LabelRun[] {
  if (raw === undefined) fail('The run file has no "labels".');
  if (!Array.isArray(raw)) fail('"labels" should be a list of runs such as ["H", 412].');
  if (raw.length === 0) fail('"labels" is empty. It should hold one run for every stretch of the match.');
  const runs: LabelRun[] = [];
  let covered = 0;
  for (let i = 0; i < raw.length; i++) {
    const entry: unknown = raw[i];
    const n = i + 1;
    if (!Array.isArray(entry) || entry.length !== 2) fail(`Label run ${fmt(n)} should look like ["H", 412]. Got ${show(entry)}.`);
    const [code, length] = entry as unknown[];
    if (!isLabelCode(code)) fail(`Label run ${fmt(n)} uses ${show(code)}. Labels can only be H, A, D or X.`);
    if (!isWhole(length) || length <= 0) fail(`Label run ${fmt(n)} has a length of ${show(length)}. Lengths are whole numbers above 0.`);
    covered += length;
    runs.push([code, length]);
  }
  if (covered !== frames) {
    fail(`The labels cover ${fmt(covered)} frames but the match has ${fmt(frames)}. Every frame needs exactly one label.`);
  }
  return mergeRuns(runs);
}

function checkSpells(raw: unknown, frames: number, index: { runStart: Int32Array; runLabel: Uint8Array }): RunFileSpell[] | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (!Array.isArray(raw)) fail('"spells" should be a list.');
  const out: RunFileSpell[] = [];
  let previousEnd = -1;
  for (let i = 0; i < raw.length; i++) {
    const entry: unknown = raw[i];
    const n = i + 1;
    if (!isObject(entry)) fail(`Spell ${fmt(n)} should be an object with a start, an end and a label.`);
    const { start, end, label } = entry;
    if (!isWhole(start) || !isWhole(end)) fail(`Spell ${fmt(n)} needs a whole-number start and end frame.`);
    if (!isLabelCode(label)) fail(`Spell ${fmt(n)} has the label ${show(label)}. Labels can only be H, A, D or X.`);
    if (start < 0 || end > frames - 1) fail(`Spell ${fmt(n)} (frames ${fmt(start)}–${fmt(end)}) reaches outside the match (0–${fmt(frames - 1)}).`);
    if (end < start) fail(`Spell ${fmt(n)} starts at frame ${fmt(start)} but ends at frame ${fmt(end)}.`);
    if (start <= previousEnd) fail(`Spell ${fmt(n)} starts at frame ${fmt(start)}, before spell ${fmt(n - 1)} ends (${fmt(previousEnd)}).`);
    previousEnd = end;

    // The spell must agree with the labels for every one of its frames.
    const want = labelOf(label);
    const r = lowerIndex(index.runStart, start);
    const runEnd = (r + 1 < index.runStart.length ? index.runStart[r + 1] : frames) - 1;
    if (index.runLabel[r] !== want) {
      fail(`Spell ${fmt(n)} (frames ${fmt(start)}–${fmt(end)}) is marked ${LABEL_WORDS[want]}, but the labels say ${LABEL_WORDS[index.runLabel[r]]} at frame ${fmt(start)}.`);
    }
    if (end > runEnd) {
      fail(`Spell ${fmt(n)} (frames ${fmt(start)}–${fmt(end)}) is marked ${LABEL_WORDS[want]}, but the labels say ${LABEL_WORDS[index.runLabel[r + 1]]} at frame ${fmt(runEnd + 1)}.`);
    }

    const spell: RunFileSpell = { start, end, label };
    const confidence = share(entry.confidence, `Spell ${fmt(n)}'s confidence`);
    if (confidence !== undefined) spell.confidence = confidence;
    const ballSeen = share(entry.ball_seen, `Spell ${fmt(n)}'s ball_seen`);
    if (ballSeen !== undefined) spell.ball_seen = ballSeen;
    if (entry.evidence !== undefined && entry.evidence !== null) {
      const e = entry.evidence;
      if (!isObject(e)) fail(`Spell ${fmt(n)}'s evidence should give ball, shape and timing.`);
      const parts = { ball: e.ball, shape: e.shape, timing: e.timing };
      for (const [key, value] of Object.entries(parts)) {
        if (!isFiniteNumber(value) || value < 0) fail(`Spell ${fmt(n)}'s evidence.${key} is ${show(value)}. It should be a number from 0 up.`);
      }
      spell.evidence = parts as { ball: number; shape: number; timing: number };
    }
    out.push(spell);
  }
  return out;
}

function checkTransitions(raw: unknown, frames: number, index: { runStart: Int32Array; runLabel: Uint8Array }): RunFileTransition[] | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (!Array.isArray(raw)) fail('"transitions" should be a list.');
  const out: RunFileTransition[] = [];
  let previous = 0;
  for (let i = 0; i < raw.length; i++) {
    const entry: unknown = raw[i];
    const n = i + 1;
    if (!isObject(entry)) fail(`Transition ${fmt(n)} should be an object with a frame, from and to.`);
    const { frame, from, to } = entry;
    if (!isWhole(frame) || frame < 1 || frame > frames - 1) {
      fail(`Transition ${fmt(n)} is at frame ${show(frame)}. It should be a whole number from 1 to ${fmt(frames - 1)}.`);
    }
    if (!isLabelCode(from) || !isLabelCode(to)) fail(`Transition ${fmt(n)} (frame ${fmt(frame)}) needs "from" and "to" labels of H, A, D or X.`);
    if (frame <= previous) fail(`Transition ${fmt(n)} (frame ${fmt(frame)}) is not after the one before it (frame ${fmt(previous)}).`);
    previous = frame;

    const r = lowerIndex(index.runStart, frame);
    const before = index.runLabel[index.runStart[r] === frame ? r - 1 : r] as Label;
    const after = index.runLabel[r] as Label;
    if (index.runStart[r] !== frame) {
      fail(`The change at frame ${fmt(frame)} is not a change: the labels stay ${LABEL_WORDS[after]} there.`);
    }
    if (before !== labelOf(from) || after !== labelOf(to)) {
      fail(
        `The change at frame ${fmt(frame)} says ${LABEL_WORDS[labelOf(from)]} to ${LABEL_WORDS[labelOf(to)]}, but the labels go from ${LABEL_WORDS[before]} to ${LABEL_WORDS[after]} there.`,
      );
    }

    const t: RunFileTransition = { frame, from, to };
    const confidence = share(entry.confidence, `The confidence of the change at frame ${fmt(frame)}`);
    if (confidence !== undefined) t.confidence = confidence;
    if (entry.why !== undefined && entry.why !== null) {
      if (typeof entry.why !== "string") fail(`The reason ("why") for the change at frame ${fmt(frame)} should be text.`);
      t.why = entry.why;
    }
    out.push(t);
  }
  return out;
}

function readPair(value: unknown): [number, number] | null | undefined {
  if (value === null) return null;
  if (!Array.isArray(value) || value.length !== 2) return undefined;
  const x: unknown = value[0];
  const y: unknown = value[1];
  if (!isFiniteNumber(x) || !isFiniteNumber(y)) return undefined;
  return [x, y];
}

function checkTracking(raw: unknown, frames: number): Tracking | null {
  if (raw === undefined || raw === null) return null;
  if (!isObject(raw)) fail('"tracking" should be an object with "every" and "samples".');
  const every = raw.every;
  if (!isWhole(every) || every <= 0) fail(`tracking.every should be a whole number of frames above 0. Got ${show(every)}.`);
  const samples = raw.samples;
  if (!Array.isArray(samples)) fail("tracking.samples should be a list.");
  if (samples.length === 0) return null;
  const needed = Math.ceil(frames / every);
  if (samples.length > needed + 1) {
    fail(`Tracking has ${fmt(samples.length)} samples, but a match of ${fmt(frames)} frames only needs ${fmt(needed)} (one every ${fmt(every)} frames).`);
  }
  const count = Math.min(samples.length, needed);

  let homeSlots = 0;
  let awaySlots = 0;
  for (let i = 0; i < count; i++) {
    const s: unknown = samples[i];
    if (!isObject(s)) fail(`Tracking sample ${fmt(i)} should be an object with "b", "h" and "a".`);
    if (!Array.isArray(s.h) || !Array.isArray(s.a)) fail(`Tracking sample ${fmt(i)} needs "h" and "a" lists of players.`);
    if (s.h.length > homeSlots) homeSlots = s.h.length;
    if (s.a.length > awaySlots) awaySlots = s.a.length;
  }
  // automatic tracking has no identities: a team of eleven can be spread over several dozen track slots
  if (homeSlots > 96 || awaySlots > 96) fail(`Tracking lists ${fmt(Math.max(homeSlots, awaySlots))} track slots for one team in a sample. That is too many to be a team.`);

  const ball = new Float32Array(count * 2).fill(NaN);
  const home = new Float32Array(count * homeSlots * 2).fill(NaN);
  const away = new Float32Array(count * awaySlots * 2).fill(NaN);
  for (let i = 0; i < count; i++) {
    const s = samples[i] as { b?: unknown; h: unknown[]; a: unknown[] };
    const b = readPair(s.b === undefined ? null : s.b);
    if (b === undefined) fail(`Tracking sample ${fmt(i)}: "b" should be [x, y] in metres, or null when the ball is not seen.`);
    if (b) {
      ball[i * 2] = b[0];
      ball[i * 2 + 1] = b[1];
    }
    for (let p = 0; p < s.h.length; p++) {
      const xy = readPair(s.h[p]);
      if (xy === undefined) fail(`Tracking sample ${fmt(i)}: home player ${p + 1} should be [x, y] in metres, or null.`);
      if (xy) {
        home[(i * homeSlots + p) * 2] = xy[0];
        home[(i * homeSlots + p) * 2 + 1] = xy[1];
      }
    }
    for (let p = 0; p < s.a.length; p++) {
      const xy = readPair(s.a[p]);
      if (xy === undefined) fail(`Tracking sample ${fmt(i)}: away player ${p + 1} should be [x, y] in metres, or null.`);
      if (xy) {
        away[(i * awaySlots + p) * 2] = xy[0];
        away[(i * awaySlots + p) * 2 + 1] = xy[1];
      }
    }
  }
  return { every, count, homeSlots, awaySlots, ball, home, away };
}

/* ───────────────────────── building ───────────────────────── */

/** Share of a stretch of frames (end inclusive) in which tracking saw the ball, or null without tracking. */
export function ballSeenShare(tracking: Tracking | null, start: number, end: number): number | null {
  if (!tracking) return null;
  const first = Math.ceil(start / tracking.every);
  const last = Math.min(tracking.count - 1, Math.floor(end / tracking.every));
  if (last < first) return null;
  let seen = 0;
  for (let i = first; i <= last; i++) if (tracking.ball[i * 2] === tracking.ball[i * 2]) seen++;
  return seen / (last - first + 1);
}

export interface RunParts {
  match: CheckedMatch;
  labels: LabelRun[];
  spells?: RunFileSpell[];
  transitions?: RunFileTransition[];
  tracking: Tracking | null;
  videoOffset: number;
  generator: { name: string; version: string } | null;
}

/** Assembles the viewer's form from parts that have already been checked. */
export function buildRun(parts: RunParts, source: RunSource, fileName: string | null): Run {
  const { match, tracking } = parts;
  const index = indexRuns(parts.labels);
  const segments: Segment[] = buildSegments(index, match.frames, parts.spells);
  for (const segment of segments) {
    if (segment.ballSeen === null) segment.ballSeen = ballSeenShare(tracking, segment.start, segment.end);
  }
  const changes = buildChanges(index, parts.transitions);
  return {
    source,
    fileName,
    generator: parts.generator,
    title: match.title,
    home: match.home,
    away: match.away,
    fps: match.fps,
    frames: match.frames,
    pitch: match.pitch,
    periods: match.periods,
    runStart: index.runStart,
    runLabel: index.runLabel,
    segments,
    segStart: Int32Array.from(segments, (s) => s.start),
    changes,
    changeFrame: Int32Array.from(changes, (c) => c.frame),
    tracking,
    videoOffset: parts.videoOffset,
    hasSpellDetail: (parts.spells?.length ?? 0) > 0,
    hasChangeDetail: (parts.transitions?.length ?? 0) > 0,
  };
}

/** Everything in a run file except tracking, checked. Throws RunFileError with a readable message. */
export function checkRunFile(json: unknown): Omit<RunParts, "tracking"> & { rawTracking: unknown } {
  if (!isObject(json)) fail("A run file is a JSON object. This file holds something else.");
  const schema = json.schema;
  if (schema === undefined) fail('This file has no "schema" field, so it is not an Excensia Vantage run file.');
  if (schema !== RUN_SCHEMA) {
    if (typeof schema === "string" && schema.startsWith("excensia.vantage.run/")) {
      fail(`This run file is in the format ${show(schema)}. This viewer reads "${RUN_SCHEMA}".`);
    }
    fail(`This file's schema is ${show(schema)}, so it is not an Excensia Vantage run file ("${RUN_SCHEMA}").`);
  }

  let generator: RunParts["generator"] = null;
  if (json.generator !== undefined && json.generator !== null) {
    const g = json.generator;
    if (!isObject(g) || typeof g.name !== "string" || typeof g.version !== "string") {
      fail('"generator" should give a name and a version, both as text.');
    }
    generator = { name: g.name, version: g.version };
  }

  const match = checkMatch(json.match);
  const labels = checkLabels(json.labels, match.frames);
  const index = indexRuns(labels);
  const spells = checkSpells(json.spells, match.frames, index);
  const transitions = checkTransitions(json.transitions, match.frames, index);

  let videoOffset = 0;
  if (json.video !== undefined && json.video !== null) {
    const v = json.video;
    if (!isObject(v)) fail('"video" should be an object such as {"offset_frames": 0}.');
    if (v.offset_frames !== undefined) {
      if (!isWhole(v.offset_frames)) fail(`video.offset_frames should be a whole number of frames. Got ${show(v.offset_frames)}.`);
      videoOffset = v.offset_frames;
    }
  }

  return { match, labels, spells, transitions, videoOffset, generator, rawTracking: json.tracking };
}

/** Opens a parsed run file. Throws RunFileError with a readable message when the file is not right. */
export function loadRunFile(json: unknown, fileName: string | null = null, source: RunSource = "file"): Run {
  const { rawTracking, ...parts } = checkRunFile(json);
  const tracking = checkTracking(rawTracking, parts.match.frames);
  return buildRun({ ...parts, tracking }, source, fileName);
}

/** Opens the text of a run file. */
export function parseRunFile(text: string, fileName: string | null = null): Run {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    const looksEmpty = text.trim().length === 0;
    fail(looksEmpty ? "This file is empty." : "This file is not JSON, so it can't be read as a run file.");
  }
  return loadRunFile(json, fileName, "file");
}

/** A match with no answers yet: what the viewer shows for a video that has no run file. */
export function blankRun(frames: number, fps: number, title: string): Run {
  const total = Math.max(1, Math.floor(frames));
  return buildRun(
    {
      match: { title, home: "Home", away: "Away", fps, frames: total, pitch: { length: 105, width: 68 }, periods: [] },
      labels: [["X", total]],
      tracking: null,
      videoOffset: 0,
      generator: null,
    },
    "blank",
    null,
  );
}

/** The run file contract, written back out from a loaded run. */
export function toRunFile(run: Run): RunFile {
  const codes = ["H", "A", "D", "X"] as const;
  const labels: LabelRun[] = [];
  for (let r = 0; r < run.runStart.length; r++) {
    const end = r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames;
    labels.push([codes[run.runLabel[r]], end - run.runStart[r]]);
  }
  const file: RunFile = {
    schema: RUN_SCHEMA,
    ...(run.generator ? { generator: run.generator } : {}),
    match: {
      title: run.title,
      home: { name: run.home },
      away: { name: run.away },
      fps: run.fps,
      frames: run.frames,
      pitch: run.pitch,
      periods: run.periods.map((p) => ({ name: p.name, start: p.start, end: p.end })),
    },
    labels,
  };
  if (run.hasSpellDetail) {
    file.spells = run.segments
      .filter((s) => s.confidence !== null || s.evidence !== null)
      .map((s) => {
        const spell: RunFileSpell = { start: s.start, end: s.end, label: codes[s.label] };
        if (s.confidence !== null) spell.confidence = s.confidence;
        if (s.evidence) spell.evidence = { ball: round2(s.evidence.ball), shape: round2(s.evidence.shape), timing: round2(s.evidence.timing) };
        if (s.ballSeen !== null) spell.ball_seen = round2(s.ballSeen);
        return spell;
      });
  }
  if (run.hasChangeDetail) {
    file.transitions = run.changes.map((c) => {
      const t: RunFileTransition = { frame: c.frame, from: codes[c.from], to: codes[c.to] };
      if (c.confidence !== null) t.confidence = c.confidence;
      if (c.why !== null) t.why = c.why;
      return t;
    });
  }
  const t = run.tracking;
  if (t) {
    const pair = (data: Float32Array, at: number): [number, number] | null =>
      data[at] === data[at] ? [round1(data[at]), round1(data[at + 1])] : null;
    const samples = new Array(t.count);
    for (let i = 0; i < t.count; i++) {
      const h = new Array(t.homeSlots);
      const a = new Array(t.awaySlots);
      for (let p = 0; p < t.homeSlots; p++) h[p] = pair(t.home, (i * t.homeSlots + p) * 2);
      for (let p = 0; p < t.awaySlots; p++) a[p] = pair(t.away, (i * t.awaySlots + p) * 2);
      samples[i] = { b: pair(t.ball, i * 2), h, a };
    }
    file.tracking = { every: t.every, samples };
  }
  file.video = { offset_frames: run.videoOffset };
  return file;
}

const round1 = (v: number) => Math.round(v * 10) / 10 + 0; // "+ 0" turns a negative zero into zero
const round2 = (v: number) => Math.round(v * 100) / 100 + 0;
