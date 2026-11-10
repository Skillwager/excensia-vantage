import {
  LABEL_CODES,
  X,
  type Change,
  type Label,
  type LabelCode,
  type LabelRun,
  type Run,
  type RunFileSpell,
  type RunFileTransition,
  type Segment,
} from "./types";

export function labelOf(code: LabelCode): Label {
  return LABEL_CODES.indexOf(code) as Label;
}

export function codeOf(label: Label): LabelCode {
  return LABEL_CODES[label];
}

export function isLabelCode(value: unknown): value is LabelCode {
  return value === "H" || value === "A" || value === "D" || value === "X";
}

/** Total frames covered by a run-length list. */
export function totalFrames(runs: readonly LabelRun[]): number {
  let total = 0;
  for (const run of runs) total += run[1];
  return total;
}

/** Join neighbouring runs that carry the same label and drop empty ones. */
export function mergeRuns(runs: readonly LabelRun[]): LabelRun[] {
  const out: LabelRun[] = [];
  for (const [code, length] of runs) {
    if (length <= 0) continue;
    const last = out[out.length - 1];
    if (last && last[0] === code) last[1] += length;
    else out.push([code, length]);
  }
  return out;
}

/** One label per frame. */
export function expandLabels(runs: readonly LabelRun[]): Uint8Array {
  const out = new Uint8Array(totalFrames(runs));
  let at = 0;
  for (const [code, length] of runs) {
    out.fill(labelOf(code), at, at + length);
    at += length;
  }
  return out;
}

/** The shortest run-length list for a label-per-frame array. */
export function compressLabels(perFrame: ArrayLike<number>): LabelRun[] {
  const out: LabelRun[] = [];
  let start = 0;
  for (let i = 1; i <= perFrame.length; i++) {
    if (i === perFrame.length || perFrame[i] !== perFrame[start]) {
      if (i > start) out.push([LABEL_CODES[perFrame[start] as Label], i - start]);
      start = i;
    }
  }
  return out;
}

/**
 * Spells from labels: one per unbroken run of Home, Away or Dead ball.
 * Frames with no answer are not spells.
 */
export function deriveSpells(runs: readonly LabelRun[]): RunFileSpell[] {
  const out: RunFileSpell[] = [];
  let at = 0;
  for (const [code, length] of mergeRuns(runs)) {
    if (code !== "X") out.push({ start: at, end: at + length - 1, label: code });
    at += length;
  }
  return out;
}

/** Transitions from labels: one at every frame where the label changes. */
export function deriveTransitions(runs: readonly LabelRun[]): RunFileTransition[] {
  const out: RunFileTransition[] = [];
  const merged = mergeRuns(runs);
  let at = 0;
  for (let i = 0; i < merged.length; i++) {
    if (i > 0) out.push({ frame: at, from: merged[i - 1][0], to: merged[i][0] });
    at += merged[i][1];
  }
  return out;
}

/** Labels from spells. Frames no spell covers get no answer. Spells must be in order and not overlap. */
export function spellsToLabels(spells: readonly RunFileSpell[], frames: number): LabelRun[] {
  const out: LabelRun[] = [];
  let at = 0;
  for (const spell of spells) {
    if (spell.start < at || spell.end < spell.start || spell.end >= frames) {
      throw new RangeError(`Spell ${spell.start}–${spell.end} is out of order or outside the match.`);
    }
    if (spell.start > at) out.push(["X", spell.start - at]);
    out.push([spell.label, spell.end - spell.start + 1]);
    at = spell.end + 1;
  }
  if (at < frames) out.push(["X", frames - at]);
  return mergeRuns(out);
}

/** Labels from the first label and every change after it. */
export function transitionsToLabels(first: LabelCode, transitions: readonly RunFileTransition[], frames: number): LabelRun[] {
  const out: LabelRun[] = [];
  let at = 0;
  let current = first;
  for (const change of transitions) {
    if (change.frame <= at || change.frame >= frames) {
      throw new RangeError(`Change at frame ${change.frame} is out of order or outside the match.`);
    }
    if (change.from !== current) {
      throw new RangeError(`Change at frame ${change.frame} starts from ${change.from}, but the label before it is ${current}.`);
    }
    out.push([current, change.frame - at]);
    at = change.frame;
    current = change.to;
  }
  out.push([current, frames - at]);
  return mergeRuns(out);
}

/* ───────────────────────── look-ups on a loaded run ───────────────────────── */

/** Index of the last entry of a sorted array that is <= value (or 0). */
export function lowerIndex(sorted: Int32Array, value: number): number {
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (sorted[mid] <= value) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export function clampFrame(run: Pick<Run, "frames">, frame: number): number {
  if (!(frame > 0)) return 0;
  return frame > run.frames - 1 ? run.frames - 1 : frame;
}

export function labelAt(run: Pick<Run, "runStart" | "runLabel">, frame: number): Label {
  return run.runLabel[lowerIndex(run.runStart, frame)] as Label;
}

export function segmentIndexAt(run: Pick<Run, "segStart">, frame: number): number {
  return lowerIndex(run.segStart, frame);
}

export function segmentAt(run: Pick<Run, "segStart" | "segments">, frame: number): Segment {
  return run.segments[lowerIndex(run.segStart, frame)];
}

/** The next change strictly after this frame, or null. */
export function nextChange(run: Pick<Run, "changes" | "changeFrame">, frame: number): Change | null {
  const { changes, changeFrame } = run;
  if (!changes.length) return null;
  const f = Math.floor(frame);
  if (changeFrame[0] > f) return changes[0];
  const i = lowerIndex(changeFrame, f) + 1;
  return i < changes.length ? changes[i] : null;
}

/** The change before this frame. A playhead sitting on a change steps to the one before it. */
export function previousChange(run: Pick<Run, "changes" | "changeFrame">, frame: number): Change | null {
  const { changes, changeFrame } = run;
  if (!changes.length) return null;
  const f = Math.floor(frame) - 1;
  if (changeFrame[0] > f) return null;
  return changes[lowerIndex(changeFrame, f)];
}

/** The change closest to this frame, or null when the label never changes. */
export function nearestChange(run: Pick<Run, "changes" | "changeFrame">, frame: number): Change | null {
  const { changes, changeFrame } = run;
  if (!changes.length) return null;
  const i = lowerIndex(changeFrame, frame);
  const here = changes[i];
  const after = changes[i + 1];
  if (changeFrame[0] > frame) return changes[0];
  if (!after) return here;
  return frame - here.frame <= after.frame - frame ? here : after;
}

/* ───────────────────────── building the viewer's form ───────────────────────── */

export interface LabelIndex {
  runStart: Int32Array;
  runLabel: Uint8Array;
}

export function indexRuns(runs: readonly LabelRun[]): LabelIndex {
  const merged = mergeRuns(runs);
  const runStart = new Int32Array(merged.length);
  const runLabel = new Uint8Array(merged.length);
  let at = 0;
  for (let i = 0; i < merged.length; i++) {
    runStart[i] = at;
    runLabel[i] = labelOf(merged[i][0]);
    at += merged[i][1];
  }
  return { runStart, runLabel };
}

function normaliseEvidence(evidence: RunFileSpell["evidence"]): Segment["evidence"] {
  if (!evidence) return null;
  const sum = evidence.ball + evidence.shape + evidence.timing;
  if (!(sum > 0)) return null;
  return { ball: evidence.ball / sum, shape: evidence.shape / sum, timing: evidence.timing / sum };
}

/**
 * Segments: every label run, split wherever a spell from the file starts or ends,
 * so each segment carries one label and at most one spell's evidence.
 * Spells are expected to be valid (checked by the loader).
 */
export function buildSegments(index: LabelIndex, frames: number, spells: readonly RunFileSpell[] | undefined): Segment[] {
  const out: Segment[] = [];
  const { runStart, runLabel } = index;
  let s = 0;
  const list = spells ?? [];
  for (let r = 0; r < runStart.length; r++) {
    const runFirst = runStart[r];
    const runLast = (r + 1 < runStart.length ? runStart[r + 1] : frames) - 1;
    const label = runLabel[r] as Label;
    let at = runFirst;
    while (s < list.length && list[s].start <= runLast) {
      const spell = list[s];
      if (spell.start > at) out.push({ start: at, end: spell.start - 1, label, confidence: null, evidence: null, ballSeen: null });
      out.push({
        start: spell.start,
        end: spell.end,
        label,
        confidence: spell.confidence ?? null,
        evidence: normaliseEvidence(spell.evidence),
        ballSeen: spell.ball_seen ?? null,
      });
      at = spell.end + 1;
      s++;
    }
    if (at <= runLast) out.push({ start: at, end: runLast, label, confidence: null, evidence: null, ballSeen: null });
  }
  return out;
}

/** Every change of label, with the file's reason and confidence where it gave one. */
export function buildChanges(index: LabelIndex, transitions: readonly RunFileTransition[] | undefined): Change[] {
  const byFrame = new Map<number, RunFileTransition>();
  for (const t of transitions ?? []) byFrame.set(t.frame, t);
  const out: Change[] = [];
  for (let r = 1; r < index.runStart.length; r++) {
    const frame = index.runStart[r];
    const detail = byFrame.get(frame);
    out.push({
      frame,
      from: index.runLabel[r - 1] as Label,
      to: index.runLabel[r] as Label,
      confidence: detail?.confidence ?? null,
      why: detail?.why?.trim() ? detail.why.trim() : null,
    });
  }
  return out;
}

/** Run-length labels back out of a loaded run. */
export function runsOf(run: Pick<Run, "runStart" | "runLabel" | "frames">): LabelRun[] {
  const out: LabelRun[] = [];
  for (let r = 0; r < run.runStart.length; r++) {
    const end = r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames;
    out.push([LABEL_CODES[run.runLabel[r] as Label], end - run.runStart[r]]);
  }
  return out;
}

/** Frames carrying each label between two frames (end exclusive): [home, away, dead, none]. */
export function countLabels(run: Pick<Run, "runStart" | "runLabel" | "frames">, from: number, to: number): [number, number, number, number] {
  const out: [number, number, number, number] = [0, 0, 0, 0];
  const a = Math.max(0, Math.floor(from));
  const b = Math.min(run.frames, Math.floor(to));
  if (b <= a) return out;
  for (let r = lowerIndex(run.runStart, a); r < run.runStart.length; r++) {
    const start = run.runStart[r];
    if (start >= b) break;
    const end = r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames;
    out[run.runLabel[r]] += Math.min(end, b) - Math.max(start, a);
  }
  return out;
}

export { X as NO_ANSWER };
