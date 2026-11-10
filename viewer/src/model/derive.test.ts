import { describe, expect, it } from "vitest";
import {
  buildChanges,
  buildSegments,
  compressLabels,
  countLabels,
  deriveSpells,
  deriveTransitions,
  expandLabels,
  indexRuns,
  labelAt,
  mergeRuns,
  nearestChange,
  nextChange,
  previousChange,
  runsOf,
  segmentAt,
  spellsToLabels,
  totalFrames,
  transitionsToLabels,
} from "./derive";
import { loadRunFile } from "./loader";
import { A, D, H, X, type LabelRun } from "./types";

const LABELS: LabelRun[] = [
  ["X", 120],
  ["H", 412],
  ["D", 60],
  ["A", 8],
  ["H", 100],
  ["X", 20],
  ["D", 30],
];
const FRAMES = 750;

function run(labels: LabelRun[] = LABELS) {
  return loadRunFile({
    schema: "excensia.vantage.run/1",
    match: { home: { name: "Home" }, away: { name: "Away" }, fps: 25, frames: totalFrames(labels) },
    labels,
  });
}

describe("run-length labels", () => {
  it("add up", () => {
    expect(totalFrames(LABELS)).toBe(FRAMES);
  });

  it("join neighbouring runs of one label and drop empty runs", () => {
    expect(
      mergeRuns([
        ["H", 3],
        ["H", 2],
        ["D", 0],
        ["A", 1],
        ["A", 4],
        ["H", 1],
      ]),
    ).toEqual([
      ["H", 5],
      ["A", 5],
      ["H", 1],
    ]);
  });

  it("expand to one label per frame and compress back", () => {
    const perFrame = expandLabels(LABELS);
    expect(perFrame).toHaveLength(FRAMES);
    expect(perFrame[0]).toBe(X);
    expect(perFrame[119]).toBe(X);
    expect(perFrame[120]).toBe(H);
    expect(perFrame[531]).toBe(H);
    expect(perFrame[532]).toBe(D);
    expect(perFrame[749]).toBe(D);
    expect(compressLabels(perFrame)).toEqual(LABELS);
    expect(compressLabels(new Uint8Array(0))).toEqual([]);
  });
});

describe("labels → spells → labels", () => {
  it("derives one spell per run of Home, Away or Dead ball, as in the contract example", () => {
    const spells = deriveSpells(LABELS);
    expect(spells[0]).toEqual({ start: 120, end: 531, label: "H" });
    expect(spells.map((s) => [s.start, s.end, s.label])).toEqual([
      [120, 531, "H"],
      [532, 591, "D"],
      [592, 599, "A"],
      [600, 699, "H"],
      [720, 749, "D"],
    ]);
  });

  it("gets the labels back from the spells, with no answer wherever no spell reaches", () => {
    expect(spellsToLabels(deriveSpells(LABELS), FRAMES)).toEqual(LABELS);
  });

  it("refuses spells that overlap or run past the end", () => {
    expect(() =>
      spellsToLabels(
        [
          { start: 0, end: 10, label: "H" },
          { start: 10, end: 20, label: "A" },
        ],
        30,
      ),
    ).toThrow(RangeError);
    expect(() => spellsToLabels([{ start: 0, end: 30, label: "H" }], 30)).toThrow(RangeError);
  });
});

describe("labels → transitions → labels", () => {
  it("derives a transition at every change of label, as in the contract example", () => {
    const transitions = deriveTransitions(LABELS);
    expect(transitions).toEqual([
      { frame: 120, from: "X", to: "H" },
      { frame: 532, from: "H", to: "D" },
      { frame: 592, from: "D", to: "A" },
      { frame: 600, from: "A", to: "H" },
      { frame: 700, from: "H", to: "X" },
      { frame: 720, from: "X", to: "D" },
    ]);
  });

  it("gets the labels back from the first label and the transitions", () => {
    expect(transitionsToLabels("X", deriveTransitions(LABELS), FRAMES)).toEqual(LABELS);
    expect(transitionsToLabels("H", [], 40)).toEqual([["H", 40]]);
  });

  it("refuses transitions that do not follow on", () => {
    expect(() => transitionsToLabels("H", [{ frame: 5, from: "A", to: "D" }], 10)).toThrow(/starts from A/);
    expect(() =>
      transitionsToLabels(
        "H",
        [
          { frame: 5, from: "H", to: "D" },
          { frame: 5, from: "D", to: "A" },
        ],
        10,
      ),
    ).toThrow(RangeError);
  });

  it("makes no transition where neighbouring runs carry the same label", () => {
    expect(
      deriveTransitions([
        ["H", 5],
        ["H", 5],
        ["A", 5],
      ]),
    ).toEqual([{ frame: 10, from: "H", to: "A" }]);
  });
});

describe("spells ↔ transitions agree", () => {
  it("every spell starts at a transition (or frame 0) and ends just before one (or at the last frame)", () => {
    const starts = new Set([0, ...deriveTransitions(LABELS).map((t) => t.frame)]);
    const ends = new Set([FRAMES - 1, ...deriveTransitions(LABELS).map((t) => t.frame - 1)]);
    for (const spell of deriveSpells(LABELS)) {
      expect(starts.has(spell.start)).toBe(true);
      expect(ends.has(spell.end)).toBe(true);
    }
  });
});

describe("looking things up in a loaded run", () => {
  const r = run();

  it("finds the label at any frame", () => {
    expect(labelAt(r, 0)).toBe(X);
    expect(labelAt(r, 119)).toBe(X);
    expect(labelAt(r, 120)).toBe(H);
    expect(labelAt(r, 531.9)).toBe(H);
    expect(labelAt(r, 532)).toBe(D);
    expect(labelAt(r, 599)).toBe(A);
    expect(labelAt(r, 749)).toBe(D);
  });

  it("finds the segment at a frame", () => {
    expect(segmentAt(r, 300)).toMatchObject({ start: 120, end: 531, label: H });
    expect(segmentAt(r, 592)).toMatchObject({ start: 592, end: 599, label: A });
  });

  it("steps to the next and the previous change", () => {
    expect(nextChange(r, 0)!.frame).toBe(120);
    expect(nextChange(r, 119.4)!.frame).toBe(120);
    expect(nextChange(r, 120)!.frame).toBe(532);
    expect(nextChange(r, 720)).toBeNull();
    expect(previousChange(r, 0)).toBeNull();
    expect(previousChange(r, 120)).toBeNull();
    expect(previousChange(r, 121)!.frame).toBe(120);
    expect(previousChange(r, 532)!.frame).toBe(120);
    expect(previousChange(r, 749)!.frame).toBe(720);
  });

  it("finds the nearest change", () => {
    expect(nearestChange(r, 0)!.frame).toBe(120);
    expect(nearestChange(r, 300)!.frame).toBe(120);
    expect(nearestChange(r, 400)!.frame).toBe(532);
    expect(nearestChange(r, 596)!.frame).toBe(592);
    expect(nearestChange(r, 597)!.frame).toBe(600);
    expect(nearestChange(r, 749)!.frame).toBe(720);
    expect(nearestChange(run([["H", 50]]), 10)).toBeNull();
  });

  it("counts frames per label in a window", () => {
    expect(countLabels(r, 0, FRAMES)).toEqual([512, 8, 90, 140]);
    expect(countLabels(r, 100, 140)).toEqual([20, 0, 0, 20]);
    expect(countLabels(r, 590, 602)).toEqual([2, 8, 2, 0]);
    expect(countLabels(r, 400, 400)).toEqual([0, 0, 0, 0]);
    expect(countLabels(r, -50, 5000)).toEqual([512, 8, 90, 140]);
  });

  it("gives the run-length labels back", () => {
    expect(runsOf(r)).toEqual(LABELS);
  });
});

describe("segments and changes", () => {
  it("split a label run where the file's spells split it", () => {
    const index = indexRuns([
      ["H", 100],
      ["A", 50],
    ]);
    const segments = buildSegments(index, 150, [
      { start: 10, end: 39, label: "H", confidence: 0.9 },
      { start: 40, end: 99, label: "H", confidence: 0.5, evidence: { ball: 1, shape: 2, timing: 1 }, ball_seen: 0.2 },
      { start: 100, end: 149, label: "A" },
    ]);
    expect(segments.map((s) => [s.start, s.end, s.label, s.confidence])).toEqual([
      [0, 9, H, null],
      [10, 39, H, 0.9],
      [40, 99, H, 0.5],
      [100, 149, A, null],
    ]);
    expect(segments[2].evidence).toEqual({ ball: 0.25, shape: 0.5, timing: 0.25 });
    expect(segments[2].ballSeen).toBe(0.2);
  });

  it("cover every frame exactly once", () => {
    const r = run();
    let at = 0;
    for (const s of r.segments) {
      expect(s.start).toBe(at);
      expect(s.end).toBeGreaterThanOrEqual(s.start);
      at = s.end + 1;
    }
    expect(at).toBe(FRAMES);
  });

  it("list every change, with the file's reason where it gave one", () => {
    const index = indexRuns(LABELS);
    const changes = buildChanges(index, [{ frame: 532, from: "H", to: "D", confidence: 0.81, why: " Ball left the pitch. " }]);
    expect(changes).toHaveLength(6);
    expect(changes[1]).toEqual({ frame: 532, from: H, to: D, confidence: 0.81, why: "Ball left the pitch." });
    expect(changes[0]).toEqual({ frame: 120, from: X, to: H, confidence: null, why: null });
  });
});
