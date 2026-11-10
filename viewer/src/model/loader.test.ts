import { describe, expect, it } from "vitest";
import { blankRun, loadRunFile, parseRunFile, RunFileError, toRunFile } from "./loader";
import { runsOf } from "./derive";
import { A, D, H, X, type RunFile } from "./types";

/** A small, valid run file: 100 frames at 25 fps. */
function sample(): RunFile {
  return {
    schema: "excensia.vantage.run/1",
    generator: { name: "excensia-vantage", version: "0.1.0" },
    match: {
      title: "Home vs Away",
      home: { name: "Home" },
      away: { name: "Away" },
      fps: 25,
      frames: 100,
      pitch: { length: 105, width: 68 },
      periods: [
        { name: "1st half", start: 0, end: 49 },
        { name: "2nd half", start: 60, end: 99 },
      ],
    },
    labels: [
      ["X", 10],
      ["H", 30],
      ["D", 20],
      ["A", 40],
    ],
    spells: [
      { start: 10, end: 39, label: "H", confidence: 0.93, evidence: { ball: 0.7, shape: 0.2, timing: 0.1 }, ball_seen: 0.64 },
      { start: 40, end: 59, label: "D", confidence: 0.9 },
    ],
    transitions: [
      { frame: 10, from: "X", to: "H", confidence: 0.97, why: "Home kicked off." },
      { frame: 40, from: "H", to: "D", confidence: 0.81, why: "Ball left the pitch on the right touchline." },
    ],
    tracking: {
      every: 5,
      samples: Array.from({ length: 20 }, (_, i) => ({
        b: i % 4 === 3 ? null : ([i * 1.5, -3.4] as [number, number]),
        h: [
          [-40.1, 0.2],
          [-20 + i, 5],
        ] as [number, number][],
        a: [
          [38, 1.1],
          [20 - i, -5],
        ] as [number, number][],
      })),
    },
    video: { offset_frames: 12 },
  };
}

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/** Loads a file after a change and returns the message it is refused with. */
function refusal(change: (file: any) => void): string {
  const file = clone(sample());
  change(file);
  try {
    loadRunFile(file);
  } catch (error) {
    expect(error).toBeInstanceOf(RunFileError);
    return (error as Error).message;
  }
  throw new Error("The file was accepted.");
}

describe("loading a run file", () => {
  it("opens a valid file", () => {
    const run = loadRunFile(sample(), "match.json");
    expect(run.source).toBe("file");
    expect(run.fileName).toBe("match.json");
    expect(run.title).toBe("Home vs Away");
    expect(run.home).toBe("Home");
    expect(run.away).toBe("Away");
    expect(run.fps).toBe(25);
    expect(run.frames).toBe(100);
    expect(run.pitch).toEqual({ length: 105, width: 68 });
    expect(run.periods).toEqual([
      { name: "1st half", start: 0, end: 49 },
      { name: "2nd half", start: 60, end: 99 },
    ]);
    expect(Array.from(run.runStart)).toEqual([0, 10, 40, 60]);
    expect(Array.from(run.runLabel)).toEqual([X, H, D, A]);
    expect(run.videoOffset).toBe(12);
    expect(run.generator).toEqual({ name: "excensia-vantage", version: "0.1.0" });
    expect(run.hasSpellDetail).toBe(true);
    expect(run.hasChangeDetail).toBe(true);
  });

  it("keeps the evidence from spells and the reasons from transitions", () => {
    const run = loadRunFile(sample());
    const home = run.segments.find((s) => s.label === H)!;
    expect(home.start).toBe(10);
    expect(home.end).toBe(39);
    expect(home.confidence).toBe(0.93);
    expect(home.ballSeen).toBe(0.64);
    expect(home.evidence!.ball).toBeCloseTo(0.7, 6);
    expect(home.evidence!.shape + home.evidence!.ball + home.evidence!.timing).toBeCloseTo(1, 6);
    expect(run.changes.map((c) => c.frame)).toEqual([10, 40, 60]);
    expect(run.changes[1].why).toBe("Ball left the pitch on the right touchline.");
    expect(run.changes[1].confidence).toBe(0.81);
    // the third change was not in the file: it is derived, with no reason
    expect(run.changes[2]).toEqual({ frame: 60, from: D, to: A, confidence: null, why: null });
  });

  it("reads tracking into compact arrays, with gaps where the ball or a player is not seen", () => {
    const file = sample();
    file.tracking!.samples[2].h[1] = null;
    const run = loadRunFile(file);
    const t = run.tracking!;
    expect(t.every).toBe(5);
    expect(t.count).toBe(20);
    expect(t.homeSlots).toBe(2);
    expect(t.awaySlots).toBe(2);
    expect(t.ball[0]).toBeCloseTo(0, 5);
    expect(t.ball[2]).toBeCloseTo(1.5, 5);
    expect(Number.isNaN(t.ball[3 * 2])).toBe(true);
    expect(t.home[0]).toBeCloseTo(-40.1, 4);
    expect(Number.isNaN(t.home[(2 * 2 + 1) * 2])).toBe(true);
    expect(t.away[(19 * 2 + 1) * 2]).toBeCloseTo(1, 5);
  });

  it("fills in the share of each spell with the ball in view when the file does not give it", () => {
    const run = loadRunFile(sample());
    const dead = run.segments.find((s) => s.label === D)!;
    // samples 8..11 cover frames 40..59; sample 11 has no ball
    expect(dead.ballSeen).toBeCloseTo(0.75, 6);
  });

  it("derives spells and transitions from labels when the file has none", () => {
    const file = sample();
    delete file.spells;
    delete file.transitions;
    const run = loadRunFile(file);
    expect(run.hasSpellDetail).toBe(false);
    expect(run.hasChangeDetail).toBe(false);
    expect(run.segments.map((s) => [s.start, s.end, s.label])).toEqual([
      [0, 9, X],
      [10, 39, H],
      [40, 59, D],
      [60, 99, A],
    ]);
    expect(run.segments.every((s) => s.confidence === null && s.evidence === null)).toBe(true);
    expect(run.changes.map((c) => [c.frame, c.from, c.to, c.why])).toEqual([
      [10, X, H, null],
      [40, H, D, null],
      [60, D, A, null],
    ]);
  });

  it("opens a file with only the required parts", () => {
    const run = loadRunFile({
      schema: "excensia.vantage.run/1",
      match: { home: { name: "Reds" }, away: { name: "Blues" }, fps: 30, frames: 6 },
      labels: [
        ["H", 2],
        ["H", 1],
        ["A", 3],
      ],
    });
    expect(run.title).toBe("Reds vs Blues");
    expect(run.pitch).toEqual({ length: 105, width: 68 });
    expect(run.periods).toEqual([]);
    expect(run.tracking).toBeNull();
    expect(run.videoOffset).toBe(0);
    expect(run.videoWithheld).toBeNull();
    expect(Array.from(run.runStart)).toEqual([0, 3]); // neighbouring runs of one label are joined
  });

  it("reads that a match's video is withheld, and writes it back", () => {
    const why = "The rules of this match do not allow its video beside the labels.";
    const file = { schema: "excensia.vantage.run/1", match: { home: { name: "Reds" }, away: { name: "Blues" }, fps: 25, frames: 4 }, labels: [["H", 4]], video: { offset_frames: 0, withheld: `  ${why} ` } };
    const run = loadRunFile(file);
    expect(run.videoWithheld).toBe(why);
    expect(toRunFile(run).video).toEqual({ offset_frames: 0, withheld: why });
    expect(toRunFile(loadRunFile({ ...file, video: { offset_frames: 0 } })).video).toEqual({ offset_frames: 0 });
    expect(() => loadRunFile({ ...file, video: { withheld: 7 } })).toThrow(/video\.withheld/);
    expect(() => loadRunFile({ ...file, video: { withheld: "  " } })).toThrow(/video\.withheld/);
  });

  it("ignores fields it does not know", () => {
    const file = clone(sample()) as any;
    file.notes = "anything";
    file.match.venue = "Somewhere";
    expect(() => loadRunFile(file)).not.toThrow();
  });

  it("accepts tracking that stops short of the end, and one spare sample", () => {
    const short = sample();
    short.tracking!.samples.length = 12;
    expect(loadRunFile(short).tracking!.count).toBe(12);
    const spare = sample();
    spare.tracking!.samples.push({ b: null, h: [], a: [] });
    expect(loadRunFile(spare).tracking!.count).toBe(20);
  });

  it("treats empty tracking as no tracking", () => {
    const file = sample();
    file.tracking!.samples = [];
    expect(loadRunFile(file).tracking).toBeNull();
  });

  it("reads the text of a file", () => {
    const run = parseRunFile(JSON.stringify(sample()), "a.json");
    expect(run.frames).toBe(100);
    expect(run.fileName).toBe("a.json");
  });
});

describe("refusing a bad run file, in words a person can act on", () => {
  it("text that is not JSON", () => {
    expect(() => parseRunFile("frame,label\n0,H\n")).toThrow(/not JSON/);
    expect(() => parseRunFile("   ")).toThrow(/empty/);
  });

  it("something that is not a run file", () => {
    expect(() => loadRunFile([1, 2, 3])).toThrow(/JSON object/);
    expect(() => loadRunFile(null)).toThrow(RunFileError);
    expect(refusal((f) => delete f.schema)).toMatch(/no "schema" field/);
    expect(refusal((f) => (f.schema = "something/else"))).toMatch(/not an Excensia Vantage run file/);
    expect(refusal((f) => (f.schema = "excensia.vantage.run/2"))).toMatch(/format "excensia\.vantage\.run\/2".*reads "excensia\.vantage\.run\/1"/);
  });

  it("a match section that is missing or wrong", () => {
    expect(refusal((f) => delete f.match)).toMatch(/no "match" section/);
    expect(refusal((f) => delete f.match.home)).toMatch(/match\.home\.name/);
    expect(refusal((f) => (f.match.away.name = "  "))).toMatch(/match\.away\.name/);
    expect(refusal((f) => (f.match.fps = 0))).toMatch(/match\.fps.*above 0.*Got 0/);
    expect(refusal((f) => (f.match.fps = "25"))).toMatch(/match\.fps/);
    expect(refusal((f) => (f.match.frames = 99.5))).toMatch(/match\.frames.*whole number/);
    expect(refusal((f) => (f.match.frames = -4))).toMatch(/match\.frames/);
    expect(refusal((f) => (f.match.pitch = { length: 105 }))).toMatch(/match\.pitch/);
    expect(refusal((f) => (f.generator = "engine"))).toMatch(/"generator"/);
  });

  it("periods that do not fit the match", () => {
    expect(refusal((f) => (f.match.periods[1].end = 100))).toMatch(/Period 2 \("2nd half"\) ends at frame 100, after the last frame of the match \(99\)/);
    expect(refusal((f) => (f.match.periods[1].start = 40))).toMatch(/Period 2 .* starts at frame 40, before period 1 ends \(49\)/);
    expect(refusal((f) => (f.match.periods[0] = { name: "1st half", start: 30, end: 20 }))).toMatch(/starts at frame 30 but ends at frame 20/);
    expect(refusal((f) => (f.match.periods = "two halves"))).toMatch(/match\.periods/);
  });

  it("labels that are missing, malformed or do not cover the match", () => {
    expect(refusal((f) => delete f.labels)).toMatch(/no "labels"/);
    expect(refusal((f) => (f.labels = []))).toMatch(/"labels" is empty/);
    expect(refusal((f) => (f.labels = "HHHA"))).toMatch(/"labels" should be a list/);
    expect(refusal((f) => (f.labels[2] = ["Q", 20]))).toMatch(/Label run 3 uses "Q"\. Labels can only be H, A, D or X/);
    expect(refusal((f) => (f.labels[1] = ["H", 0]))).toMatch(/Label run 2 has a length of 0/);
    expect(refusal((f) => (f.labels[1] = ["H", 2.5]))).toMatch(/Label run 2 has a length of 2\.5/);
    expect(refusal((f) => (f.labels[0] = "X10"))).toMatch(/Label run 1 should look like \["H", 412\]/);
    expect(refusal((f) => (f.labels[3] = ["A", 30]))).toMatch(/The labels cover 90 frames but the match has 100/);
    expect(refusal((f) => f.labels.push(["H", 5]))).toMatch(/The labels cover 105 frames but the match has 100/);
  });

  it("spells that disagree with the labels", () => {
    expect(refusal((f) => (f.spells[0].label = "A"))).toMatch(/Spell 1 \(frames 10–39\) is marked Away, but the labels say Home at frame 10/);
    expect(refusal((f) => (f.spells[0].end = 45))).toMatch(/Spell 1 \(frames 10–45\) is marked Home, but the labels say Dead ball at frame 40/);
    expect(refusal((f) => (f.spells[1].start = 39))).toMatch(/Spell 2 starts at frame 39, before spell 1 ends \(39\)/);
    expect(refusal((f) => (f.spells[1].end = 100))).toMatch(/reaches outside the match \(0–99\)/);
    expect(refusal((f) => (f.spells[0].confidence = 1.4))).toMatch(/Spell 1's confidence is 1\.4\. It should be a number from 0 to 1/);
    expect(refusal((f) => (f.spells[0].ball_seen = -0.1))).toMatch(/ball_seen/);
    expect(refusal((f) => (f.spells[0].evidence = { ball: 1 }))).toMatch(/evidence\.shape/);
    expect(refusal((f) => (f.spells = {}))).toMatch(/"spells" should be a list/);
  });

  it("transitions that disagree with the labels", () => {
    expect(refusal((f) => (f.transitions[1].to = "A"))).toMatch(/The change at frame 40 says Home to Away, but the labels go from Home to Dead ball there/);
    expect(refusal((f) => (f.transitions[1].frame = 41))).toMatch(/The change at frame 41 is not a change: the labels stay Dead ball there/);
    expect(refusal((f) => (f.transitions[0].frame = 0))).toMatch(/Transition 1 is at frame 0/);
    expect(refusal((f) => f.transitions.reverse())).toMatch(/Transition 2 \(frame 10\) is not after the one before it \(frame 40\)/);
    expect(refusal((f) => (f.transitions[0].why = 7))).toMatch(/reason \("why"\)/);
  });

  it("tracking that is malformed", () => {
    expect(refusal((f) => (f.tracking.every = 0))).toMatch(/tracking\.every/);
    expect(refusal((f) => (f.tracking.samples = "none"))).toMatch(/tracking\.samples should be a list/);
    expect(refusal((f) => (f.tracking.samples[3].b = [1]))).toMatch(/Tracking sample 3: "b" should be \[x, y\]/);
    expect(refusal((f) => (f.tracking.samples[4].h[1] = ["a", 2]))).toMatch(/Tracking sample 4: home player 2 should be \[x, y\]/);
    expect(refusal((f) => (f.tracking.samples[5].a = null))).toMatch(/Tracking sample 5 needs "h" and "a"/);
    expect(
      refusal((f) => {
        for (let i = 0; i < 5; i++) f.tracking.samples.push({ b: null, h: [], a: [] });
      }),
    ).toMatch(/Tracking has 25 samples, but a match of 100 frames only needs 20 \(one every 5 frames\)/);
  });

  it("a video offset that is not a whole number", () => {
    expect(refusal((f) => (f.video.offset_frames = 1.5))).toMatch(/video\.offset_frames/);
  });
});

describe("writing a run back out", () => {
  it("gives back the file it was loaded from", () => {
    const file = sample();
    const again = toRunFile(loadRunFile(file));
    expect(again.schema).toBe(file.schema);
    expect(again.match).toEqual(file.match);
    expect(again.labels).toEqual(file.labels);
    expect(again.video).toEqual(file.video);
    expect(again.tracking!.every).toBe(5);
    expect(again.tracking!.samples).toEqual(file.tracking!.samples);
    expect(again.transitions!.slice(0, 2)).toEqual(file.transitions);
    expect(again.spells![0]).toEqual(file.spells![0]);
    // and that output loads again to the same thing
    const reloaded = loadRunFile(clone(again));
    expect(runsOf(reloaded)).toEqual(file.labels);
  });
});

describe("a video with no run file yet", () => {
  it("is a match with no answers", () => {
    const run = blankRun(250.9, 25, "Clip");
    expect(run.source).toBe("blank");
    expect(run.frames).toBe(250);
    expect(runsOf(run)).toEqual([["X", 250]]);
    expect(run.changes).toEqual([]);
    expect(run.tracking).toBeNull();
    expect(run.segments).toHaveLength(1);
    expect(run.segments[0].label).toBe(X);
    expect(A).toBe(1);
  });
});
