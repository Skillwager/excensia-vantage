import { describe, expect, it } from "vitest";
import { DEMO_FRAMES, demoRun, demoRunFile, generateDemo } from "./demo";
import { runsOf, totalFrames } from "./derive";
import { loadRunFile } from "./loader";
import { matchStats } from "./stats";
import { A, D, H, X } from "./types";

const { run, scene } = demoRun();
const stats = matchStats(run);

describe("the simulated match", () => {
  it("is about 95 minutes at 25 fps, in two halves", () => {
    expect(run.fps).toBe(25);
    expect(run.frames).toBe(142_500);
    expect(run.frames).toBe(DEMO_FRAMES);
    expect(run.periods).toEqual([
      { name: "1st half", start: 0, end: 70_499 },
      { name: "2nd half", start: 72_000, end: 142_499 },
    ]);
    expect(run.source).toBe("demo");
    expect(run.home).toBe("Home");
    expect(run.away).toBe("Away");
  });

  it("is the same match every time", () => {
    const again = demoRun().run;
    expect(runsOf(again)).toEqual(runsOf(run));
    expect(again.changes).toEqual(run.changes);
    expect(again.segments).toEqual(run.segments);
    // Float32Arrays hold NaN where the ball is unseen, so compare bytes
    const bytes = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    expect(Buffer.compare(bytes(again.tracking!.ball), bytes(run.tracking!.ball))).toBe(0);
    expect(Buffer.compare(bytes(again.tracking!.home), bytes(run.tracking!.home))).toBe(0);
    expect(Buffer.compare(bytes(again.tracking!.away), bytes(run.tracking!.away))).toBe(0);
  });

  it("is a different match for a different seed", () => {
    const other = generateDemo(99);
    expect(other.parts.labels).not.toEqual(runsOf(run));
    expect(totalFrames(other.parts.labels)).toBe(DEMO_FRAMES);
  });

  it("has labels that cover every frame", () => {
    expect(totalFrames(runsOf(run))).toBe(run.frames);
    expect(stats.frames.reduce((s, v) => s + v, 0)).toBe(run.frames);
  });

  it("has possession spells of about 14 s on average, with a long tail", () => {
    const spells = stats.home.spells + stats.away.spells;
    const mean = stats.liveFrames / run.fps / spells;
    expect(mean).toBeGreaterThan(12.5);
    expect(mean).toBeLessThan(15.5);
    // the long tail: the median sits well under the mean, and the longest spell is many times it
    const median = (stats.home.medianSeconds + stats.away.medianSeconds) / 2;
    expect(median).toBeLessThan(mean * 0.82);
    expect(stats.longestSpells[0].seconds).toBeGreaterThan(mean * 4);
    expect(spells).toBeGreaterThan(180);
  });

  it("has about 40% dead-ball time", () => {
    const share = stats.frames[D] / run.frames;
    expect(share).toBeGreaterThan(0.37);
    expect(share).toBeLessThan(0.43);
  });

  it("gives no answer only for the lead-in, the break and a few loose balls", () => {
    const share = stats.frames[X] / run.frames;
    expect(share).toBeGreaterThan(0.01);
    expect(share).toBeLessThan(0.03);
    expect(run.runLabel[0]).toBe(X);
    expect(run.runStart[1]).toBe(120);
  });

  it("gives both sides a real share of the ball, with a swing after the break", () => {
    expect(stats.homeShare).toBeGreaterThan(0.42);
    expect(stats.homeShare).toBeLessThan(0.6);
    const [first, second] = stats.periods;
    const before = first.counts[H] / (first.counts[H] + first.counts[A]);
    const after = second.counts[H] / (second.counts[H] + second.counts[A]);
    expect(before).toBeGreaterThan(0.52);
    expect(after).toBeLessThan(0.49);
  });

  it("tracks 11 players a side five times a second", () => {
    const t = run.tracking!;
    expect(t.every).toBe(5);
    expect(t.count).toBe(28_500);
    expect(t.homeSlots).toBe(11);
    expect(t.awaySlots).toBe(11);
    expect(t.home.some((v) => Number.isNaN(v))).toBe(false);
    expect(t.away.some((v) => Number.isNaN(v))).toBe(false);
  });

  it("loses sight of the ball on about 40% of samples", () => {
    const t = run.tracking!;
    let unseen = 0;
    for (let i = 0; i < t.count; i++) if (Number.isNaN(t.ball[i * 2])) unseen++;
    const share = unseen / t.count;
    expect(share).toBeGreaterThan(0.36);
    expect(share).toBeLessThan(0.44);
  });

  it("moves players at speeds people can run, and keeps them by the pitch", () => {
    const t = run.tracking!;
    const dt = t.every / run.fps;
    let fastest = 0;
    let furthestX = 0;
    let furthestY = 0;
    for (const team of [t.home, t.away]) {
      for (let i = 1; i < t.count; i++) {
        for (let p = 0; p < 11; p++) {
          const a = ((i - 1) * 11 + p) * 2;
          const b = (i * 11 + p) * 2;
          const v = Math.hypot(team[b] - team[a], team[b + 1] - team[a + 1]) / dt;
          if (v > fastest) fastest = v;
          furthestX = Math.max(furthestX, Math.abs(team[b]));
          furthestY = Math.max(furthestY, Math.abs(team[b + 1]));
        }
      }
    }
    expect(fastest).toBeLessThan(10);
    expect(furthestX).toBeLessThanOrEqual(55.01);
    expect(furthestY).toBeLessThanOrEqual(36.51);
  });

  it("keeps each side's shape: a keeper by the goal and the rest spread out ahead of him", () => {
    const t = run.tracking!;
    // a quiet moment in open play, ten minutes in
    const i = Math.round((10 * 60 * run.fps) / t.every);
    const keeperX = t.home[i * 11 * 2];
    expect(Math.abs(keeperX)).toBeGreaterThan(30);
    let spreadY = 0;
    for (let p = 1; p < 11; p++) spreadY = Math.max(spreadY, Math.abs(t.home[(i * 11 + p) * 2 + 1]));
    expect(spreadY).toBeGreaterThan(8);
  });

  it("starts every half with a kick-off from the centre spot", () => {
    const t = run.tracking!;
    for (const frame of [0, 72_000]) {
      const i = frame / t.every;
      expect(Math.abs(t.ball[i * 2])).toBeLessThan(1.5);
      expect(Math.abs(t.ball[i * 2 + 1])).toBeLessThan(1.5);
      // every player in their own half
      const homeSide = Math.sign(t.home[i * 11 * 2]);
      for (let p = 0; p < 11; p++) {
        expect(t.home[(i * 11 + p) * 2] * homeSide).toBeGreaterThanOrEqual(-0.5);
        expect(t.away[(i * 11 + p) * 2] * homeSide).toBeLessThanOrEqual(0.5);
      }
    }
    // and the teams change ends at the break
    expect(Math.sign(t.home[0])).toBe(-Math.sign(t.home[(72_000 / t.every) * 11 * 2]));
  });

  it("gives every spell a confidence, evidence that adds up, and a share of ball in view", () => {
    for (const s of run.segments) {
      if (s.label === X) continue;
      expect(s.confidence).not.toBeNull();
      expect(s.confidence!).toBeGreaterThanOrEqual(0.3);
      expect(s.confidence!).toBeLessThanOrEqual(1);
      expect(s.evidence!.ball + s.evidence!.shape + s.evidence!.timing).toBeCloseTo(1, 5);
      expect(s.ballSeen!).toBeGreaterThanOrEqual(0);
      expect(s.ballSeen!).toBeLessThanOrEqual(1);
    }
    const unsure = run.segments.filter((s) => (s.label === H || s.label === A) && s.confidence! < 0.6).length;
    expect(unsure).toBeGreaterThan(8);
  });

  it("gives every change a reason", () => {
    expect(run.changes.length).toBeGreaterThan(250);
    for (const c of run.changes) {
      expect(c.why).toBeTruthy();
      expect(c.confidence).not.toBeNull();
    }
    expect(run.changes[0]).toMatchObject({ frame: 120, from: X, to: H, why: "Home kicked off." });
  });

  it("tells its story: Home lead 1–0 at 72:14 with the ball hidden, then Away level", () => {
    expect(scene.goals).toEqual([
      { frame: 47_500, side: 0 },
      { frame: 126_250, side: 1 },
    ]);
    const frame = (72 * 60 + 14) * 25;
    const segment = run.segments.find((s) => s.start <= frame && s.end >= frame)!;
    expect(segment.label).toBe(H);
    expect(segment.confidence).toBeLessThan(0.6);
    const t = run.tracking!;
    expect(Number.isNaN(t.ball[Math.round(frame / t.every) * 2])).toBe(true);
    const goal = run.changes.find((c) => c.frame === 47_500)!;
    expect(goal).toMatchObject({ from: H, to: D, why: "Goal for Home." });
  });
});

describe("the simulated match as a run file", () => {
  const file = demoRunFile();

  it("is in the contract's shape", () => {
    expect(file.schema).toBe("excensia.vantage.run/1");
    expect(file.generator).toEqual({ name: "excensia-vantage", version: "0.1.0" });
    expect(file.match).toEqual({
      title: "Home vs Away",
      home: { name: "Home" },
      away: { name: "Away" },
      fps: 25,
      frames: 142_500,
      pitch: { length: 105, width: 68 },
      periods: [
        { name: "1st half", start: 0, end: 70_499 },
        { name: "2nd half", start: 72_000, end: 142_499 },
      ],
    });
    expect(file.labels[0]).toEqual(["X", 120]);
    expect(file.video).toEqual({ offset_frames: 0 });
    expect(file.tracking!.every).toBe(5);
    expect(file.tracking!.samples).toHaveLength(28_500);
    const first = file.tracking!.samples[0];
    expect(Object.keys(first).sort()).toEqual(["a", "b", "h"]);
    expect(first.h).toHaveLength(11);
    expect(first.a).toHaveLength(11);
    expect(first.b).toEqual([0, 0]);
    expect(file.tracking!.samples.some((s) => s.b === null)).toBe(true);
    const spell = file.spells![0];
    expect(Object.keys(spell).sort()).toEqual(["ball_seen", "confidence", "end", "evidence", "label", "start"]);
    expect(Object.keys(spell.evidence!).sort()).toEqual(["ball", "shape", "timing"]);
    expect(spell.start).toBe(120);
    const change = file.transitions![0];
    expect(Object.keys(change).sort()).toEqual(["confidence", "frame", "from", "to", "why"]);
  });

  it("holds numbers to the contract's precision", () => {
    const text = JSON.stringify(file.tracking!.samples.slice(5000, 5020));
    expect(text).not.toMatch(/\.\d{2,}/); // one decimal at most
  });

  it("opens in the strict loader as the very same match", () => {
    const loaded = loadRunFile(JSON.parse(JSON.stringify(file)));
    expect(runsOf(loaded)).toEqual(runsOf(run));
    expect(loaded.changes).toEqual(run.changes);
    expect(loaded.periods).toEqual(run.periods);
    expect(loaded.segments.length).toBe(run.segments.length);
    for (let i = 0; i < run.segments.length; i += 7) {
      const a = loaded.segments[i];
      const b = run.segments[i];
      expect([a.start, a.end, a.label, a.confidence]).toEqual([b.start, b.end, b.label, b.confidence]);
    }
    const t = loaded.tracking!;
    const want = run.tracking!;
    expect(t.count).toBe(want.count);
    const bytes = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    expect(Buffer.compare(bytes(t.ball), bytes(want.ball))).toBe(0);
    expect(Buffer.compare(bytes(t.home), bytes(want.home))).toBe(0);
    expect(Buffer.compare(bytes(t.away), bytes(want.away))).toBe(0);
  });
});
