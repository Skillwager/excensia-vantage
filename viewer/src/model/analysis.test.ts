import { describe, expect, it } from "vitest";
import { demoRun } from "./demo";
import { explain, labelWords } from "./explain";
import { buildInsights, turnoverThirds } from "./insights";
import { blankRun, loadRunFile } from "./loader";
import { emptyMoment, sampleMoment } from "./moment";
import { attackDir, convexHull, shapeIndex, shapeSeries, shapesAt, teamShape } from "./shape";
import { matchStats, spellsOf } from "./stats";
import { A, D, H, X, type RunFile } from "./types";

/** Two players a side and a ball that goes missing for a while: 100 frames, a sample every 5. */
function small(): RunFile {
  return {
    schema: "excensia.vantage.run/1",
    match: { home: { name: "Home" }, away: { name: "Away" }, fps: 25, frames: 100 },
    labels: [
      ["H", 40],
      ["A", 20],
      ["D", 40],
    ],
    tracking: {
      every: 5,
      samples: Array.from({ length: 20 }, (_, i) => ({
        b: i >= 6 && i <= 9 ? null : ([i * 2, 0] as [number, number]),
        h: [
          [-40, 0],
          [i, 10],
        ] as [number, number][],
        a: [
          [40, 0],
          [20 - i, -10],
        ] as [number, number][],
      })),
    },
  };
}

describe("reading tracking between its samples", () => {
  const run = loadRunFile(small());
  const moment = emptyMoment(run.tracking);

  it("gives a sample back exactly on its own frame", () => {
    sampleMoment(run.tracking, 10, run.fps, moment);
    expect(moment.has).toBe(true);
    expect(moment.home[2]).toBeCloseTo(2, 5);
    expect(moment.home[3]).toBeCloseTo(10, 5);
    expect(moment.away[2]).toBeCloseTo(18, 5);
    expect(moment.ballSeen).toBe(true);
    expect(moment.ballX).toBeCloseTo(4, 5);
  });

  it("moves players and the ball smoothly between samples", () => {
    sampleMoment(run.tracking, 12.5, run.fps, moment);
    expect(moment.home[2]).toBeCloseTo(2.5, 3);
    expect(moment.away[2]).toBeCloseTo(17.5, 3);
    expect(moment.ballX).toBeCloseTo(5, 5);
    expect(moment.home[0]).toBeCloseTo(-40, 5);
  });

  it("estimates a hidden ball between its last and next sighting, with a doubt that grows away from them", () => {
    // seen at sample 5 (x = 10) and again at sample 10 (x = 20); hidden at frames 30..49
    sampleMoment(run.tracking, 37.5, run.fps, moment);
    expect(moment.ballSeen).toBe(false);
    expect(moment.ballX).toBeCloseTo(15, 3);
    expect(moment.hiddenFor).toBeCloseTo(0.5, 5);
    expect(moment.hiddenUntil).toBeCloseTo(0.5, 5);
    const middle = moment.doubt;
    sampleMoment(run.tracking, 30, run.fps, moment);
    expect(moment.ballSeen).toBe(false);
    expect(moment.doubt).toBeLessThan(middle);
    expect(moment.trust).toBe(1);
  });

  it("holds the ball where it was last seen through a long absence, and trusts that less", () => {
    const file = small();
    file.match.frames = 1000;
    file.labels = [["H", 1000]];
    file.tracking!.samples = Array.from({ length: 200 }, (_, i) => ({ b: i < 10 || i > 150 ? ([i, 5] as [number, number]) : null, h: [[0, 0]], a: [[1, 1]] }));
    const long = loadRunFile(file);
    const m = emptyMoment(long.tracking);
    sampleMoment(long.tracking, 400, long.fps, m); // sample 80: 14 s after the last sighting
    expect(m.ballSeen).toBe(false);
    expect(m.ballX).toBeCloseTo(9, 5);
    expect(m.hiddenFor).toBeCloseTo(14.2, 3);
    expect(m.trust).toBeLessThan(0.5);
    expect(m.doubt).toBe(9);
  });

  it("does not slide a player across the pitch when a track jumps", () => {
    const file = small();
    file.tracking!.samples[4].h[1] = [-30, -30];
    const jumpy = loadRunFile(file);
    const m = emptyMoment(jumpy.tracking);
    sampleMoment(jumpy.tracking, 17, jumpy.fps, m); // between sample 3 (3, 10) and sample 4 (-30, -30)
    expect(m.home[2]).toBeCloseTo(3, 5);
    sampleMoment(jumpy.tracking, 18, jumpy.fps, m);
    expect(m.home[2]).toBeCloseTo(-30, 5);
  });

  it("moves the simulated match without a jolt from one sample to the next", () => {
    const { run } = demoRun();
    const a = emptyMoment(run.tracking);
    const b = emptyMoment(run.tracking);
    let worstPlayer = 0;
    let worstBall = 0;
    // a quarter of a frame at a time through open play, a restart and a goal
    for (const from of [15_000, 47_300, 108_200]) {
      sampleMoment(run.tracking, from, run.fps, a);
      for (let frame = from + 0.25; frame < from + 400; frame += 0.25) {
        sampleMoment(run.tracking, frame, run.fps, b);
        for (const team of ["home", "away"] as const) {
          for (let p = 0; p < 11; p++) {
            worstPlayer = Math.max(worstPlayer, Math.hypot(b[team][p * 2] - a[team][p * 2], b[team][p * 2 + 1] - a[team][p * 2 + 1]));
          }
        }
        if (a.ballSeen && b.ballSeen) worstBall = Math.max(worstBall, Math.hypot(b.ballX - a.ballX, b.ballY - a.ballY));
        a.home.set(b.home);
        a.away.set(b.away);
        a.ballSeen = b.ballSeen;
        a.ballX = b.ballX;
        a.ballY = b.ballY;
      }
    }
    // a quarter of a frame is a hundredth of a second: a sprinter covers about 10 cm, a hard shot about 35 cm
    expect(worstPlayer).toBeLessThan(0.14);
    expect(worstBall).toBeLessThan(0.4);
  });

  it("says so when a run has no tracking, or the frame is past its end", () => {
    const none = blankRun(100, 25, "Clip");
    expect(sampleMoment(none.tracking, 10, 25, emptyMoment(null)).has).toBe(false);
    const file = small();
    file.tracking!.samples.length = 8;
    const short = loadRunFile(file);
    const m = emptyMoment(short.tracking);
    expect(sampleMoment(short.tracking, 34, 25, m).has).toBe(true);
    expect(sampleMoment(short.tracking, 80, 25, m).has).toBe(false);
    expect(Number.isNaN(m.home[0])).toBe(true);
  });
});

describe("team shape", () => {
  it("finds the convex hull", () => {
    const hull = convexHull([0, 0, 4, 0, 4, 4, 0, 4, 2, 2, 1, 3]);
    expect(hull).toHaveLength(8);
    expect(hull).toEqual([0, 0, 4, 0, 4, 4, 0, 4]);
  });

  it("measures width, depth and the defensive line from the outfield players", () => {
    // keeper at slot 0; attacking towards +x
    const positions = new Float32Array([-50, 0, -30, -20, -28, -6, -29, 7, -31, 21, -10, 0, 5, -15, 6, 16]);
    const shape = teamShape(positions, 0, 1, 105)!;
    expect(shape.count).toBe(7);
    expect(shape.width).toBeCloseTo(41, 5);
    expect(shape.depth).toBeCloseTo(37, 5);
    expect(shape.lineX).toBeCloseTo(-29.5, 5);
    expect(shape.lineHeight).toBeCloseTo(23, 5);
    // attacking the other way, the deepest four are the ones nearest +x
    const other = teamShape(positions, 0, -1, 105)!;
    expect(other.lineX).toBeCloseTo((6 + 5 - 10 - 28) / 4, 5);
  });

  it("needs at least three outfield players", () => {
    expect(teamShape(new Float32Array([0, 0, 1, 1, NaN, NaN]), -1, 1, 105)).toBeNull();
  });

  const { run } = demoRun();
  const index = shapeIndex(run)!;

  it("works out the goalkeepers from the tracking alone", () => {
    expect(index.keepers).toEqual([0, 0]);
  });

  it("works out which way each team attacks, and that they change ends at the break", () => {
    const early = 10 * 60 * 25;
    const late = 80 * 60 * 25;
    expect(attackDir(index, 0, early)).toBe(1);
    expect(attackDir(index, 1, early)).toBe(-1);
    expect(attackDir(index, 0, late)).toBe(-1);
    expect(attackDir(index, 1, late)).toBe(1);
  });

  it("gives believable shapes: the side with the ball wider than the side without", () => {
    const moment = emptyMoment(run.tracking);
    let wider = 0;
    let looked = 0;
    for (const s of spellsOf(run)) {
      if ((s.label !== H && s.label !== A) || s.seconds < 14) continue;
      const frame = s.start + Math.round(10 * run.fps);
      sampleMoment(run.tracking, frame, run.fps, moment);
      const pair = shapesAt(run, moment, frame);
      const mine = s.label === H ? pair.home! : pair.away!;
      const theirs = s.label === H ? pair.away! : pair.home!;
      looked++;
      if (mine.width > theirs.width) wider++;
      expect(mine.lineHeight).toBeGreaterThan(5);
      expect(mine.lineHeight).toBeLessThan(80);
    }
    expect(looked).toBeGreaterThan(40);
    expect(wider / looked).toBeGreaterThan(0.8);
  });

  it("charts a measure across a window, second by second", () => {
    const series = shapeSeries(run, "width", 25 * 600, 25 * 660, 1);
    expect(series.t).toHaveLength(61);
    expect(series.t[0]).toBe(600);
    expect(series.t[60]).toBe(660);
    for (let i = 0; i < 61; i++) {
      expect(series.home[i]).toBeGreaterThan(10);
      expect(series.home[i]).toBeLessThan(68.5);
    }
  });
});

describe("the match in numbers", () => {
  const run = loadRunFile({
    schema: "excensia.vantage.run/1",
    match: {
      home: { name: "Home" },
      away: { name: "Away" },
      fps: 10,
      frames: 1000,
      periods: [
        { name: "1st half", start: 0, end: 499 },
        { name: "2nd half", start: 500, end: 999 },
      ],
    },
    labels: [
      ["H", 300],
      ["A", 100],
      ["D", 100],
      ["A", 250],
      ["H", 50],
      ["X", 100],
      ["H", 100],
    ],
  });
  const stats = matchStats(run, 0.5);

  it("counts frames and shares", () => {
    expect(stats.frames).toEqual([450, 350, 100, 100]);
    expect(stats.liveFrames).toBe(800);
    expect(stats.homeShare).toBeCloseTo(0.5625, 6);
    expect(stats.inPlayShare).toBeCloseTo(800 / 900, 6);
  });

  it("counts spells, their average and the longest", () => {
    expect(stats.home.spells).toBe(3);
    expect(stats.home.meanSeconds).toBeCloseTo(15, 6);
    expect(stats.home.medianSeconds).toBeCloseTo(10, 6);
    expect(stats.home.longest).toMatchObject({ start: 0, end: 299, seconds: 30 });
    expect(stats.away.spells).toBe(2);
    expect(stats.away.longest).toMatchObject({ start: 500, seconds: 25 });
    expect(stats.dead).toMatchObject({ spells: 1, frames: 100, meanSeconds: 10 });
    expect(stats.longestSpells.map((s) => s.seconds)).toEqual([30, 25, 10, 10, 5]);
  });

  it("counts changes and straight turnovers", () => {
    expect(stats.changes).toBe(6);
    expect(stats.turnovers).toBe(2);
  });

  it("splits the match into bands and periods", () => {
    // 30 s bands over 100 s: the last 10 s are too short to stand alone, so they join the third band
    expect(stats.bands.map((b) => [b.start, b.end])).toEqual([
      [0, 300],
      [300, 600],
      [600, 1000],
    ]);
    expect(stats.bands[0].counts).toEqual([300, 0, 0, 0]);
    expect(stats.bands[1].counts).toEqual([0, 200, 100, 0]);
    expect(stats.bands[2].counts).toEqual([150, 150, 0, 100]);
    expect(stats.bands.reduce((s, b) => s + b.counts.reduce((x, y) => x + y, 0), 0)).toBe(1000);
    expect(stats.periods[0].counts).toEqual([300, 100, 100, 0]);
    expect(stats.periods[1].counts).toEqual([150, 250, 0, 100]);
  });
});

describe("findings written from the run", () => {
  const { run } = demoRun();
  const stats = matchStats(run);
  const insights = buildInsights(run, stats);

  it("writes four to six of them, each pointing at a frame of the match", () => {
    expect(insights.length).toBeGreaterThanOrEqual(4);
    expect(insights.length).toBeLessThanOrEqual(6);
    const ids = insights.map((i) => i.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const insight of insights) {
      expect(insight.frame).toBeGreaterThanOrEqual(0);
      expect(insight.frame).toBeLessThan(run.frames);
      expect(insight.headline.length).toBeGreaterThan(20);
      expect(insight.headline).not.toMatch(/NaN|undefined|null/);
      expect(insight.detail).not.toMatch(/NaN|undefined|null/);
      expect(insight.linkLabel.length).toBeGreaterThan(3);
    }
  });

  it("says who had more of the ball, and who had it after the break", () => {
    const share = insights.find((i) => i.id === "share")!;
    expect(share.headline).toMatch(/^Home had more of the ball: \d\d% of the time it was in play\.$/);
    const afterBreak = insights.find((i) => i.id === "break")!;
    expect(afterBreak.headline).toMatch(/^Away held the ball longer after the break: \d\d%, up from \d\d%\.$/);
    expect(afterBreak.frame).toBe(72_000);
  });

  it("says where the ball was lost, from where the ball was at each turnover", () => {
    const thirds = turnoverThirds(run)!;
    const placed = thirds.counts[0].concat(thirds.counts[1]).reduce((s, v) => s + v, 0);
    expect(placed).toBeGreaterThan(stats.turnovers * 0.8);
    expect(placed).toBeLessThanOrEqual(stats.turnovers);
    const finding = insights.find((i) => i.id === "turnovers")!;
    expect(finding.headline).toMatch(/lost the ball most often in (their own third|the middle third|the final third): \d+ of \d+ times in open play\./);
  });

  it("points at the longest spell, the longest stoppage and the least certain call", () => {
    const longest = insights.find((i) => i.id === "longest")!;
    expect(longest.frame).toBe(stats.longestSpells[0].start);
    const stoppage = insights.find((i) => i.id === "stoppage")!;
    expect(stoppage.headline).toMatch(/^The longest stoppage lasted /);
    const unsure = insights.find((i) => i.id === "unsure")!;
    expect(unsure.headline).toMatch(/^The model was least sure at \d\d:\d\d: \d\d% confident/);
  });

  it("has nothing to say about a match with no answers", () => {
    const blank = blankRun(1000, 25, "Clip");
    expect(buildInsights(blank, matchStats(blank))).toEqual([]);
  });

  it("still finds things without tracking or spells", () => {
    const file = {
      schema: "excensia.vantage.run/1",
      match: { home: { name: "Reds" }, away: { name: "Blues" }, fps: 25, frames: 142_500 },
      labels: Array.from({ length: 300 }, (_, i) => [i % 3 === 2 ? "D" : i % 3 === 0 ? "H" : "A", 475]),
    };
    const plain = loadRunFile(file);
    const found = buildInsights(plain, matchStats(plain));
    expect(found.length).toBeGreaterThanOrEqual(4);
    expect(found.map((i) => i.id)).toContain("turnovers");
    expect(found.find((i) => i.id === "turnovers")!.kicker).toBe("When the ball changed sides");
    expect(found.map((i) => i.id)).toContain("tempo");
  });
});

describe("explaining a call", () => {
  const { run } = demoRun();
  const moment = emptyMoment(run.tracking);
  const at = (frame: number) => {
    sampleMoment(run.tracking, frame, run.fps, moment);
    const now = shapesAt(run, moment, frame);
    const earlier = emptyMoment(run.tracking);
    sampleMoment(run.tracking, Math.max(0, frame - 75), run.fps, earlier);
    return explain(run, frame, moment, now, shapesAt(run, earlier, Math.max(0, frame - 75)));
  };

  it("names the call", () => {
    expect(labelWords(run, H)).toBe("Home");
    expect(labelWords(run, A)).toBe("Away");
    expect(labelWords(run, D)).toBe("Dead ball");
    expect(labelWords(run, X)).toBe("No answer");
  });

  it("explains the hidden ball at 72:14 in plain words", () => {
    const text = at((72 * 60 + 14) * 25);
    expect(text.estimate).toBe("Home");
    expect(text.ball).toMatch(/^The ball is hidden behind players\. Home keep it: .* and nobody from Away has controlled it\./);
    expect(text.shape).toMatch(/^Home are spread \d+ m wide with their back line \d+ m from their own goal/);
    expect(text.timing).toMatch(/^Home are \d+ s into a spell of \d+ s\. It began at 72:0\d: /);
    expect(text.timing).toMatch(/The next change is at 72:33\.$/);
  });

  it("explains a dead ball and a moment with no answer", () => {
    const goal = at(47_500 + 250);
    expect(goal.estimate).toBe("Dead ball");
    expect(goal.ball).toMatch(/not in play|out of play/);
    expect(goal.timing).toMatch(/^The ball has been dead for 10 s of .* It began at 31:40: Goal for Home\./);
    const breakTime = at(71_000);
    expect(breakTime.estimate).toBe("No answer");
    expect(breakTime.ball).toMatch(/outside the periods of play/);
  });

  it("never leaks a broken value into a sentence", () => {
    for (let frame = 0; frame < run.frames; frame += 1777) {
      const text = at(frame);
      for (const sentence of [text.ball, text.shape, text.timing]) {
        expect(sentence).not.toMatch(/NaN|undefined|null|Infinity/);
        expect(sentence.length).toBeGreaterThan(15);
      }
    }
  });

  it("is honest when there is no tracking", () => {
    const plain = loadRunFile({
      schema: "excensia.vantage.run/1",
      match: { home: { name: "Reds" }, away: { name: "Blues" }, fps: 25, frames: 500 },
      labels: [
        ["H", 250],
        ["A", 250],
      ],
    });
    const m = emptyMoment(null);
    sampleMoment(null, 100, 25, m);
    const text = explain(plain, 100, m, { home: null, away: null }, null);
    expect(text.estimate).toBe("Reds");
    expect(text.ball).toMatch(/no tracking/);
    expect(text.shape).toMatch(/no player tracking/);
    expect(text.timing).toBe("Reds are 4 s into a spell of 10 s. The next change is at 00:10.");
  });
});
