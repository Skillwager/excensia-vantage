import { describe, expect, it } from "vitest";
import {
  formatClock,
  formatClockPrecise,
  formatCount,
  formatDuration,
  formatFrameClock,
  formatMinuteMark,
  formatPercent,
  formatShares,
  formatSpan,
  formatSplit,
  formatTick,
  formatTotalClock,
  frameToSeconds,
  secondsToFrame,
  tickMinutes,
} from "./time";

describe("frames and seconds", () => {
  it("turns a frame into seconds", () => {
    expect(frameToSeconds(0, 25)).toBe(0);
    expect(frameToSeconds(108_350, 25)).toBe(4334);
    expect(frameToSeconds(30, 29.97)).toBeCloseTo(1.001, 3);
  });

  it("turns seconds into the frame showing then, clamped to the match", () => {
    expect(secondsToFrame(0, 25, 100)).toBe(0);
    expect(secondsToFrame(1, 25, 100)).toBe(25);
    expect(secondsToFrame(1.039, 25, 100)).toBe(25);
    expect(secondsToFrame(1.04, 25, 100)).toBe(26);
    expect(secondsToFrame(-3, 25, 100)).toBe(0);
    expect(secondsToFrame(999, 25, 100)).toBe(99);
    expect(secondsToFrame(4334, 25, 142_500)).toBe(108_350);
  });

  it("goes there and back for every frame of a second", () => {
    for (let f = 0; f < 60; f++) expect(secondsToFrame(frameToSeconds(f, 25), 25, 1000)).toBe(f);
    for (let f = 0; f < 60; f++) expect(secondsToFrame(frameToSeconds(f, 30), 30, 1000)).toBe(f);
  });
});

describe("the clock", () => {
  it("reads minutes and seconds, with minutes past 59", () => {
    expect(formatClock(0)).toBe("00:00");
    expect(formatClock(9)).toBe("00:09");
    expect(formatClock(60)).toBe("01:00");
    expect(formatClock(4334)).toBe("72:14");
    expect(formatClock(5400)).toBe("90:00");
    expect(formatClock(5700)).toBe("95:00");
    expect(formatClock(7265)).toBe("121:05");
  });

  it("never rounds up into the next second", () => {
    expect(formatClock(59.96)).toBe("00:59");
    expect(formatClock(59.999)).toBe("00:59");
    expect(formatClock(4334.96)).toBe("72:14");
  });

  it("holds at zero for nonsense", () => {
    expect(formatClock(-5)).toBe("00:00");
    expect(formatClock(NaN)).toBe("00:00");
  });

  it("reads a frame", () => {
    expect(formatFrameClock(108_350, 25)).toBe("72:14");
    expect(formatFrameClock(108_374, 25)).toBe("72:14");
    expect(formatFrameClock(108_375, 25)).toBe("72:15");
    expect(formatFrameClock(108_350.9, 25)).toBe("72:14");
    expect(formatFrameClock(0, 25)).toBe("00:00");
  });

  it("reads a frame to the hundredth, for stepping", () => {
    expect(formatClockPrecise(108_350, 25)).toBe("72:14.00");
    expect(formatClockPrecise(108_351, 25)).toBe("72:14.04");
    expect(formatClockPrecise(108_359, 25)).toBe("72:14.36");
    expect(formatClockPrecise(108_374, 25)).toBe("72:14.96");
    expect(formatClockPrecise(108_375, 25)).toBe("72:15.00");
    expect(formatClockPrecise(15, 30)).toBe("00:00.50");
  });

  it("reads the whole match", () => {
    expect(formatTotalClock(142_500, 25)).toBe("95:00");
    expect(formatTotalClock(135_000, 25)).toBe("90:00");
  });
});

describe("lengths of time", () => {
  it("in plain words", () => {
    expect(formatDuration(0)).toBe("0 s");
    expect(formatDuration(0.44)).toBe("0.4 s");
    expect(formatDuration(3)).toBe("3 s");
    expect(formatDuration(9.94)).toBe("9.9 s");
    expect(formatDuration(9.96)).toBe("10 s");
    expect(formatDuration(14.2)).toBe("14 s");
    expect(formatDuration(59.4)).toBe("59 s");
    expect(formatDuration(59.6)).toBe("1 min");
    expect(formatDuration(72)).toBe("1 min 12 s");
    expect(formatDuration(180)).toBe("3 min");
    expect(formatDuration(3312)).toBe("55 min 12 s");
    expect(formatDuration(-1)).toBe("0 s");
  });

  it("as a clock for tables", () => {
    expect(formatSpan(0)).toBe("0:00");
    expect(formatSpan(14.4)).toBe("0:14");
    expect(formatSpan(72)).toBe("1:12");
    expect(formatSpan(3312)).toBe("55:12");
  });
});

describe("numbers", () => {
  it("marks minutes", () => {
    expect(formatMinuteMark(0)).toBe("0'");
    expect(formatMinuteMark(45)).toBe("45'");
  });

  it("groups thousands", () => {
    expect(formatCount(142_500)).toBe("142,500");
    expect(formatCount(7)).toBe("7");
  });

  it("writes shares as whole percentages", () => {
    expect(formatPercent(0.531)).toBe("53%");
    expect(formatPercent(1)).toBe("100%");
    expect(formatPercent(NaN)).toBe("–");
  });

  it("writes several shares that always add up to 100", () => {
    expect(formatShares([44_080, 37_772, 58_417, 2_231])).toEqual(["31%", "26%", "41%", "2%"]);
    expect(formatShares([1, 1, 1])).toEqual(["34%", "33%", "33%"]);
    expect(formatShares([5, 0, 5])).toEqual(["50%", "0%", "50%"]);
    expect(formatShares([0, 0])).toEqual(["–", "–"]);
    for (const parts of [[3, 3, 3, 1], [17, 29, 31, 23], [1, 999], [333, 333, 334]]) {
      expect(formatShares(parts).reduce((sum, s) => sum + parseInt(s, 10), 0)).toBe(100);
    }
  });

  it("writes a split that always adds up to 100", () => {
    expect(formatSplit(535, 465)).toEqual(["54%", "46%"]);
    expect(formatSplit(1, 1)).toEqual(["50%", "50%"]);
    expect(formatSplit(525, 475)).toEqual(["53%", "47%"]);
    expect(formatSplit(0, 0)).toEqual(["–", "–"]);
    const [a, b] = formatSplit(0.3349, 0.6651);
    expect(parseInt(a, 10) + parseInt(b, 10)).toBe(100);
  });
});

describe("timeline ticks", () => {
  it("spaces ticks to suit the zoom", () => {
    expect(tickMinutes(5700, 1000)).toBe(15); // the whole 95 minutes across 1000 px
    expect(tickMinutes(5700, 340)).toBe(30);
    expect(tickMinutes(600, 1000)).toBe(1);
    expect(tickMinutes(60, 1000)).toBe(0.25);
  });

  it("labels whole minutes as minutes and finer ticks as a clock", () => {
    expect(formatTick(900, 15)).toBe("15'");
    expect(formatTick(4320, 1)).toBe("72'");
    expect(formatTick(4350, 0.5)).toBe("72:30");
  });
});
