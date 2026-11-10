import { describe, expect, it } from "vitest";
import { CSV_HEADER, exportFileName, inspectCsv, labelsCsv, labelsCsvBytes, validateExport } from "./csv";
import { demoRun } from "./demo";
import { runsOf } from "./derive";
import type { LabelRun } from "./types";

const RUNS: LabelRun[] = [
  ["X", 2],
  ["H", 3],
  ["D", 1],
  ["A", 2],
];

describe("the label export", () => {
  it("is exactly: header frame,label, one row per frame from 0, \\n line ends", () => {
    expect(labelsCsv(RUNS)).toBe("frame,label\n0,X\n1,X\n2,H\n3,H\n4,H\n5,D\n6,A\n7,A\n");
  });

  it("is those same characters as bytes, one byte each, with no BOM and no \\r", () => {
    const bytes = labelsCsvBytes(RUNS);
    const expected = "frame,label\n0,X\n1,X\n2,H\n3,H\n4,H\n5,D\n6,A\n7,A\n";
    expect(bytes).toHaveLength(expected.length);
    expect(Array.from(bytes)).toEqual(Array.from(expected, (c) => c.charCodeAt(0)));
    expect(bytes[0]).toBe(0x66); // "f", not a byte-order mark
    expect(bytes.includes(0x0d)).toBe(false);
    expect(bytes[bytes.length - 1]).toBe(0x0a);
  });

  it("is only a header for a match with no frames", () => {
    expect(labelsCsv([])).toBe(`${CSV_HEADER}\n`);
  });

  it("numbers rows straight across run boundaries, into five and six digits", () => {
    const text = labelsCsv([
      ["H", 99_998],
      ["A", 4],
    ]);
    const lines = text.split("\n");
    expect(lines).toHaveLength(1 + 100_002 + 1); // the header, the rows, and the empty piece after the last line end
    expect(lines[0]).toBe("frame,label");
    expect(lines[1]).toBe("0,H");
    expect(lines[99_998]).toBe("99997,H");
    expect(lines[99_999]).toBe("99998,A");
    expect(lines[100_002]).toBe("100001,A");
    expect(lines[100_003]).toBe("");
  });
});

describe("reading an export back", () => {
  it("accepts a well-formed file and counts its rows", () => {
    expect(inspectCsv(labelsCsv(RUNS))).toEqual({ ok: true, rows: 8, problem: null });
  });

  it("names what is wrong with a malformed one", () => {
    expect(inspectCsv("frame,label\r\n0,H\r\n").problem).toMatch(/Windows line ends/);
    expect(inspectCsv("label,frame\n0,H\n").problem).toMatch(/first line/);
    expect(inspectCsv("frame,label\n0,H").problem).toMatch(/does not end in a new line/);
    expect(inspectCsv("frame,label\n0,H\n2,H\n").problem).toMatch(/Row 1 is numbered 2/);
    expect(inspectCsv("frame,label\n0,H\n1,Q\n").problem).toMatch(/Row 1 has the label "Q"/);
    expect(inspectCsv("frame,label\n0,H\n1\n").problem).toMatch(/Row 1 has no comma/);
  });
});

describe("validating before export", () => {
  it("passes labels that cover every frame with only H, A, D and X, and hands back the file", () => {
    const report = validateExport(RUNS, 8);
    expect(report.ok).toBe(true);
    expect(report.frames).toBe(8);
    expect(report.counts).toEqual([3, 2, 1, 2]);
    expect(report.checks.map((c) => [c.id, c.ok])).toEqual([
      ["cover", true],
      ["labels", true],
      ["rows", true],
    ]);
    expect(report.checks[0].text).toBe("Every frame has a label: 8 of 8.");
    expect(report.csv).toBe(labelsCsv(RUNS));
  });

  it("fails when frames are missing, and writes no file", () => {
    const report = validateExport(RUNS, 10);
    expect(report.ok).toBe(false);
    expect(report.csv).toBeNull();
    expect(report.checks[0]).toMatchObject({ id: "cover", ok: false });
    expect(report.checks[0].text).toMatch(/2 frames have no label: the labels cover 8 of 10/);
    expect(report.checks[2].ok).toBe(false);
  });

  it("fails when the labels run past the end of the match", () => {
    const report = validateExport(RUNS, 6);
    expect(report.ok).toBe(false);
    expect(report.checks[0].text).toMatch(/2 frames past the end/);
  });

  it("fails on a label that is not H, A, D or X", () => {
    const report = validateExport(
      [
        ["H", 4],
        ["Q", 3],
        ["home", 1],
      ],
      8,
    );
    expect(report.ok).toBe(false);
    expect(report.checks[0].ok).toBe(true);
    expect(report.checks[1]).toMatchObject({ id: "labels", ok: false });
    expect(report.checks[1].text).toMatch(/4 frames carry a label the export does not allow: "Q", "home"/);
    expect(report.csv).toBeNull();
  });

  it("fails on a run with no length", () => {
    const report = validateExport(
      [
        ["H", 4],
        ["A", 0],
        ["D", 4],
      ],
      8,
    );
    expect(report.ok).toBe(false);
    expect(report.checks[0].text).toMatch(/1 label run has no length/);
  });

  it("fails on an empty match", () => {
    expect(validateExport([], 0).ok).toBe(false);
  });

  it("passes the whole simulated match, row for row", () => {
    const { run } = demoRun();
    const report = validateExport(runsOf(run), run.frames);
    expect(report.ok).toBe(true);
    expect(report.counts.reduce((s, v) => s + v, 0)).toBe(run.frames);
    const csv = report.csv!;
    expect(csv.startsWith("frame,label\n0,X\n1,X\n")).toBe(true);
    expect(csv.endsWith(`\n${run.frames - 1},${["H", "A", "D", "X"][run.runLabel[run.runLabel.length - 1]]}\n`)).toBe(true);
    expect(inspectCsv(csv)).toEqual({ ok: true, rows: run.frames, problem: null });
    expect(csv.includes("\r")).toBe(false);
  });
});

describe("the export's file name", () => {
  it("comes from the match title", () => {
    expect(exportFileName("Home vs Away")).toBe("home-vs-away-possession.csv");
    expect(exportFileName("  Élan FC — U14s (cup)  ")).toBe("elan-fc-u14s-cup-possession.csv");
    expect(exportFileName("???")).toBe("match-possession.csv");
  });
});
