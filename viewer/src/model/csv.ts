import { LABEL_CODES, type LabelRun } from "./types";
import { isLabelCode } from "./derive";

export const CSV_HEADER = "frame,label";

/**
 * The label export, exactly:
 *   frame,label\n
 *   0,H\n
 *   1,H\n
 *   …one row per frame from 0, every line ending in \n.
 */
export function labelsCsv(runs: readonly LabelRun[]): string {
  const parts: string[] = [`${CSV_HEADER}\n`];
  let frame = 0;
  for (const [code, length] of runs) {
    const rows: string[] = new Array(Math.max(0, length));
    for (let i = 0; i < length; i++) rows[i] = `${frame + i},${code}\n`;
    parts.push(rows.join(""));
    frame += Math.max(0, length);
  }
  return parts.join("");
}

/** The export as bytes (UTF-8; the file is plain ASCII, so one byte per character). */
export function labelsCsvBytes(runs: readonly LabelRun[]): Uint8Array {
  return new TextEncoder().encode(labelsCsv(runs));
}

/** Reads a written export back, row by row, and checks its shape byte for byte. */
export function inspectCsv(text: string): { ok: boolean; rows: number; problem: string | null } {
  if (text.includes("\r")) return { ok: false, rows: 0, problem: "The file has Windows line ends." };
  if (!text.startsWith(`${CSV_HEADER}\n`)) return { ok: false, rows: 0, problem: "The first line is not frame,label." };
  if (!text.endsWith("\n")) return { ok: false, rows: 0, problem: "The last line does not end in a new line." };
  let rows = 0;
  let at = CSV_HEADER.length + 1;
  while (at < text.length) {
    const end = text.indexOf("\n", at);
    const comma = text.indexOf(",", at);
    if (comma < 0 || comma > end) return { ok: false, rows, problem: `Row ${rows} has no comma.` };
    if (text.slice(at, comma) !== String(rows)) {
      return { ok: false, rows, problem: `Row ${rows} is numbered ${text.slice(at, comma) || "nothing"}.` };
    }
    const label = text.slice(comma + 1, end);
    if (!isLabelCode(label)) return { ok: false, rows, problem: `Row ${rows} has the label "${label}".` };
    rows++;
    at = end + 1;
  }
  return { ok: true, rows, problem: null };
}

export interface ExportCheck {
  id: "cover" | "labels" | "rows";
  ok: boolean;
  /** One plain sentence saying what was checked and what was found. */
  text: string;
}

export interface ExportReport {
  ok: boolean;
  frames: number;
  /** Frames per label: [home, away, dead ball, no answer]. */
  counts: [number, number, number, number];
  checks: ExportCheck[];
  /** The checked file, ready to save. Null when a check did not pass. */
  csv: string | null;
}

const fmt = (n: number) => Math.round(n).toLocaleString("en-GB");

/**
 * Checks the labels, writes the export and reads it back:
 *  1. the labels cover every frame of the match, once each;
 *  2. every label is H, A, D or X;
 *  3. the file written has one row per frame, numbered from 0 with no gap.
 * `runs` is deliberately loose (unknown labels, odd lengths) so a damaged list is reported, not thrown.
 */
export function validateExport(runs: readonly (readonly [unknown, unknown])[], matchFrames: number): ExportReport {
  const counts: [number, number, number, number] = [0, 0, 0, 0];
  let covered = 0;
  let badLength = 0;
  let strangeFrames = 0;
  const strange = new Set<string>();

  for (const [code, length] of runs) {
    const n = typeof length === "number" && Number.isInteger(length) && length > 0 ? length : 0;
    if (n === 0) {
      badLength++;
      continue;
    }
    covered += n;
    if (isLabelCode(code)) counts[LABEL_CODES.indexOf(code)] += n;
    else {
      strange.add(String(code));
      strangeFrames += n;
    }
  }

  const checks: ExportCheck[] = [];

  const coverOk = matchFrames > 0 && covered === matchFrames && badLength === 0;
  checks.push({
    id: "cover",
    ok: coverOk,
    text: coverOk
      ? `Every frame has a label: ${fmt(covered)} of ${fmt(matchFrames)}.`
      : badLength > 0
        ? `${fmt(badLength)} label ${badLength === 1 ? "run has" : "runs have"} no length, so frames are missing.`
        : covered < matchFrames
          ? `${fmt(matchFrames - covered)} frames have no label: the labels cover ${fmt(covered)} of ${fmt(matchFrames)}.`
          : `The labels run ${fmt(covered - matchFrames)} frames past the end of the match.`,
  });

  const labelsOk = strange.size === 0;
  checks.push({
    id: "labels",
    ok: labelsOk,
    text: labelsOk
      ? "Only Home, Away, Dead ball and No answer are used."
      : `${fmt(strangeFrames)} frames carry a label the export does not allow: ${[...strange]
          .slice(0, 4)
          .map((s) => `"${s}"`)
          .join(", ")}.`,
  });

  let csv: string | null = null;
  if (coverOk && labelsOk) {
    const text = labelsCsv(runs as readonly LabelRun[]);
    const read = inspectCsv(text);
    const rowsOk = read.ok && read.rows === matchFrames;
    checks.push({
      id: "rows",
      ok: rowsOk,
      text: rowsOk
        ? `The file has one row per frame, 0 to ${fmt(matchFrames - 1)}, with no gaps.`
        : (read.problem ?? `The file has ${fmt(read.rows)} rows for ${fmt(matchFrames)} frames.`),
    });
    if (rowsOk) csv = text;
  } else {
    checks.push({ id: "rows", ok: false, text: "No file was written, because a check above did not pass." });
  }

  return { ok: checks.every((c) => c.ok), frames: matchFrames, counts, checks, csv };
}

/** A file name for the export, from the match title: "home-vs-away-possession.csv". */
export function exportFileName(title: string): string {
  const slug = title
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return `${slug || "match"}-possession.csv`;
}
