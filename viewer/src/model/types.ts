/**
 * Excensia Vantage run file, `excensia.vantage.run/1`.
 *
 * The first half of this file is the contract exactly as the engine writes it (JSON).
 * The second half is the compact form the viewer works from once a file is loaded.
 */

export const RUN_SCHEMA = "excensia.vantage.run/1" as const;

/** H home has the ball, A away has the ball, D dead ball, X no answer. */
export type LabelCode = "H" | "A" | "D" | "X";

/** Run-length pair: a label and how many frames it lasts. */
export type LabelRun = [LabelCode, number];

export interface RunFileSpell {
  /** First frame of the spell. */
  start: number;
  /** Last frame of the spell (inclusive). */
  end: number;
  label: LabelCode;
  /** 0..1 */
  confidence?: number;
  /** How much each kind of evidence weighed in the call. */
  evidence?: { ball: number; shape: number; timing: number };
  /** Share of the spell's frames in which the ball was seen, 0..1. */
  ball_seen?: number;
}

export interface RunFileTransition {
  /** First frame of the new label. */
  frame: number;
  from: LabelCode;
  to: LabelCode;
  confidence?: number;
  why?: string;
}

/** Metres, origin at the pitch centre, x along the length. */
export type XY = [number, number];

export interface RunFileSample {
  /** Ball, or null when the ball is not seen. */
  b: XY | null;
  /** Home players. The index is the track slot; null marks a player not seen in this sample. */
  h: (XY | null)[];
  /** Away players. */
  a: (XY | null)[];
}

export interface RunFile {
  schema: typeof RUN_SCHEMA;
  generator?: { name: string; version: string };
  match: {
    title?: string;
    home: { name: string };
    away: { name: string };
    fps: number;
    frames: number;
    pitch?: { length: number; width: number };
    periods?: { name: string; start: number; end: number }[];
  };
  /** Run-length labels. The lengths add up to match.frames. */
  labels: LabelRun[];
  spells?: RunFileSpell[];
  transitions?: RunFileTransition[];
  /** Sample i is frame i * every. */
  tracking?: { every: number; samples: RunFileSample[] };
  /** Run frame f is video frame f + offset_frames. */
  video?: { offset_frames: number };
}

/* ───────────────────────── the viewer's own form ───────────────────────── */

/** 0 home, 1 away, 2 dead ball, 3 no answer. */
export type Label = 0 | 1 | 2 | 3;
export const H = 0 as const;
export const A = 1 as const;
export const D = 2 as const;
export const X = 3 as const;
export const LABEL_CODES: readonly LabelCode[] = ["H", "A", "D", "X"];

export interface Period {
  name: string;
  /** First frame. */
  start: number;
  /** Last frame (inclusive). */
  end: number;
}

export interface Evidence {
  ball: number;
  shape: number;
  timing: number;
}

/** A stretch of frames with one label and one set of evidence. */
export interface Segment {
  start: number;
  /** Last frame (inclusive). */
  end: number;
  label: Label;
  confidence: number | null;
  /** Normalised so the three add up to 1. */
  evidence: Evidence | null;
  ballSeen: number | null;
}

/** A frame where the label changes. */
export interface Change {
  /** First frame of the new label. */
  frame: number;
  from: Label;
  to: Label;
  confidence: number | null;
  why: string | null;
}

export interface Tracking {
  every: number;
  /** Number of samples. */
  count: number;
  homeSlots: number;
  awaySlots: number;
  /** count * 2 values, NaN where the ball is not seen. */
  ball: Float32Array;
  /** count * homeSlots * 2 values, NaN where a player is not seen. */
  home: Float32Array;
  away: Float32Array;
}

export type RunSource = "demo" | "file" | "blank";

export interface Run {
  source: RunSource;
  fileName: string | null;
  generator: { name: string; version: string } | null;
  title: string;
  home: string;
  away: string;
  fps: number;
  frames: number;
  pitch: { length: number; width: number };
  periods: Period[];
  /** Labels as merged runs: start frame and label of each run. */
  runStart: Int32Array;
  runLabel: Uint8Array;
  /** Label runs, split wherever the file's spells split them. */
  segments: Segment[];
  segStart: Int32Array;
  changes: Change[];
  changeFrame: Int32Array;
  tracking: Tracking | null;
  videoOffset: number;
  /** True when the file itself carried spells (confidence, evidence). */
  hasSpellDetail: boolean;
  /** True when the file itself carried transitions (reasons). */
  hasChangeDetail: boolean;
}

/** Extras the simulated match carries for drawing its own footage. Never part of a run file. */
export interface DemoScene {
  goals: { frame: number; side: 0 | 1 }[];
}
