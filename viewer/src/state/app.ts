import { createContext, useContext } from "react";
import type { EvidenceKind } from "../model/explain";
import type { DemoScene, Run } from "../model/types";
import type { TimelineView } from "../render/timeline";
import type { Playback } from "./playback";

export type TabId = "match" | "possession" | "shape" | "insights";

export interface ViewSettings {
  /** Track labels beside players in the match view. */
  labels: boolean;
  /** The drawn match view turns to follow play. */
  follow: boolean;
  /** Team outlines on the team shape pitch. */
  outlines: boolean;
}

export interface AppState {
  playback: Playback;
  run: Run;
  /** Extras of the simulated match; null for a run from a file. */
  scene: DemoScene | null;
  videoUrl: string | null;
  videoName: string | null;
  settings: ViewSettings;
  setSettings: (patch: Partial<ViewSettings>) => void;
  tab: TabId;
  setTab: (tab: TabId) => void;
  /** The kind of evidence picked in the inspector. With a video and its camera it is also drawn on the video. */
  evidence: EvidenceKind;
  setEvidence: (kind: EvidenceKind) => void;
  /** Moves to a frame, optionally opening a tab, and stops there. */
  goTo: (frame: number, tab?: TabId) => void;
  pickRunFile: () => void;
  pickVideo: () => void;
  backToDemo: () => void;
  saveDemoRunFile: () => void;
  openSettings: () => void;
  validateExport: () => void;
  timeline: ViewStore;
}

export const AppContext = createContext<AppState | null>(null);

export function useApp(): AppState {
  const app = useContext(AppContext);
  if (!app) throw new Error("useApp must be used inside the app.");
  return app;
}

/** The stretch of the match the possession timeline shows. Shared so the scrubber can mark it. */
export class ViewStore {
  view: TimelineView;
  frames: number;
  private listeners = new Set<() => void>();

  constructor(frames: number) {
    this.frames = frames;
    this.view = { start: 0, span: frames };
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** A string that changes whenever the view does, for useSyncExternalStore. */
  key = (): string => `${Math.round(this.view.start)}:${Math.round(this.view.span)}`;

  get zoomed(): boolean {
    return this.view.span < this.frames - 0.5;
  }

  set(view: TimelineView): void {
    if (view.start === this.view.start && view.span === this.view.span) return;
    this.view = view;
    for (const listener of this.listeners) listener();
  }

  reset(frames = this.frames): void {
    this.frames = frames;
    this.set({ start: 0, span: frames });
  }
}
