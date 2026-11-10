import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import wordmarkUrl from "./assets/wordmark.png";
import { Header } from "./components/Header";
import { AlertIcon, CrossIcon, FolderIcon, Mark } from "./components/icons";
import { InsightsTab } from "./components/InsightsTab";
import { MatchTab } from "./components/MatchTab";
import { PossessionTab } from "./components/PossessionTab";
import { ExportSheet, SettingsSheet, type ExportResult, type SaveOutcome } from "./components/Sheets";
import { TeamShapeTab } from "./components/TeamShapeTab";
import { exportFileName, validateExport } from "./model/csv";
import { demoRun } from "./model/demo";
import { runsOf } from "./model/derive";
import { blankRun, parseRunFile, RunFileError, toRunFile } from "./model/loader";
import type { DemoScene, Run } from "./model/types";
import { resetLabelSprites } from "./render/labels";
import { refreshBoards, setBoardWordmark } from "./render/matchView";
import { AppContext, ViewStore, type AppState, type TabId, type ViewSettings } from "./state/app";
import { useFontRepaint } from "./state/hooks";
import { Playback } from "./state/playback";

/** The moment the simulated match opens on: 72:14, as in the concept. */
const DEMO_START_SECONDS = 72 * 60 + 14;

const VIDEO_NAME = /\.(mp4|m4v|mov|webm|mkv|avi|ogv)$/i;

const TAB_IDS: readonly TabId[] = ["match", "possession", "shape", "insights"];

/**
 * A link can open the viewer at a moment and a tab: #t=72:14 (or seconds, #t=4334) and #tab=shape.
 * With &play it also starts the match by itself as soon as it is up.
 * Nothing is written back to the address.
 */
function readLink(): { seconds: number | null; tab: TabId | null; play: boolean } {
  let seconds: number | null = null;
  let tab: TabId | null = null;
  let play = false;
  try {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const t = params.get("t");
    if (t) {
      const parts = t.split(":").map(Number);
      const value = parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0];
      if (Number.isFinite(value) && value >= 0) seconds = value;
    }
    const wanted = params.get("tab");
    if (wanted && (TAB_IDS as readonly string[]).includes(wanted)) tab = wanted as TabId;
    play = params.has("play") && params.get("play") !== "0";
  } catch {
    // an address this page cannot read is no reason to stop
  }
  return { seconds, tab, play };
}

interface HostDownloads {
  save(request: { filename: string; data: Blob }): Promise<unknown>;
}

interface HostPage {
  use?: (name: "downloads") => Promise<HostDownloads | null>;
}

/**
 * When the viewer runs as a hosted Claude page, a script may not start a download itself: the host
 * shows its own save prompt instead. Asked for once, at start, so a click never waits for it.
 */
const hostPage = (window as unknown as { claude?: HostPage }).claude;
const hostDownloads: Promise<HostDownloads | null> | null = hostPage?.use ? hostPage.use("downloads").catch(() => null) : null;

function browserSave(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

/** Hands a file to the person looking at the page, and says how that went. */
async function saveFile(name: string, parts: BlobPart[], type: string): Promise<SaveOutcome> {
  const blob = new Blob(parts, { type });
  if (!hostDownloads) {
    browserSave(name, blob);
    return "saved";
  }
  const downloads = await hostDownloads;
  if (!downloads) {
    // a hosted view that cannot hand files over; the browser's own save may still work there
    browserSave(name, blob);
    return "unsure";
  }
  try {
    await downloads.save({ filename: name, data: blob });
    return "saved";
  } catch (error) {
    const code = (error as { code?: string } | null)?.code;
    return code === "declined" || code === "rate_limited" ? "declined" : "blocked";
  }
}

interface Loaded {
  run: Run;
  scene: DemoScene | null;
}

export function App() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [playback, setPlayback] = useState<Playback | null>(null);
  const [video, setVideo] = useState<{ url: string; name: string } | null>(null);
  const link = useMemo(readLink, []);
  const startWhenReady = useRef(link.play);
  const [tab, setTab] = useState<TabId>(link.tab ?? "possession");
  const [settings, setSettingsState] = useState<ViewSettings>({ labels: true, follow: true, outlines: true });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exported, setExported] = useState<ExportResult | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dropping, setDropping] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const timeline = useMemo(() => new ViewStore(1), []);
  const runInput = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const demo = useRef<Loaded | null>(null);
  const loadedRef = useRef(loaded);
  loadedRef.current = loaded;
  const videoRef = useRef(video);
  videoRef.current = video;
  // why the match on screen may have no video beside it, or null; set the moment a match is shown
  const withheld = useRef<string | null>(null);

  const show = useCallback(
    (next: Loaded, frame: number) => {
      timeline.reset(next.run.frames);
      withheld.current = next.run.videoWithheld;
      if (next.run.videoWithheld) {
        setVideo((current) => {
          if (current) URL.revokeObjectURL(current.url);
          return null;
        });
      }
      setLoaded(next);
      setPlayback((current) => {
        if (current) {
          current.setRun(next.run, frame);
          return current;
        }
        const created = new Playback(next.run);
        created.seek(frame);
        return created;
      });
    },
    [timeline],
  );

  // build the simulated match once the page has painted
  useEffect(() => {
    let cancelled = false;
    const raf = requestAnimationFrame(() => {
      setTimeout(() => {
        if (cancelled) return;
        const built = demoRun();
        demo.current = { run: built.run, scene: built.scene };
        // A page can carry its own run (`python -m vantage export --page`): a script element of
        // type application/json with the id "vantage-run". Such a page opens on that match.
        const carried = document.getElementById("vantage-run");
        if (carried?.textContent) {
          try {
            const run = parseRunFile(carried.textContent, carried.getAttribute("data-name") ?? "match.run.json");
            show({ run, scene: null }, (link.seconds ?? 0) * run.fps);
            return;
          } catch (error) {
            setNotice(error instanceof RunFileError ? `The match built into this page can't be opened. ${error.message}` : "The match built into this page can't be opened.");
          }
        }
        show(demo.current, (link.seconds ?? DEMO_START_SECONDS) * built.run.fps);
      }, 0);
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [show, link]);

  // &play in the address: start once, as soon as the match is up
  useEffect(() => {
    if (!startWhenReady.current || !playback || !loaded) return;
    startWhenReady.current = false;
    playback.play();
  }, [playback, loaded]);

  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      setBoardWordmark(image);
      playback?.invalidate();
    };
    image.src = wordmarkUrl;
  }, [playback]);
  useFontRepaint(playback, () => {
    refreshBoards();
    resetLabelSprites();
  });
  useEffect(() => () => playback?.dispose(), [playback]);

  const dropVideo = useCallback(() => {
    setVideo((current) => {
      if (current) URL.revokeObjectURL(current.url);
      return null;
    });
  }, []);

  const openRun = useCallback(
    async (file: File) => {
      setBusy(`Reading ${file.name}…`);
      try {
        const text = await file.text();
        const run = parseRunFile(text, file.name);
        const closed = run.videoWithheld && videoRef.current ? videoRef.current.name : null;
        show({ run, scene: null }, 0);
        setNotice(closed ? `${closed} was closed. ${run.videoWithheld}` : null);
        setTab("possession");
      } catch (error) {
        setNotice(error instanceof RunFileError ? `${file.name} can't be opened. ${error.message}` : `${file.name} can't be opened. It could not be read.`);
      } finally {
        setBusy(null);
      }
    },
    [show],
  );

  const openVideo = useCallback(
    (file: File) => {
      if (withheld.current) {
        setNotice(`${file.name} was not opened. ${withheld.current}`);
        return;
      }
      const url = URL.createObjectURL(file);
      const probe = document.createElement("video");
      probe.preload = "metadata";
      probe.muted = true;
      const done = () => {
        probe.onloadedmetadata = null;
        probe.ondurationchange = null;
        probe.onerror = null;
        probe.removeAttribute("src");
        probe.load();
      };
      let settled = false;
      const ready = (seconds: number) => {
        if (settled) return;
        settled = true;
        done();
        setVideo((current) => {
          if (current) URL.revokeObjectURL(current.url);
          return { url, name: file.name };
        });
        const current = loadedRef.current;
        // a video on its own gets an empty match to sit in until its run file arrives
        if (!current || current.run.source !== "file") {
          show({ run: blankRun(Math.max(1, Math.round(seconds * 25)), 25, file.name.replace(/\.[^.]+$/, "")), scene: null }, 0);
          setNotice("This video has no run file yet. Open the run file for this match to see who has the ball.");
        } else setNotice(null);
        setTab("possession");
      };
      probe.onloadedmetadata = () => {
        if (Number.isFinite(probe.duration)) {
          ready(probe.duration);
          return;
        }
        // a recording made in a browser often does not say how long it is until it has been read to its end
        probe.ondurationchange = () => {
          if (Number.isFinite(probe.duration)) ready(probe.duration);
        };
        probe.currentTime = 1e9;
        setTimeout(() => ready(Number.isFinite(probe.duration) ? probe.duration : 0), 4000);
      };
      probe.onerror = () => {
        if (settled) return;
        settled = true;
        done();
        URL.revokeObjectURL(url);
        setNotice(`${file.name} can't be played in this browser. Try an MP4 (H.264) or WebM file.`);
      };
      probe.src = url;
    },
    [show],
  );

  const openFiles = useCallback(
    async (files: File[]) => {
      const runs = files.filter((f) => /\.json$/i.test(f.name) || f.type === "application/json");
      const videos = files.filter((f) => f.type.startsWith("video/") || VIDEO_NAME.test(f.name));
      if (!runs.length && !videos.length) {
        setNotice(files.length ? `${files[0].name} is not a run file (.json) or a video.` : null);
        return;
      }
      if (runs[0]) await openRun(runs[0]);
      if (videos[0]) openVideo(videos[0]);
      setSettingsOpen(false);
    },
    [openRun, openVideo],
  );

  // drag and drop anywhere
  useEffect(() => {
    let depth = 0;
    const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes("Files");
    const enter = (event: DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth++;
      setDropping(true);
    };
    const over = (event: DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    };
    const leave = (event: DragEvent) => {
      if (!hasFiles(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDropping(false);
    };
    const drop = (event: DragEvent) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth = 0;
      setDropping(false);
      void openFiles(Array.from(event.dataTransfer?.files ?? []));
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, [openFiles]);

  // the keyboard: space, arrows, and [ ] for changes
  useEffect(() => {
    if (!playback) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], dialog")) return;
      // a control reached with the keyboard keeps its own keys
      const control = target?.closest<HTMLElement>('button, a, [role="slider"], [role="tab"], [role="option"]');
      const keyboardFocus = control?.matches(":focus-visible") ?? false;
      switch (event.key) {
        case " ":
        case "Spacebar":
          if (keyboardFocus && control?.getAttribute("role") !== "slider") return;
          playback.toggle();
          break;
        case "ArrowLeft":
        case "ArrowRight": {
          if (keyboardFocus && control?.getAttribute("role") !== "slider" && control?.tagName !== "BUTTON") return;
          if (keyboardFocus && control?.classList.contains("scrub")) return;
          const direction = event.key === "ArrowLeft" ? -1 : 1;
          playback.step(direction * (event.shiftKey ? Math.round(playback.run.fps) : 1));
          break;
        }
        case "[":
          playback.toPreviousChange();
          break;
        case "]":
          playback.toNextChange();
          break;
        default:
          return;
      }
      event.preventDefault();
    };
    // some browsers press a focused button when the space bar comes up, whatever happened on the way down
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key !== " " && event.key !== "Spacebar") return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable], dialog")) return;
      const control = target?.closest<HTMLElement>("button, a");
      if (control && !control.matches(":focus-visible")) event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [playback]);

  const runExport = useCallback(() => {
    const current = loadedRef.current;
    if (!current) return;
    const report = validateExport(runsOf(current.run), current.run.frames);
    const fileName = exportFileName(current.run.title);
    if (report.csv === null) {
      setExported({ report, fileName, bytes: 0, outcome: "saved" });
      return;
    }
    const csv = report.csv;
    const bytes = new Blob([csv]).size;
    setExported({ report, fileName, bytes, outcome: "asking" });
    void saveFile(fileName, [csv], "text/csv;charset=utf-8").then((outcome) => {
      setExported((current) => (current && current.report === report ? { ...current, outcome } : current));
    });
  }, []);

  if (!loaded || !playback) {
    return (
      <div className="app app--loading">
        <Header tab={tab} onTab={setTab} onSettings={() => undefined} />
        <main className="loading" aria-busy="true">
          <Mark size={44} className="loading__mark" />
          <p className="loading__text">Setting up the simulated match</p>
        </main>
      </div>
    );
  }

  const state: AppState = {
    playback,
    run: loaded.run,
    scene: loaded.scene,
    videoUrl: video?.url ?? null,
    videoName: video?.name ?? null,
    settings,
    setSettings: (patch) => setSettingsState((current) => ({ ...current, ...patch })),
    tab,
    setTab,
    goTo: (frame, nextTab) => {
      playback.pause();
      playback.seek(frame);
      if (nextTab) setTab(nextTab);
    },
    pickRunFile: () => runInput.current?.click(),
    pickVideo: () => videoInput.current?.click(),
    backToDemo: () => {
      dropVideo();
      if (demo.current) show(demo.current, DEMO_START_SECONDS * demo.current.run.fps);
      setNotice(null);
      setSettingsOpen(false);
      setTab("possession");
    },
    saveDemoRunFile: () => {
      if (!demo.current) return;
      void saveFile("simulated-match.run.json", [JSON.stringify(toRunFile(demo.current.run))], "application/json").then((outcome) => {
        if (outcome === "blocked") setNotice("This view can't save files. Open the viewer on your computer to save the run file.");
      });
    },
    openSettings: () => setSettingsOpen(true),
    validateExport: runExport,
    timeline,
  };

  return (
    <AppContext.Provider value={state}>
      <div className="app" data-tab={tab}>
        <Header tab={tab} onTab={setTab} onSettings={() => setSettingsOpen(true)} />
        <main className="main">
          <div className="main__panel" id="tabpanel" role="tabpanel" aria-labelledby={`tab-${tab}`}>
            {tab === "possession" ? <PossessionTab /> : tab === "match" ? <MatchTab /> : tab === "shape" ? <TeamShapeTab /> : <InsightsTab />}
          </div>
        </main>

        {notice ? (
          <div className="notice" role="alert">
            <AlertIcon size={18} />
            <p>{notice}</p>
            <button type="button" className="icon-btn icon-btn--sm" aria-label="Dismiss" onClick={() => setNotice(null)}>
              <CrossIcon size={16} />
            </button>
          </div>
        ) : null}
        {busy ? (
          <div className="notice notice--busy" role="status">
            <p>{busy}</p>
          </div>
        ) : null}
        {dropping ? (
          <div className="drop" aria-hidden="true">
            <div className="drop__box">
              <FolderIcon size={30} />
              <p className="drop__title">Drop a run file or a video</p>
              <p className="drop__text">Files are read in this browser only. Nothing is uploaded.</p>
            </div>
          </div>
        ) : null}

        <SettingsSheet open={settingsOpen} onClose={() => setSettingsOpen(false)} />
        <ExportSheet
          result={exported}
          onClose={() => setExported(null)}
          onSaveAgain={() => {
            const report = exported?.report;
            if (!exported || !report?.csv) return;
            setExported({ ...exported, outcome: "asking" });
            void saveFile(exported.fileName, [report.csv], "text/csv;charset=utf-8").then((outcome) => {
              setExported((current) => (current && current.report === report ? { ...current, outcome } : current));
            });
          }}
        />
        <input
          ref={runInput}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = "";
            void openFiles(files);
          }}
        />
        <input
          ref={videoInput}
          type="file"
          accept="video/*,.mp4,.m4v,.mov,.webm,.mkv"
          hidden
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = "";
            void openFiles(files);
          }}
        />
      </div>
    </AppContext.Provider>
  );
}
