import { useApp } from "../state/app";
import { Inspector } from "./Inspector";
import { DownloadIcon, FolderIcon, ReplayIcon } from "./icons";
import { MatchPanel } from "./MatchPanel";
import { ShapePanel } from "./ShapePanel";
import { Timeline, TimelineHead } from "./Timeline";
import { Transport } from "./Transport";

/** Says where the match on screen came from, and offers the way to open your own. */
export function SourceBar() {
  const { run, videoName, openSettings } = useApp();
  return (
    <footer className="source">
      {run.source === "demo" ? (
        <span className="tag">Simulated match</span>
      ) : (
        <span className="source__text" title={run.fileName ?? videoName ?? undefined}>
          {run.source === "blank" ? (videoName ?? "Video") : (run.fileName ?? run.title)}
        </span>
      )}
      <span className="source__dot" aria-hidden="true">
        ·
      </span>
      <button type="button" className="source__open" onClick={openSettings}>
        <FolderIcon size={14} />
        Open your own files
      </button>
    </footer>
  );
}

export function PossessionTab() {
  const { playback, run, validateExport } = useApp();
  const hasChanges = run.changes.length > 0;
  return (
    <div className="poss">
      <div className="poss__left">
        <MatchPanel />
        <section className="panel tlp" aria-label="Possession timeline">
          <div className="tlp__rows">
            <TimelineHead />
            <Timeline />
          </div>
          <div className="tlp__transport">
            <Transport />
          </div>
        </section>
      </div>
      <div className="poss__right">
        <ShapePanel />
        <Inspector />
        <div className="actions">
          <button
            type="button"
            className="btn btn--outline btn--lg"
            disabled={!hasChanges}
            title={hasChanges ? "Jump to three seconds before the nearest change and play through it" : "This match has no changes of possession"}
            onClick={() => playback.replayTransition()}
          >
            <ReplayIcon size={18} />
            Replay transition
          </button>
          <button type="button" className="btn btn--primary btn--lg" onClick={validateExport}>
            <DownloadIcon size={20} />
            Validate export
          </button>
        </div>
        <SourceBar />
      </div>
    </div>
  );
}
