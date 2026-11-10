import type { ExportReport } from "../model/csv";
import { formatCount, formatShares } from "../model/time";
import { useApp } from "../state/app";
import { AlertIcon, CheckIcon, DownloadIcon, FilmIcon, FolderIcon } from "./icons";
import { Sheet } from "./Sheet";

/**
 * How a save ended. "asking": the save prompt is still open. "unsure": this view could not confirm
 * the save. "blocked": this view cannot hand files over at all.
 */
export type SaveOutcome = "saved" | "asking" | "declined" | "unsure" | "blocked";

export interface ExportResult {
  report: ExportReport;
  fileName: string;
  /** Size of the file, in bytes. */
  bytes: number;
  outcome: SaveOutcome;
}

const SAVE_LINE: Record<SaveOutcome, string> = {
  saved: "",
  asking: "Confirm the save to keep the file.",
  declined: "The file was not saved.",
  unsure: "If no file arrived, open the viewer on your computer to save it.",
  blocked: "This view can't save files. Open the viewer on your computer to save it.",
};

/** What the export check found, and the file it wrote. */
export function ExportSheet({ result, onClose, onSaveAgain }: { result: ExportResult | null; onClose: () => void; onSaveAgain: () => void }) {
  const { run } = useApp();
  if (!result) return null;
  const { report } = result;
  const size = result.bytes >= 1_000_000 ? `${(result.bytes / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(result.bytes / 1000))} KB`;
  const shares = formatShares(report.counts);
  const saved = result.outcome === "saved";
  const fileLine = `${result.fileName} · ${size} · one row for each of ${formatCount(report.frames)} frames`;
  return (
    <Sheet
      open
      onClose={onClose}
      title={!report.ok ? "The export did not pass" : saved ? "Export checked and saved" : "Export checked"}
      description={!report.ok ? "Nothing was saved. Here is what was found." : saved ? fileLine : `${fileLine}. ${SAVE_LINE[result.outcome]}`}
      footer={
        <>
          {report.ok && result.outcome !== "asking" && result.outcome !== "blocked" ? (
            <button type="button" className="btn btn--quiet" onClick={onSaveAgain}>
              <DownloadIcon size={16} />
              {saved ? "Save again" : "Save the file"}
            </button>
          ) : null}
          <button type="button" className="btn btn--primary" onClick={onClose} autoFocus>
            Done
          </button>
        </>
      }
    >
      <ul className="checks">
        {report.checks.map((check) => (
          <li key={check.id} className={check.ok ? "checks__item" : "checks__item checks__item--bad"}>
            <span className="checks__mark">{check.ok ? <CheckIcon size={15} /> : <AlertIcon size={15} />}</span>
            {check.text}
          </li>
        ))}
      </ul>
      <dl className="counts">
        <div>
          <dt>
            <span className="legend__dot legend__dot--home" />
            {run.home}
          </dt>
          <dd className="num">{shares[0]}</dd>
        </div>
        <div>
          <dt>
            <span className="legend__dot legend__dot--away" />
            {run.away}
          </dt>
          <dd className="num">{shares[1]}</dd>
        </div>
        <div>
          <dt>
            <span className="legend__dot legend__dot--dead" />
            Dead ball
          </dt>
          <dd className="num">{shares[2]}</dd>
        </div>
        <div>
          <dt>
            <span className="legend__hatch" />
            No answer
          </dt>
          <dd className="num">{shares[3]}</dd>
        </div>
      </dl>
      {report.ok ? <p className="sheet__note">The file starts with the line frame,label and then has one row per frame from 0: H for {run.home}, A for {run.away}, D for a dead ball and X where there is no answer.</p> : null}
    </Sheet>
  );
}

const SHORTCUTS: [string, string][] = [
  ["Space", "Play or pause"],
  ["← →", "One frame back or forward"],
  ["Shift ← →", "One second back or forward"],
  ["[  ]", "Previous or next change"],
];

/** Files, a few choices about the picture, and the keyboard. */
export function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { run, videoName, settings, setSettings, pickRunFile, pickVideo, backToDemo, saveDemoRunFile } = useApp();
  const toggle = (key: keyof typeof settings, label: string, hint: string) => (
    <label className="switch">
      <input type="checkbox" checked={settings[key]} onChange={(event) => setSettings({ [key]: event.target.checked })} />
      <span className="switch__track" aria-hidden="true" />
      <span className="switch__text">
        <span className="switch__label">{label}</span>
        <span className="switch__hint">{hint}</span>
      </span>
    </label>
  );
  return (
    <Sheet open={open} onClose={onClose} title="Files and settings" wide>
      <section className="sheet__section">
        <h3 className="label">Your match</h3>
        <p className="sheet__now">
          {run.source === "demo" ? "Showing the simulated match." : run.source === "blank" ? `Showing ${videoName ?? "a video"} with no run file yet.` : `Showing ${run.fileName ?? run.title}${videoName ? ` with ${videoName}` : ""}.`}
        </p>
        <div className="sheet__row">
          <button type="button" className="btn btn--quiet" onClick={pickRunFile}>
            <FolderIcon size={16} />
            Open a run file
          </button>
          <button type="button" className="btn btn--quiet" onClick={pickVideo}>
            <FilmIcon size={16} />
            Add the match video
          </button>
          {run.source !== "demo" ? (
            <button type="button" className="btn btn--ghost" onClick={backToDemo}>
              Back to the simulated match
            </button>
          ) : null}
        </div>
        <p className="sheet__note">You can also drop a run file (.json) or a video anywhere on the page. Files are read in this browser only; nothing is uploaded.</p>
        {run.source === "demo" ? (
          <p className="sheet__note">
            Want to see the format?{" "}
            <button type="button" className="link" onClick={saveDemoRunFile}>
              Save the simulated match as a run file
            </button>
            .
          </p>
        ) : null}
      </section>
      <section className="sheet__section">
        <h3 className="label">Picture</h3>
        {toggle("labels", "Player labels", "Show each player's track label in the match view.")}
        {toggle("follow", "Follow the play", "The drawn match view turns with the ball, as a camera would.")}
        {toggle("outlines", "Team outlines", "Outline each team's shape on the Team shape pitch.")}
      </section>
      <section className="sheet__section">
        <h3 className="label">Keyboard</h3>
        <dl className="keys">
          {SHORTCUTS.map(([keys, what]) => (
            <div key={keys}>
              <dt>
                {keys
                  .split(" ")
                  .filter(Boolean)
                  .map((k) => (
                    <kbd key={k}>{k}</kbd>
                  ))}
              </dt>
              <dd>{what}</dd>
            </div>
          ))}
        </dl>
      </section>
    </Sheet>
  );
}
