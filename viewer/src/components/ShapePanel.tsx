import { useMemo } from "react";
import { emptyMoment, sampleMoment } from "../model/moment";
import { shapeIndex } from "../model/shape";
import { drawBallTop, drawPitchTop, drawPlayersTop, fitPitch } from "../render/pitchTop";
import { useApp } from "../state/app";
import { useCanvas } from "../state/hooks";

/** Both teams from above at the selected moment. */
export function ShapePanel() {
  const { playback, run, pickRunFile } = useApp();
  const moment = useMemo(() => emptyMoment(run.tracking), [run]);
  const keepers = useMemo<[number, number]>(() => shapeIndex(run)?.keepers ?? [-1, -1], [run]);

  const canvasRef = useCanvas(
    playback,
    (ctx, size, frame) => {
      const lay = fitPitch(size.width, size.height, run.pitch, Math.max(7, Math.min(11, size.height * 0.04)), 1.5);
      drawPitchTop(ctx, size.width, size.height, lay);
      const tracking = run.tracking;
      if (!tracking) return;
      sampleMoment(tracking, frame, run.fps, moment);
      const radius = Math.max(4, Math.min(6.2, lay.sy * 1.62));
      drawBallTop(ctx, lay, moment, { radius: Math.max(2.6, radius * 0.6), caption: true });
      drawPlayersTop(ctx, lay, moment, { labels: size.width > 300, keepers, homeSlots: tracking.homeSlots, radius, width: size.width, labelSize: size.width > 480 ? 11 : 10 });
    },
    [run],
  );

  return (
    <section className="panel shape" aria-label="Team shape at the selected moment">
      <header className="panel-head">
        <h2 className="panel-title">
          Team shape <span className="panel-title__sub">(selected moment)</span>
        </h2>
      </header>
      <div className="shape__pitch">
        <canvas ref={canvasRef} className="shape__canvas" role="img" aria-label="Both teams from above" />
        {run.tracking ? null : (
          <div className="pitch-empty">
            <p className="pitch-empty__title">No tracking in this run</p>
            <p className="pitch-empty__text">Team shape needs player positions. Open a run file that includes tracking to see it here.</p>
            <button type="button" className="btn btn--quiet" onClick={pickRunFile}>
              Open a run file
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
