import { useEffect, useMemo, useRef } from "react";
import { emptyMoment, sampleMoment } from "../model/moment";
import { shapeIndex } from "../model/shape";
import { formatClock } from "../model/time";
import { cameraLookAt } from "../render/camera";
import { drawMatchView } from "../render/matchView";
import { useApp } from "../state/app";
import { useCanvas, usePlayback } from "../state/hooks";
import { FilmIcon } from "./icons";

/** The match drawn from its tracking, from a high camera on the halfway line. */
function DrawnMatch() {
  const { playback, run, settings } = useApp();
  const moment = useMemo(() => emptyMoment(run.tracking), [run]);
  const earlier = useMemo(() => emptyMoment(run.tracking), [run]);
  const keepers = useMemo<[number, number]>(() => shapeIndex(run)?.keepers ?? [-1, -1], [run]);
  const canvasRef = useCanvas(
    playback,
    (ctx, size, frame) => {
      const tracking = run.tracking;
      if (!tracking) return;
      // where everyone was an eighth of a second ago: players stride as they run, and a fast ball leaves a streak
      sampleMoment(tracking, Math.max(0, frame - run.fps * 0.125), run.fps, earlier);
      sampleMoment(tracking, frame, run.fps, moment);
      const look = settings.follow ? cameraLookAt(tracking, run.fps, frame) : null;
      drawMatchView(
        ctx,
        size.width,
        size.height,
        {
          pitch: run.pitch,
          moment,
          lookX: look ? look.x : 0,
          lookY: look ? look.y : 0,
          keepers,
          homeSlots: tracking.homeSlots,
          before: frame >= run.fps * 0.125 ? earlier : null,
          beforeSeconds: 0.125,
        },
        { labels: settings.labels && size.width >= 520 },
      );
    },
    [run, settings.labels, settings.follow],
  );
  return <canvas ref={canvasRef} className="match__media" onClick={() => playback.toggle()} aria-label="The match, drawn from tracking" role="img" />;
}

function MatchVideo({ url }: { url: string }) {
  const { playback } = useApp();
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const attach = () => playback.attachVideo(video);
    if (video.readyState >= 1) attach();
    video.addEventListener("loadedmetadata", attach);
    return () => {
      video.removeEventListener("loadedmetadata", attach);
      playback.attachVideo(null);
    };
  }, [playback, url]);
  return <video ref={ref} className="match__media" src={url} playsInline preload="auto" onClick={() => playback.toggle()} />;
}

/** The match itself: the video when one is loaded, otherwise the match drawn from its tracking. */
export function MatchPanel() {
  const { playback, run, scene, videoUrl, pickVideo } = useApp();
  const second = usePlayback(playback, (p) => Math.floor(Math.floor(p.frame) / p.run.fps));
  const drawn = !videoUrl && run.tracking !== null;

  let home = 0;
  let away = 0;
  if (scene) {
    const frame = (second + 1) * run.fps - 1;
    for (const goal of scene.goals) {
      if (goal.frame <= frame) {
        if (goal.side === 0) home++;
        else away++;
      }
    }
  }

  return (
    <section className="match" aria-label="Match view">
      {videoUrl ? (
        <MatchVideo url={videoUrl} />
      ) : drawn ? (
        <DrawnMatch />
      ) : (
        <div className="match__empty">
          <FilmIcon size={26} />
          <p className="match__empty-title">No picture for this match yet</p>
          <p className="match__empty-text">This run has no player tracking to draw from. Add the match video to watch it alongside the timeline.</p>
          <button type="button" className="btn btn--quiet" onClick={pickVideo}>
            Add the match video
          </button>
        </div>
      )}
      <div className="match__chip">
        <span className="num match__clock">{formatClock(second)}</span>
        {run.source !== "blank" ? (
          <>
            <span className="match__team">{run.home}</span>
            {scene ? (
              <span className="num match__score">
                {home} – {away}
              </span>
            ) : (
              <span className="match__versus">v</span>
            )}
            <span className="match__team">{run.away}</span>
          </>
        ) : null}
      </div>
      {drawn && run.source !== "demo" ? <div className="match__tag">Drawn from tracking</div> : null}
    </section>
  );
}
