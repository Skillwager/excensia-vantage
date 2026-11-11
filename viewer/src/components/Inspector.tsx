import { useMemo } from "react";
import { segmentAt } from "../model/derive";
import { explain, type EvidenceKind } from "../model/explain";
import { emptyMoment, emptyTrails, readTrails, sampleMoment, type Moment } from "../model/moment";
import { shapeIndex, shapesAt } from "../model/shape";
import { formatClock, formatCount, formatPercent } from "../model/time";
import { A, H, X, type Run } from "../model/types";
import { cameraLookAt } from "../render/camera";
import { drawBallFootage, drawShapeFootage, drawTimingFootage, footageAt } from "../render/footage";
import { drawLabel } from "../render/labels";
import { drawMatchView } from "../render/matchView";
import { C, FONT_SANS, alpha } from "../render/palette";
import { drawBallTop, drawCentreTop, drawHullTop, drawPitchTop, drawPlayersTop, fitPitch, framePitch, px, py, zoomPitch } from "../render/pitchTop";
import { UNSURE_BELOW, paintMiniStrip } from "../render/timeline";
import { useApp } from "../state/app";
import { useCanvas, usePlayback } from "../state/hooks";
import { ClockIcon, EyeIcon, EyeOffIcon, TeamIcon } from "./icons";

function ballSeenAt(run: Run, frame: number): boolean {
  const t = run.tracking;
  if (!t) return false;
  const i = Math.min(t.count - 1, Math.max(0, Math.round(frame / t.every)));
  return t.ball[i * 2] === t.ball[i * 2];
}

/** Where a zoomed tile should look: the ball, its estimate, or failing both the middle of the players. */
function focusOf(moment: Moment): { x: number; y: number } {
  if (moment.ballX === moment.ballX) return { x: moment.ballX, y: moment.ballY };
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const data of [moment.home, moment.away]) {
    for (let p = 0; p < data.length / 2; p++) {
      if (data[p * 2] !== data[p * 2]) continue;
      sx += data[p * 2];
      sy += data[p * 2 + 1];
      n++;
    }
  }
  return n ? { x: sx / n, y: sy / n } : { x: 0, y: 0 };
}

function emptyTile(ctx: CanvasRenderingContext2D, w: number, h: number, text: string) {
  ctx.fillStyle = C.pitch;
  ctx.fillRect(0, 0, w, h);
  ctx.font = `500 11px ${FONT_SANS}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = C.dim;
  ctx.fillText(text, w / 2, h / 2);
}

function caption(ctx: CanvasRenderingContext2D, w: number, h: number, text: string, colour: string) {
  drawLabel(ctx, text, w - 7, h - 12, ctx.getTransform().a, { size: 10, colour, halo: "rgba(6, 12, 9, 0.75)", align: "right", tracking: 0.5 });
}

/** 1. The ball, close up: the same picture as the match view, magnified around the ball. */
function BallTile() {
  const { playback, run, videoUrl } = useApp();
  const moment = useMemo(() => emptyMoment(run.tracking), [run]);
  const earlier = useMemo(() => emptyMoment(run.tracking), [run]);
  const keepers = useMemo<[number, number]>(() => shapeIndex(run)?.keepers ?? [-1, -1], [run]);
  const matrix = useMemo(() => new Float64Array(9), []);
  const ref = useCanvas(
    playback,
    (ctx, size, frame) => {
      const tracking = run.tracking;
      if (!tracking) return emptyTile(ctx, size.width, size.height, "No tracking");
      // with the match video and its camera, the close-up is cut out of the real picture
      const footage = videoUrl ? footageAt(playback, run, frame, matrix) : null;
      if (footage) {
        sampleMoment(tracking, footage.frame, run.fps, moment);
        if (moment.has && drawBallFootage(ctx, size.width, size.height, footage, moment, focusOf(moment))) {
          if (!moment.ballSeen) caption(ctx, size.width, size.height, "BALL NOT TRACKED", alpha(C.lime, 0.95));
          return;
        }
      }
      sampleMoment(tracking, Math.max(0, frame - run.fps * 0.125), run.fps, earlier);
      sampleMoment(tracking, frame, run.fps, moment);
      if (!moment.has) return emptyTile(ctx, size.width, size.height, "No tracking here");
      const focus = focusOf(moment);
      const look = cameraLookAt(tracking, run.fps, frame);
      drawMatchView(
        ctx,
        size.width,
        size.height,
        {
          pitch: run.pitch,
          moment,
          lookX: look.x,
          lookY: look.y,
          keepers,
          homeSlots: tracking.homeSlots,
          before: frame >= run.fps * 0.125 ? earlier : null,
          beforeSeconds: 0.125,
        },
        { labels: false, zoom: { x: focus.x, y: focus.y, metres: 17 } },
      );
      if (!moment.ballSeen) caption(ctx, size.width, size.height, "BALL HIDDEN", alpha(C.lime, 0.95));
    },
    [run, videoUrl],
  );
  return <canvas ref={ref} className="tile__canvas" role="img" aria-label="The ball, close up" />;
}

/** 2. Both teams' outlines on the whole pitch. */
function ShapeTile() {
  const { playback, run, videoUrl } = useApp();
  const moment = useMemo(() => emptyMoment(run.tracking), [run]);
  const matrix = useMemo(() => new Float64Array(9), []);
  const ref = useCanvas(
    playback,
    (ctx, size, frame) => {
      const tracking = run.tracking;
      if (!tracking) return emptyTile(ctx, size.width, size.height, "No tracking");
      // with the match video and its camera, the outlines are drawn on the real picture
      const footage = videoUrl ? footageAt(playback, run, frame, matrix) : null;
      if (footage) {
        sampleMoment(tracking, footage.frame, run.fps, moment);
        if (moment.has && drawShapeFootage(ctx, size.width, size.height, footage, moment, shapesAt(run, moment, footage.frame))) return;
      }
      sampleMoment(tracking, frame, run.fps, moment);
      const shapes = shapesAt(run, moment, frame);
      // frame the two teams rather than the whole pitch, so the outlines are large enough to read
      let lay = fitPitch(size.width, size.height, run.pitch, 6, 1.3);
      if (shapes.home && shapes.away) {
        const hl = run.pitch.length / 2;
        const hw = run.pitch.width / 2;
        const midX = (Math.min(shapes.home.minX, shapes.away.minX) + Math.max(shapes.home.maxX, shapes.away.maxX)) / 2;
        const midY = (Math.min(shapes.home.minY, shapes.away.minY) + Math.max(shapes.home.maxY, shapes.away.maxY)) / 2;
        const halfX = Math.max(30, (Math.max(shapes.home.maxX, shapes.away.maxX) - Math.min(shapes.home.minX, shapes.away.minX)) / 2 + 7);
        const halfY = Math.max(halfX * (size.height / size.width), (Math.max(shapes.home.maxY, shapes.away.maxY) - Math.min(shapes.home.minY, shapes.away.minY)) / 2 + 6);
        const cx = Math.max(-hl - 3 + halfX, Math.min(hl + 3 - halfX, midX));
        const cy = Math.max(-hw - 3 + halfY, Math.min(hw + 3 - halfY, midY));
        lay = framePitch(size.width, size.height, run.pitch, { minX: cx - halfX, maxX: cx + halfX, minY: cy - halfY, maxY: cy + halfY }, 0);
      }
      drawPitchTop(ctx, size.width, size.height, lay, 0.5);
      if (!moment.has) return;
      drawHullTop(ctx, lay, shapes.home, C.home, 0.2, 0.9);
      drawHullTop(ctx, lay, shapes.away, C.away, 0.2, 0.9);
      drawPlayersTop(ctx, lay, moment, { labels: false, keepers: [-1, -1], homeSlots: tracking.homeSlots, radius: 2.1, width: size.width });
      drawCentreTop(ctx, lay, shapes.home, C.home, 4);
      drawCentreTop(ctx, lay, shapes.away, C.away, 4);
    },
    [run, videoUrl],
  );
  return <canvas ref={ref} className="tile__canvas" role="img" aria-label="Both teams' shape on the pitch" />;
}

/** 3. The seconds before and after: where the ball and the players near it came from and go next. */
function TimingTile() {
  const { playback, run, videoUrl } = useApp();
  const buffers = useMemo(() => ({ now: emptyMoment(run.tracking), ...emptyTrails(run.tracking) }), [run]);
  const matrix = useMemo(() => new Float64Array(9), []);
  const ref = useCanvas(
    playback,
    (ctx, size, frame) => {
      const { width: w, height: h, dpr } = size;
      const stripH = 7;
      const tracking = run.tracking;
      const window = Math.round(10 * run.fps);
      if (!tracking) {
        emptyTile(ctx, w, h, "");
        // without tracking the labels either side still tell the story
        paintMiniStrip(ctx, 8, h / 2 - 7, w - 16, 14, dpr, run, frame - window, frame + window);
        ctx.fillStyle = C.lime;
        ctx.fillRect(Math.round(w / 2) - 1, h / 2 - 11, 2, 22);
        return;
      }
      const { now } = buffers;
      // with the match video and its camera, the runs are drawn on the real picture
      const footage = videoUrl ? footageAt(playback, run, frame, matrix) : null;
      const at = footage ? footage.frame : frame;
      sampleMoment(tracking, at, run.fps, now);
      if (!now.has) return emptyTile(ctx, w, h, "No tracking here");
      const focus = focusOf(now);

      // read the three seconds either side once
      readTrails(tracking, run.fps, run.frames - 1, at, buffers);
      const TRAIL_STEPS = buffers.steps;

      const strip = () => {
        // the labels ten seconds either side, with now in the middle
        paintMiniStrip(ctx, 0, h - stripH, w, stripH, dpr, run, frame - window, frame + window);
        ctx.fillStyle = "rgba(8, 10, 11, 0.6)";
        ctx.fillRect(Math.round(w / 2) - 2, h - stripH - 3, 4, stripH + 3);
        ctx.fillStyle = C.lime;
        ctx.fillRect(Math.round(w / 2) - 1, h - stripH - 3, 2, stripH + 3);
      };
      if (footage) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, w, h - stripH);
        ctx.clip();
        const drawn = drawTimingFootage(ctx, w, h - stripH, footage, now, focus, buffers, tracking.homeSlots, tracking.awaySlots);
        ctx.restore();
        if (drawn) return strip();
      }

      const lay = zoomPitch(w, h - stripH, run.pitch, focus.x, focus.y, 46);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, w, h - stripH);
      ctx.clip();
      drawPitchTop(ctx, w, h - stripH, lay, 0.45);

      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const trails = (data: Float32Array, slots: number, colour: string) => {
        for (const side of [-1, 1]) {
          ctx.beginPath();
          for (let p = 0; p < slots; p++) {
            let started = false;
            for (let j = 0; j <= TRAIL_STEPS; j++) {
              const k = TRAIL_STEPS + side * j;
              const x = data[(k * slots + p) * 2];
              if (!buffers.valid[k] || x !== x) break;
              const sx = px(lay, x);
              const sy = py(lay, data[(k * slots + p) * 2 + 1]);
              if (started) ctx.lineTo(sx, sy);
              else ctx.moveTo(sx, sy);
              started = true;
            }
          }
          ctx.lineWidth = side < 0 ? 1.6 : 1.1;
          ctx.setLineDash(side < 0 ? [] : [2, 3]);
          ctx.strokeStyle = alpha(colour, side < 0 ? 0.62 : 0.5);
          ctx.stroke();
        }
        ctx.setLineDash([]);
      };
      trails(buffers.home, tracking.homeSlots, C.home);
      trails(buffers.away, tracking.awaySlots, C.away);
      // the ball's path, where it was seen: solid behind it, dotted ahead
      for (const side of [-1, 1]) {
        ctx.beginPath();
        let drawing = false;
        for (let j = 0; j <= TRAIL_STEPS; j++) {
          const k = TRAIL_STEPS + side * j;
          const x = buffers.ball[k * 2];
          if (!buffers.valid[k] || x !== x) {
            drawing = false;
            continue;
          }
          if (drawing) ctx.lineTo(px(lay, x), py(lay, buffers.ball[k * 2 + 1]));
          else ctx.moveTo(px(lay, x), py(lay, buffers.ball[k * 2 + 1]));
          drawing = true;
        }
        ctx.lineWidth = 1.5;
        ctx.setLineDash(side < 0 ? [] : [2, 3]);
        ctx.strokeStyle = alpha("#ffffff", side < 0 ? 0.9 : 0.7);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      drawPlayersTop(ctx, lay, now, { labels: false, keepers: [-1, -1], homeSlots: tracking.homeSlots, radius: 3, width: w });
      drawBallTop(ctx, lay, now, { radius: 2.4, caption: false, maxDoubt: 5 });
      ctx.restore();
      strip();
    },
    [run, videoUrl],
  );
  return <canvas ref={ref} className="tile__canvas" role="img" aria-label="The three seconds before and after" />;
}

/** Why the model made this call: the estimate, how sure it is, and the evidence behind it. */
export function Inspector() {
  const { playback, run, evidence: kind, setEvidence: setKind } = useApp();
  // re-read once a second while playing, on every frame while stepping, and whenever the ball appears or disappears
  usePlayback(playback, (p) => {
    const f = Math.floor(p.frame);
    return p.playing ? Math.floor(f / p.run.fps) * 2 + (ballSeenAt(p.run, f) ? 1 : 0) : -f - 1;
  });
  usePlayback(playback, (p) => p.playing);

  const frame = Math.floor(playback.frame);
  const segment = segmentAt(run, frame);
  const text = useMemo(() => {
    const moment = emptyMoment(run.tracking);
    sampleMoment(run.tracking, frame, run.fps, moment);
    const shapes = shapesAt(run, moment, frame);
    const back = Math.max(0, frame - Math.round(3 * run.fps));
    const earlier = emptyMoment(run.tracking);
    sampleMoment(run.tracking, back, run.fps, earlier);
    return { ...explain(run, frame, moment, shapes, shapesAt(run, earlier, back)), ballSeen: moment.has && moment.ballSeen, tracked: moment.has };
  }, [run, frame]);

  const tone = segment.label === H ? "home" : segment.label === A ? "away" : segment.label === X ? "none" : "dead";
  const confidence = segment.confidence;
  const unsure = confidence !== null && confidence < UNSURE_BELOW;
  const weights = segment.evidence;
  const chips: { id: EvidenceKind; label: string; icon: JSX.Element; weight: number | null }[] = [
    {
      id: "ball",
      label: !text.tracked ? "The ball" : text.ballSeen ? "Ball seen" : "Ball hidden",
      icon: text.tracked && !text.ballSeen ? <EyeOffIcon size={17} /> : <EyeIcon size={17} />,
      weight: weights ? weights.ball : null,
    },
    { id: "shape", label: "Team shape", icon: <TeamIcon size={17} />, weight: weights ? weights.shape : null },
    { id: "timing", label: "Timing", icon: <ClockIcon size={17} />, weight: weights ? weights.timing : null },
  ];
  const tiles: { id: EvidenceKind; title: string; short?: string; body: JSX.Element }[] = [
    { id: "ball", title: "Ball zoom", body: <BallTile /> },
    { id: "shape", title: "Team shape", body: <ShapeTile /> },
    { id: "timing", title: "Before and after", short: "Timing", body: <TimingTile /> },
  ];

  return (
    <section className="panel insp" aria-label="Evidence inspector">
      <header className="panel-head">
        <h2 className="panel-title">Evidence inspector</h2>
        <p className="insp__when">
          <span className="insp__frame num">frame {formatCount(frame)}</span>
          <span className="num insp__clock">{formatClock(frame / run.fps)}</span>
        </p>
      </header>
      <div className="insp__body">
        <div className="insp__call">
          <p className="insp__estimate">
            <span className="insp__estimate-key">{run.hasSpellDetail ? "Model estimate:" : "Label:"}</span> <span className={`insp__estimate-value insp__estimate-value--${tone}`}>{text.estimate}</span>
          </p>
          <div className={unsure ? "conf conf--unsure" : "conf"} title={confidence === null ? "This run gives no confidence for this call" : undefined}>
            <span className="conf__label">Confidence</span>
            <span className="conf__meter" aria-hidden="true">
              <span className="conf__fill" style={{ transform: `scaleX(${confidence ?? 0})` }} />
            </span>
            <span className="conf__value num">{confidence === null ? "–" : formatPercent(confidence)}</span>
            {unsure ? <span className="conf__flag">Unsure</span> : null}
          </div>
        </div>
        <div className="chips" role="group" aria-label="Evidence">
          {chips.map((chip) => (
            <button
              key={chip.id}
              type="button"
              className="chip"
              aria-pressed={kind === chip.id}
              title={chip.weight === null ? undefined : `${chip.label}: ${formatPercent(chip.weight)} of this call`}
              onClick={() => setKind(chip.id)}
            >
              {chip.icon}
              <span className="chip__label">{chip.label}</span>
              {chip.weight === null ? null : <span className="chip__weight num">{formatPercent(chip.weight)}</span>}
            </button>
          ))}
        </div>
        <div className="tiles">
          {tiles.map((tile, i) => (
            <button key={tile.id} type="button" className="tile" aria-pressed={kind === tile.id} aria-label={`${tile.title}: show this evidence`} onClick={() => setKind(tile.id)}>
              <span className="tile__title">
                {i + 1}. <span className={tile.short ? "tile__long" : undefined}>{tile.title}</span>
                {tile.short ? <span className="tile__short">{tile.short}</span> : null}
              </span>
              {tile.body}
            </button>
          ))}
        </div>
        <p className="insp__why">
          {text[kind]}
        </p>
      </div>
    </section>
  );
}
