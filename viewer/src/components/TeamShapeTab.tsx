import { useMemo, useRef, useState } from "react";
import { emptyMoment, sampleMoment } from "../model/moment";
import { attackDir, shapeIndex, shapeSeries, shapesAt, type ShapeMetric, type ShapeSeries, type TeamShape } from "../model/shape";
import { formatClock } from "../model/time";
import { C, FONT_MONO, FONT_SANS, alpha } from "../render/palette";
import { drawBackLineTop, drawBallTop, drawCentreTop, drawHullTop, drawPitchTop, drawPlayersTop, fitPitch } from "../render/pitchTop";
import { useApp } from "../state/app";
import { useCanvas, usePlayback } from "../state/hooks";
import { SourceBar } from "./PossessionTab";
import { Transport } from "./Transport";

const METRICS: { id: ShapeMetric; label: string; unit: string; about: string }[] = [
  { id: "width", label: "Width", unit: "m", about: "How far each team stretches across the pitch." },
  { id: "depth", label: "Depth", unit: "m", about: "How far each team stretches from back to front." },
  { id: "line", label: "Back line", unit: "m", about: "How far each team's back four are from their own goal." },
];

const WINDOW_SECONDS = 90;

function Metres({ value }: { value: number | undefined }) {
  if (value === undefined || value !== value) return <>–</>;
  return (
    <>
      {Math.round(value)}
      <span className="unit"> m</span>
    </>
  );
}

function TeamFacts({ name, tone, shape, dir }: { name: string; tone: "home" | "away"; shape: TeamShape | null; dir: number }) {
  return (
    <div className="tsf">
      <p className="tsf__name">
        <span className={`legend__dot legend__dot--${tone}`} />
        {name}
        {dir !== 0 ? <span className="tsf__dir">attacking {dir > 0 ? "→" : "←"}</span> : null}
      </p>
      <dl className="tsf__list">
        <div>
          <dt>Width</dt>
          <dd className="num">
            <Metres value={shape?.width} />
          </dd>
        </div>
        <div>
          <dt>Depth</dt>
          <dd className="num">
            <Metres value={shape?.depth} />
          </dd>
        </div>
        <div>
          <dt>Back line</dt>
          <dd className="num">
            <Metres value={shape?.lineHeight} />
          </dd>
        </div>
        <div>
          <dt>Centre</dt>
          <dd className="num">
            <Metres value={shape?.centreHeight} />
          </dd>
        </div>
      </dl>
    </div>
  );
}

/** One measure of both teams, 90 seconds either side of now. */
function ShapeChart({ metric }: { metric: ShapeMetric }) {
  const { playback, run } = useApp();
  const cache = useRef<{ key: string; series: ShapeSeries } | null>(null);
  const ref = useCanvas(
    playback,
    (ctx, size, frame) => {
      const { width: w, height: h } = size;
      ctx.clearRect(0, 0, w, h);
      const fps = run.fps;
      // read a little more than is shown, and only again once the view has moved on
      const bucket = Math.floor(frame / (fps * 30));
      const key = `${playback.epoch}|${metric}|${bucket}`;
      if (!cache.current || cache.current.key !== key) {
        const from = (bucket * 30 - WINDOW_SECONDS - 2) * fps;
        const to = ((bucket + 1) * 30 + WINDOW_SECONDS + 2) * fps;
        cache.current = { key, series: shapeSeries(run, metric, from, to, 1) };
      }
      const { series } = cache.current;
      const now = frame / fps;
      const t0 = now - WINDOW_SECONDS;
      const t1 = now + WINDOW_SECONDS;
      const padL = 34;
      const padR = 8;
      const padT = 8;
      const padB = 22;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      let top = metric === "line" ? 60 : 50;
      for (let i = 0; i < series.t.length; i++) {
        if (series.t[i] < t0 || series.t[i] > t1) continue;
        if (series.home[i] > top) top = series.home[i];
        if (series.away[i] > top) top = series.away[i];
      }
      top = Math.ceil(top / 10) * 10;
      const X = (t: number) => padL + ((t - t0) / (t1 - t0)) * plotW;
      const Y = (v: number) => padT + (1 - v / top) * plotH;

      // quiet grid: three levels and the minutes
      ctx.font = `400 10.5px ${FONT_MONO}`;
      ctx.textBaseline = "middle";
      ctx.textAlign = "right";
      for (let v = 0; v <= top; v += top / 2) {
        const y = Math.round(Y(v)) + 0.5;
        ctx.strokeStyle = alpha(C.ink, v === 0 ? 0.16 : 0.07);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.fillStyle = C.dim;
        ctx.fillText(`${Math.round(v)}`, padL - 7, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.font = `500 10.5px ${FONT_SANS}`;
      for (let t = Math.ceil(t0 / 60) * 60; t <= t1; t += 60) {
        if (t < 0 || t > run.frames / fps) continue;
        const x = Math.round(X(t)) + 0.5;
        ctx.strokeStyle = alpha(C.ink, 0.07);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.fillStyle = C.dim;
        ctx.fillText(`${Math.round(t / 60)}'`, x, padT + plotH + 6);
      }

      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT - 2, plotW, plotH + 4);
      ctx.clip();
      const draw = (data: Float32Array, colour: string) => {
        ctx.beginPath();
        let on = false;
        for (let i = 0; i < series.t.length; i++) {
          const v = data[i];
          if (v !== v) {
            on = false;
            continue;
          }
          const x = X(series.t[i]);
          const y = Y(v);
          if (on) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          on = true;
        }
        ctx.lineWidth = 1.8;
        ctx.lineJoin = "round";
        ctx.strokeStyle = colour;
        ctx.stroke();
      };
      draw(series.home, C.home);
      draw(series.away, C.away);
      ctx.restore();

      // now
      const nx = Math.round(X(now));
      ctx.fillStyle = C.lime;
      ctx.fillRect(nx - 1, padT - 2, 2, plotH + 4);
    },
    [run, metric],
  );
  return <canvas ref={ref} className="tsc__canvas" role="img" aria-label="Team shape over time" />;
}

/** The big pitch: both teams, their outlines, centres and back lines at the selected moment. */
export function TeamShapeTab() {
  const { playback, run, settings, pickRunFile } = useApp();
  const [metric, setMetric] = useState<ShapeMetric>("width");
  const moment = useMemo(() => emptyMoment(run.tracking), [run]);
  const index = useMemo(() => shapeIndex(run), [run]);
  // the figures beside the pitch refresh four times a second while playing
  const tick = usePlayback(playback, (p) => Math.floor((Math.floor(p.frame) / p.run.fps) * 4));
  usePlayback(playback, (p) => p.playing);

  const canvasRef = useCanvas(
    playback,
    (ctx, size, frame) => {
      const lay = fitPitch(size.width, size.height, run.pitch, Math.max(10, Math.min(26, size.width * 0.03)), 1);
      drawPitchTop(ctx, size.width, size.height, lay, 0.7);
      const tracking = run.tracking;
      if (!tracking) return;
      sampleMoment(tracking, frame, run.fps, moment);
      if (!moment.has) return;
      const shapes = shapesAt(run, moment, frame);
      if (settings.outlines) {
        drawHullTop(ctx, lay, shapes.home, C.home);
        drawHullTop(ctx, lay, shapes.away, C.away);
      }
      drawBackLineTop(ctx, lay, shapes.home, C.home);
      drawBackLineTop(ctx, lay, shapes.away, C.away);
      const radius = Math.max(4, Math.min(8, lay.sy * 1.05));
      drawBallTop(ctx, lay, moment, { radius: radius * 0.6, caption: true });
      drawPlayersTop(ctx, lay, moment, {
        labels: size.width > 360,
        keepers: index?.keepers ?? [-1, -1],
        homeSlots: tracking.homeSlots,
        radius,
        width: size.width,
        labelSize: size.width > 700 ? 11.5 : 10,
      });
      drawCentreTop(ctx, lay, shapes.home, C.home, 7);
      drawCentreTop(ctx, lay, shapes.away, C.away, 7);
    },
    [run, settings.outlines],
  );

  const frame = Math.floor(playback.frame);
  const shapes = useMemo(() => {
    const m = emptyMoment(run.tracking);
    sampleMoment(run.tracking, frame, run.fps, m);
    return shapesAt(run, m, frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, tick, frame]);

  if (!run.tracking) {
    return (
      <div className="page">
        <header className="page__head">
          <div>
            <p className="label">Team shape</p>
            <h2 className="page__title">{run.title}</h2>
          </div>
        </header>
        <section className="panel empty">
          <p className="empty__title">No tracking in this run</p>
          <p className="empty__text">Team shape is worked out from where the players are. This run only has possession labels, so there is nothing to draw. Open a run file that includes tracking to see it.</p>
          <button type="button" className="btn btn--quiet" onClick={pickRunFile}>
            Open a run file
          </button>
        </section>
        <SourceBar />
      </div>
    );
  }

  const active = METRICS.find((m) => m.id === metric)!;
  return (
    <div className="page page--fill">
      <div className="ts">
        <section className="panel ts__pitch" aria-label="Team shape at the selected moment">
          <header className="panel-head">
            <h2 className="panel-title">
              Team shape <span className="panel-title__sub">(selected moment)</span>
            </h2>
            <ul className="legend legend--marks">
              <li>
                <span className="legend__cross" />
                Team centre
              </li>
              <li>
                <span className="legend__dash" />
                Back line
              </li>
              <li>
                <span className="num">{formatClock(frame / run.fps)}</span>
              </li>
            </ul>
          </header>
          <div className="ts__canvas-wrap">
            <canvas ref={canvasRef} className="ts__canvas" role="img" aria-label="Both teams from above" />
          </div>
        </section>
        <aside className="ts__side">
          <section className="panel ts__facts" aria-label="Shape now">
            <header className="panel-head panel-head--plain">
              <h3 className="panel-title">Shape now</h3>
            </header>
            <TeamFacts name={run.home} tone="home" shape={shapes.home} dir={attackDir(index, 0, frame)} />
            <TeamFacts name={run.away} tone="away" shape={shapes.away} dir={attackDir(index, 1, frame)} />
            <p className="panel-foot">Measured across the ten outfield players. Back line and centre are distances from the team's own goal.</p>
          </section>
          <section className="panel ts__chart" aria-label="Shape over time">
            <header className="panel-head panel-head--plain">
              <h3 className="panel-title">Over time</h3>
              <div className="seg" role="group" aria-label="Measure">
                {METRICS.map((m) => (
                  <button key={m.id} type="button" className="seg__btn" aria-pressed={metric === m.id} onClick={() => setMetric(m.id)}>
                    {m.label}
                  </button>
                ))}
              </div>
            </header>
            <p className="panel-note ts__about">{active.about} Metres, 90 seconds either side of now.</p>
            <div className="tsc">
              <ShapeChart metric={metric} />
            </div>
          </section>
        </aside>
        <section className="panel ts__transport" aria-label="Playback">
          <Transport compact />
        </section>
      </div>
      <SourceBar />
    </div>
  );
}
