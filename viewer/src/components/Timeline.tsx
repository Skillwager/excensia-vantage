import { useCallback, useEffect, useRef, useState, useSyncExternalStore, type PointerEvent as ReactPointerEvent } from "react";
import { lowerIndex, segmentAt } from "../model/derive";
import { labelWords } from "../model/explain";
import { formatClock, formatDuration, formatPercent } from "../model/time";
import { TL, UNSURE_BELOW, clampView, frameAtX, paintPlayhead, paintTimeline, xAtFrame, type TimelineView } from "../render/timeline";
import { useApp } from "../state/app";
import { fontEpoch, useCanvas } from "../state/hooks";
import { ZoomInIcon, ZoomOutIcon } from "./icons";

interface Tip {
  x: number;
  time: string;
  title: string;
  detail: string;
  why: string | null;
  tone: 0 | 1 | 2 | 3;
}

/** The legend, the zoom controls and the way back to the whole match. */
export function TimelineHead() {
  const { run, playback, timeline } = useApp();
  useSyncExternalStore(timeline.subscribe, timeline.key);
  const view = timeline.view;
  const minSpan = Math.round(run.fps * 20);

  const zoom = (factor: number) => {
    const centre = Math.min(view.start + view.span, Math.max(view.start, playback.frame));
    const anchor = centre >= view.start && centre <= view.start + view.span ? centre : view.start + view.span / 2;
    const span = view.span * factor;
    const k = (anchor - view.start) / view.span;
    timeline.set(clampView({ start: anchor - span * k, span }, run.frames, minSpan));
    playback.invalidate();
  };

  return (
    <div className="tlh">
      <h2 className="panel-title">Possession timeline</h2>
      <ul className="legend" aria-label="Legend">
        <li>
          <span className="legend__dot legend__dot--home" />
          {run.home}
        </li>
        <li>
          <span className="legend__dot legend__dot--away" />
          {run.away}
        </li>
        <li>
          <span className="legend__dot legend__dot--dead" />
          Dead ball
        </li>
        <li>
          <span className="legend__hatch" />
          Uncertainty
        </li>
      </ul>
      <div className="tlh__zoom">
        {timeline.zoomed ? (
          <>
            <span className="tlh__range num">
              {formatClock(view.start / run.fps)} – {formatClock((view.start + view.span) / run.fps)}
            </span>
            <button
              type="button"
              className="btn btn--chip"
              onClick={() => {
                timeline.reset();
                playback.invalidate();
              }}
            >
              Full match
            </button>
          </>
        ) : null}
        <button type="button" className="icon-btn icon-btn--sm" aria-label="Zoom out" title="Zoom out" disabled={!timeline.zoomed} onClick={() => zoom(2)}>
          <ZoomOutIcon size={15} />
        </button>
        <button type="button" className="icon-btn icon-btn--sm" aria-label="Zoom in" title="Zoom in" disabled={view.span <= minSpan + 1} onClick={() => zoom(0.5)}>
          <ZoomInIcon size={15} />
        </button>
      </div>
    </div>
  );
}

/** Three rows across the match: who has the ball, and when it is dead. Click or drag to move through it. */
export function Timeline() {
  const { run, playback, timeline } = useApp();
  const [tip, setTip] = useState<Tip | null>(null);
  const hoverX = useRef<number | null>(null);
  const cache = useRef<{ canvas: HTMLCanvasElement; key: string } | null>(null);
  const widthRef = useRef(0);
  const pointers = useRef(new Map<number, number>());
  const gesture = useRef<{ kind: "scrub"; resume: boolean } | { kind: "pinch"; distance: number; frame: number; span: number } | null>(null);
  const lastFrame = useRef(-1);
  const minSpan = Math.round(run.fps * 20);

  const canvasRef = useCanvas(
    playback,
    (ctx, size, frame) => {
      widthRef.current = size.width;
      let view = timeline.view;
      // keep the playhead in sight while zoomed in
      if (timeline.zoomed && gesture.current === null) {
        const x = xAtFrame(frame, size.width, view);
        const jumped = Math.abs(frame - lastFrame.current) > run.fps * 2;
        if (playback.playing && !jumped && x > size.width * 0.94) {
          view = clampView({ start: frame - view.span * 0.08, span: view.span }, run.frames, minSpan);
        } else if (x < 0 || x > size.width) {
          view = clampView({ start: frame - view.span / 2, span: view.span }, run.frames, minSpan);
        }
        if (view !== timeline.view) timeline.set(view);
      }
      lastFrame.current = frame;

      const key = `${playback.epoch}|${view.start}|${view.span}|${size.width}|${size.dpr}|${fontEpoch.value}`;
      let cached = cache.current;
      if (!cached || cached.key !== key) {
        const canvas = cached?.canvas ?? document.createElement("canvas");
        canvas.width = Math.round(size.width * size.dpr);
        canvas.height = Math.round(TL.height * size.dpr);
        const off = canvas.getContext("2d")!;
        off.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
        paintTimeline(off, size.width, size.dpr, run, view);
        cached = cache.current = { canvas, key };
      }
      ctx.clearRect(0, 0, size.width, size.height);
      ctx.drawImage(cached.canvas, 0, 0, size.width, TL.height);
      paintPlayhead(ctx, size.width, view, frame, hoverX.current);
    },
    [run],
  );

  const localX = (event: { clientX: number }) => {
    const box = canvasRef.current!.getBoundingClientRect();
    return event.clientX - box.left;
  };

  const describe = useCallback(
    (x: number): Tip => {
      const width = Math.max(1, widthRef.current);
      const view = timeline.view;
      const frame = Math.min(run.frames - 1, Math.max(0, Math.floor(frameAtX(x, width, view))));
      const segment = segmentAt(run, frame);
      const r = lowerIndex(run.runStart, frame);
      const runEnd = r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames;
      const seconds = (runEnd - run.runStart[r]) / run.fps;
      let detail = `for ${formatDuration(seconds)}`;
      if (segment.confidence !== null) {
        detail += segment.confidence < UNSURE_BELOW ? ` · ${formatPercent(segment.confidence)} confident, unsure` : ` · ${formatPercent(segment.confidence)} confident`;
      }
      // a change of possession close to the pointer explains itself
      let why: string | null = null;
      if (run.changes.length) {
        const i = lowerIndex(run.changeFrame, frame);
        for (const change of [run.changes[i], run.changes[i + 1]]) {
          if (change && Math.abs(xAtFrame(change.frame, width, view) - x) <= 5) {
            why = change.why ?? `${labelWords(run, change.from)} to ${labelWords(run, change.to)}`;
            break;
          }
        }
      }
      return { x, time: formatClock(frame / run.fps), title: labelWords(run, segment.label), detail, why, tone: segment.label };
    },
    [run, timeline],
  );

  const scrubTo = (x: number) => {
    const width = Math.max(1, widthRef.current);
    playback.seek(Math.min(run.frames - 1, Math.max(0, frameAtX(Math.min(width, Math.max(0, x)), width, timeline.view))));
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    const canvas = event.currentTarget;
    canvas.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, localX(event));
    if (pointers.current.size === 1) {
      const resume = playback.playing;
      if (resume) playback.pause();
      gesture.current = { kind: "scrub", resume };
      scrubTo(localX(event));
      setTip(describe(localX(event)));
    } else if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      const view = timeline.view;
      const resume = gesture.current?.kind === "scrub" ? gesture.current.resume : false;
      gesture.current = { kind: "pinch", distance: Math.max(12, Math.abs(a - b)), frame: frameAtX((a + b) / 2, widthRef.current, view), span: view.span };
      if (resume) playback.play();
      setTip(null);
    }
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const x = localX(event);
    if (pointers.current.has(event.pointerId)) pointers.current.set(event.pointerId, x);
    const g = gesture.current;
    if (g?.kind === "pinch" && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()];
      const width = Math.max(1, widthRef.current);
      const span = g.span * (g.distance / Math.max(12, Math.abs(a - b)));
      const mid = (a + b) / 2;
      const next = clampView({ start: g.frame - (mid / width) * span, span }, run.frames, minSpan);
      timeline.set(next);
      playback.invalidate();
      return;
    }
    if (g?.kind === "scrub") {
      scrubTo(x);
      setTip(describe(x));
      return;
    }
    if (event.pointerType === "mouse") {
      hoverX.current = x;
      setTip(describe(x));
      playback.invalidate();
    }
  };

  const endPointer = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    pointers.current.delete(event.pointerId);
    const g = gesture.current;
    if (pointers.current.size === 0) {
      if (g?.kind === "scrub" && g.resume) playback.play();
      gesture.current = null;
      if (event.pointerType !== "mouse") setTip(null);
    } else if (g?.kind === "pinch" && pointers.current.size < 2) {
      gesture.current = null;
    }
  };

  const onLeave = () => {
    if (gesture.current) return;
    hoverX.current = null;
    setTip(null);
    playback.invalidate();
  };

  // the wheel zooms about the pointer; a sideways wheel (or shift) pans
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const width = Math.max(1, widthRef.current);
      const view = timeline.view;
      const box = canvas.getBoundingClientRect();
      const x = event.clientX - box.left;
      const unit = event.deltaMode === 1 ? 32 : 1;
      const sideways = Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey;
      let next: TimelineView;
      if (sideways && !event.ctrlKey) {
        const delta = (Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY) * unit;
        next = { start: view.start + (delta / width) * view.span, span: view.span };
      } else {
        const span = view.span * Math.exp(Math.max(-1.2, Math.min(1.2, event.deltaY * unit * (event.ctrlKey ? 0.01 : 0.0016))));
        const anchor = frameAtX(x, width, view);
        next = { start: anchor - (x / width) * span, span };
      }
      timeline.set(clampView(next, run.frames, minSpan));
      playback.invalidate();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [canvasRef, playback, run, timeline, minSpan]);

  // a new run starts on the whole match
  useEffect(() => {
    cache.current = null;
  }, [run]);

  const width = widthRef.current || 1;
  return (
    <div className="tl">
      <div className="tl__labels" style={{ height: TL.height }} aria-hidden="true">
        <span className="tl__label tl__label--home" style={{ top: TL.rowY[0], height: TL.rowH }} title={run.home}>
          {run.home}
        </span>
        <span className="tl__label tl__label--away" style={{ top: TL.rowY[1], height: TL.rowH }} title={run.away}>
          {run.away}
        </span>
        <span className="tl__label tl__label--dead" style={{ top: TL.rowY[2], height: TL.rowH }}>
          Dead ball
        </span>
      </div>
      <div className="tl__plot" style={{ height: TL.height }}>
        <canvas
          ref={canvasRef}
          className="tl__canvas"
          role="slider"
          tabIndex={0}
          aria-label="Possession timeline. Drag to move through the match."
          aria-valuemin={0}
          aria-valuemax={Math.round(run.frames / run.fps)}
          aria-valuenow={Math.round(playback.frame / run.fps)}
          aria-valuetext={`${formatClock(playback.frame / run.fps)}, ${labelWords(run, segmentAt(run, Math.floor(playback.frame)).label)}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endPointer}
          onPointerCancel={endPointer}
          onPointerLeave={onLeave}
          onDoubleClick={() => {
            timeline.reset();
            playback.invalidate();
          }}
        />
        {tip ? (
          <div className="tip" style={{ left: Math.min(width - 8, Math.max(8, tip.x)) }} data-side={tip.x > width - 150 ? "left" : tip.x < 150 ? "right" : "centre"} role="presentation">
            <span className="tip__time num">{tip.time}</span>
            <span className={`tip__title tip__title--${tip.tone}`}>{tip.title}</span>
            <span className="tip__detail">{tip.detail}</span>
            {tip.why ? <span className="tip__why">{tip.why}</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
