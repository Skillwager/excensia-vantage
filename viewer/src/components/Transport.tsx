import { useEffect, useRef, useState, useSyncExternalStore, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { formatClock, formatTotalClock } from "../model/time";
import { useApp } from "../state/app";
import { usePlayback } from "../state/hooks";
import { SPEEDS } from "../state/playback";
import { BackIcon, ChevronDownIcon, ForwardIcon, NextChangeIcon, PauseIcon, PlayIcon, PrevChangeIcon } from "./icons";

const speedText = (speed: number) => `${speed}x`;

function SpeedMenu() {
  const { playback } = useApp();
  const speed = usePlayback(playback, (p) => p.speed);
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close, true);
    list.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.focus();
    return () => document.removeEventListener("pointerdown", close, true);
  }, [open]);

  const onKeyDown = (event: KeyboardEvent<HTMLUListElement>) => {
    const items = [...(list.current?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])];
    const at = items.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      event.stopPropagation();
      items[(at + (event.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      wrap.current?.querySelector("button")?.focus();
    }
  };

  return (
    <div className="speed" ref={wrap}>
      <button type="button" className="speed__btn" aria-haspopup="listbox" aria-expanded={open} aria-label={`Speed ${speedText(speed)}`} onClick={() => setOpen((v) => !v)}>
        <span className="num">{speedText(speed)}</span>
        <ChevronDownIcon size={15} />
      </button>
      {open ? (
        <ul className="speed__list" role="listbox" aria-label="Speed" ref={list} onKeyDown={onKeyDown}>
          {SPEEDS.map((s) => (
            <li
              key={s}
              role="option"
              tabIndex={-1}
              aria-selected={s === speed}
              className="speed__item num"
              onClick={() => {
                playback.setSpeed(s);
                setOpen(false);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  event.stopPropagation();
                  playback.setSpeed(s);
                  setOpen(false);
                  wrap.current?.querySelector("button")?.focus();
                }
              }}
            >
              {speedText(s)}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** The whole match as one line: drag to go anywhere. When the timeline is zoomed in, its window is marked here. */
function Scrubber() {
  const { playback, run, timeline } = useApp();
  const track = useRef<HTMLDivElement>(null);
  const fill = useRef<HTMLDivElement>(null);
  const knob = useRef<HTMLDivElement>(null);
  const drag = useRef<{ resume: boolean } | null>(null);
  useSyncExternalStore(timeline.subscribe, timeline.key);
  const second = usePlayback(playback, (p) => Math.floor(p.frame / p.run.fps));

  useEffect(
    () =>
      playback.onFrame((frame) => {
        const k = run.frames > 1 ? frame / (run.frames - 1) : 0;
        if (fill.current) fill.current.style.transform = `scaleX(${k})`;
        if (knob.current) knob.current.style.left = `${k * 100}%`;
      }),
    [playback, run],
  );

  const seekAt = (clientX: number) => {
    const box = track.current!.getBoundingClientRect();
    const k = Math.min(1, Math.max(0, (clientX - box.left) / Math.max(1, box.width)));
    playback.seek(k * (run.frames - 1));
  };
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const resume = playback.playing;
    if (resume) playback.pause();
    drag.current = { resume };
    seekAt(event.clientX);
  };
  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (drag.current) seekAt(event.clientX);
  };
  const onPointerUp = () => {
    if (drag.current?.resume) playback.play();
    drag.current = null;
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 60 : 5;
    if (event.key === "ArrowRight" || event.key === "ArrowUp") playback.seekSeconds(step);
    else if (event.key === "ArrowLeft" || event.key === "ArrowDown") playback.seekSeconds(-step);
    else if (event.key === "Home") playback.seek(0);
    else if (event.key === "End") playback.seek(run.frames - 1);
    else return;
    event.preventDefault();
    event.stopPropagation();
  };

  const view = timeline.view;
  return (
    <div
      className="scrub"
      role="slider"
      tabIndex={0}
      aria-label="Position in the match"
      aria-valuemin={0}
      aria-valuemax={Math.round(run.frames / run.fps)}
      aria-valuenow={second}
      aria-valuetext={`${formatClock(second)} of ${formatTotalClock(run.frames, run.fps)}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    >
      <div className="scrub__track" ref={track}>
        {timeline.zoomed ? (
          <div
            className="scrub__window"
            title="The part of the match the timeline is showing"
            style={{ left: `${(view.start / run.frames) * 100}%`, width: `${Math.max(0.6, (view.span / run.frames) * 100)}%` }}
          />
        ) : null}
        <div className="scrub__fill" ref={fill} />
        <div className="scrub__knob" ref={knob} />
      </div>
    </div>
  );
}

/** Play, pause, step and speed. */
export function Transport({ compact = false }: { compact?: boolean }) {
  const { playback, run } = useApp();
  const playing = usePlayback(playback, (p) => p.playing);
  const second = usePlayback(playback, (p) => Math.floor(Math.floor(p.frame) / p.run.fps));
  const hasChanges = run.changes.length > 0;

  return (
    <div className={compact ? "tp tp--compact" : "tp"} role="group" aria-label="Playback">
      <button type="button" className="tp__play" aria-label={playing ? "Pause" : "Play"} title={playing ? "Pause (space)" : "Play (space)"} onClick={() => playback.toggle()}>
        {playing ? <PauseIcon size={20} /> : <PlayIcon size={20} />}
      </button>
      <div className="tp__time num" aria-label={`${formatClock(second)} of ${formatTotalClock(run.frames, run.fps)}`}>
        <span className="tp__now">{formatClock(second)}</span>
        <span className="tp__sep">/</span>
        <span className="tp__total">{formatTotalClock(run.frames, run.fps)}</span>
      </div>
      <Scrubber />
      <div className="tp__steps">
        <button type="button" className="tp__btn tp__btn--icon" aria-label="Previous change" title="Previous change ( [ )" disabled={!hasChanges} onClick={() => playback.toPreviousChange()}>
          <PrevChangeIcon size={17} />
        </button>
        <button type="button" className="tp__btn" aria-label="Back 5 seconds" title="Back 5 seconds" onClick={() => playback.seekSeconds(-5)}>
          <BackIcon size={17} />
          <span className="num">5s</span>
        </button>
        <button type="button" className="tp__btn" aria-label="Forward 5 seconds" title="Forward 5 seconds" onClick={() => playback.seekSeconds(5)}>
          <span className="num">5s</span>
          <ForwardIcon size={17} />
        </button>
        <button type="button" className="tp__btn tp__btn--icon" aria-label="Next change" title="Next change ( ] )" disabled={!hasChanges} onClick={() => playback.toNextChange()}>
          <NextChangeIcon size={17} />
        </button>
      </div>
      <SpeedMenu />
    </div>
  );
}
