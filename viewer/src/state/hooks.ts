import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { Playback } from "./playback";

/** Reads a slow-moving value off the clock. The component re-renders only when that value changes. */
export function usePlayback<T>(playback: Playback, select: (playback: Playback) => T): T {
  return useSyncExternalStore(playback.subscribe, () => select(playback));
}

export interface CanvasSize {
  /** CSS pixels. */
  width: number;
  height: number;
  dpr: number;
}

/** A ref to hand to a <canvas>, which also remembers the element. */
export interface CanvasRef {
  (node: HTMLCanvasElement | null): void;
  current: HTMLCanvasElement | null;
}

/**
 * Keeps a canvas the size of its box at the screen's pixel density and calls `draw` on every frame the clock paints.
 * `draw` reads everything it needs from the latest render, so it is never stale.
 */
export function useCanvas(
  playback: Playback,
  draw: (ctx: CanvasRenderingContext2D, size: CanvasSize, frame: number) => void,
  deps: readonly unknown[] = [],
): CanvasRef {
  const [element, setElement] = useState<HTMLCanvasElement | null>(null);
  const ref = useMemo(() => {
    const fn = ((node: HTMLCanvasElement | null) => {
      fn.current = node;
      setElement(node);
    }) as CanvasRef;
    fn.current = null;
    return fn;
  }, []);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const size = useRef<CanvasSize>({ width: 0, height: 0, dpr: 1 });

  useLayoutEffect(() => {
    const canvas = element;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    size.current = { width: 0, height: 0, dpr: 1 };

    const measure = () => {
      const box = canvas.getBoundingClientRect();
      const dpr = Math.min(2.5, Math.max(1, window.devicePixelRatio || 1));
      const width = Math.max(1, Math.round(box.width));
      const height = Math.max(1, Math.round(box.height));
      const s = size.current;
      if (s.width === width && s.height === height && s.dpr === dpr) return;
      size.current = { width, height, dpr };
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      playback.invalidate();
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(canvas);

    const off = playback.onFrame((frame) => {
      const s = size.current;
      if (s.width < 2 || s.height < 2) return;
      ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
      drawRef.current(ctx, s, frame);
    });
    return () => {
      observer.disconnect();
      off();
    };
  }, [playback, element]);

  // repaint when the things the picture depends on change
  useEffect(() => {
    playback.invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

/** Goes up each time web fonts arrive. Anything that caches drawn text keys its cache on it. */
export const fontEpoch = { value: 0 };

/** Repaints every canvas once the web fonts arrive, so labels are not left in a fallback face. */
export function useFontRepaint(playback: Playback | null, onFonts?: () => void): void {
  useEffect(() => {
    const fonts = document.fonts;
    if (!fonts || !playback) return;
    let live = true;
    const repaint = () => {
      if (!live) return;
      fontEpoch.value++;
      onFonts?.();
      playback.invalidate();
    };
    void fonts.ready.then(repaint);
    fonts.addEventListener?.("loadingdone", repaint);
    return () => {
      live = false;
      fonts.removeEventListener?.("loadingdone", repaint);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playback]);
}

/** Paints a canvas that does not follow the clock: once when it appears, and again when its box or `deps` change. */
export function useStaticCanvas(draw: (ctx: CanvasRenderingContext2D, size: CanvasSize) => void, deps: readonly unknown[]): (node: HTMLCanvasElement | null) => void {
  const [element, setElement] = useState<HTMLCanvasElement | null>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useLayoutEffect(() => {
    const canvas = element;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const paint = () => {
      const box = canvas.getBoundingClientRect();
      const dpr = Math.min(2.5, Math.max(1, window.devicePixelRatio || 1));
      const width = Math.max(1, Math.round(box.width));
      const height = Math.max(1, Math.round(box.height));
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      drawRef.current(ctx, { width, height, dpr });
    };
    paint();
    const observer = new ResizeObserver(paint);
    observer.observe(canvas);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [element, ...deps]);
  return setElement;
}
