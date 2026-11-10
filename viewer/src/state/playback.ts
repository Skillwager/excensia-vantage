import { nearestChange, nextChange, previousChange } from "../model/derive";
import type { Run } from "../model/types";

export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

type Listener = () => void;
type Drawer = (frame: number) => void;

/**
 * The one clock of the viewer: which frame is showing, whether it is playing and how fast.
 * With a video attached the video is the clock while playing; without one the clock runs itself.
 * Canvases register a drawer and repaint on animation frames; React reads slower-moving values through `subscribe`.
 */
export class Playback {
  run: Run;
  /** The frame showing. A fraction while playing, so motion is smooth between frames. */
  frame = 0;
  playing = false;
  speed = 1;
  /** Bumps whenever a new run is loaded. */
  epoch = 0;
  private stopAt: number | null = null;
  private video: HTMLVideoElement | null = null;
  private listeners = new Set<Listener>();
  private drawers = new Set<Drawer>();
  private raf = 0;
  private last = 0;
  private dirty = true;

  constructor(run: Run) {
    this.run = run;
  }

  /* ── subscriptions ── */

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** Registers a canvas painter. It is called on the next animation frame and on every one while anything moves. */
  onFrame(drawer: Drawer): () => void {
    this.drawers.add(drawer);
    this.invalidate();
    return () => this.drawers.delete(drawer);
  }

  /** Asks for a repaint. */
  invalidate = (): void => {
    this.dirty = true;
    this.schedule();
  };

  private emit(): void {
    for (const listener of this.listeners) listener();
  }

  private schedule(): void {
    if (this.raf || typeof requestAnimationFrame === "undefined") return;
    this.raf = requestAnimationFrame(this.tick);
  }

  private tick = (now: number): void => {
    this.raf = 0;
    const before = this.frame;
    if (this.playing) {
      const video = this.activeVideo();
      if (video) {
        this.frame = this.clamp(video.currentTime * this.run.fps - this.run.videoOffset);
        if (video.ended) this.pauseQuietly();
      } else {
        const elapsed = this.last ? Math.min(0.1, (now - this.last) / 1000) : 0;
        this.frame = this.clamp(this.frame + elapsed * this.run.fps * this.speed);
      }
      this.last = now;
      if (this.stopAt !== null && this.frame >= this.stopAt) {
        this.frame = this.stopAt;
        this.pauseQuietly();
      } else if (this.frame >= this.run.frames - 1) {
        this.frame = this.run.frames - 1;
        this.pauseQuietly();
      }
    }
    const moved = this.frame !== before;
    if (moved || this.dirty) {
      this.dirty = false;
      for (const drawer of this.drawers) drawer(this.frame);
    }
    if (moved) this.emit();
    if (this.playing) this.schedule();
  };

  /* ── the run and the video ── */

  setRun(run: Run, frame = 0): void {
    this.pauseQuietly();
    this.run = run;
    this.epoch++;
    this.frame = this.clamp(frame);
    this.syncVideo();
    this.invalidate();
    this.emit();
  }

  attachVideo(video: HTMLVideoElement | null): void {
    if (this.video === video) return;
    if (this.video && this.playing) this.video.pause();
    this.video = video;
    if (video) {
      video.playbackRate = this.speed;
      this.syncVideo();
      if (this.playing) void video.play().catch(() => this.pause());
    }
    this.invalidate();
  }

  private activeVideo(): HTMLVideoElement | null {
    const video = this.video;
    return video && video.readyState >= 1 ? video : null;
  }

  private syncVideo(): void {
    const video = this.activeVideo();
    if (!video) return;
    // aim for the middle of the frame, so rounding never lands on the frame before
    const time = (this.frame + this.run.videoOffset + 0.5) / this.run.fps;
    const end = Number.isFinite(video.duration) ? Math.max(0, video.duration - 0.001) : time;
    video.currentTime = Math.min(end, Math.max(0, time));
  }

  /* ── transport ── */

  private clamp(frame: number): number {
    if (!(frame > 0)) return 0;
    const last = this.run.frames - 1;
    return frame > last ? last : frame;
  }

  play(): void {
    if (this.playing) return;
    if (this.frame >= this.run.frames - 1) this.frame = 0;
    this.playing = true;
    this.last = 0;
    const video = this.activeVideo();
    if (video) {
      this.syncVideo();
      video.playbackRate = this.speed;
      void video.play().catch(() => this.pause());
    }
    this.schedule();
    this.emit();
  }

  private pauseQuietly(): void {
    if (!this.playing) return;
    this.playing = false;
    this.stopAt = null;
    this.video?.pause();
  }

  pause(): void {
    if (!this.playing && this.stopAt === null) return;
    this.pauseQuietly();
    this.stopAt = null;
    this.frame = Math.floor(this.frame);
    this.invalidate();
    this.emit();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  /** Moves to a frame. Playback carries on from there if it was running. */
  seek(frame: number): void {
    this.stopAt = null;
    this.frame = this.clamp(this.playing ? frame : Math.floor(frame));
    this.last = 0;
    this.syncVideo();
    this.invalidate();
    this.emit();
  }

  /** Steps by whole frames and stops, for looking closely. */
  step(frames: number): void {
    this.pauseQuietly();
    this.seek(Math.floor(this.frame) + frames);
  }

  seekSeconds(seconds: number): void {
    this.seek(this.frame + seconds * this.run.fps);
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    if (this.video) this.video.playbackRate = speed;
    this.last = 0;
    this.emit();
  }

  toNextChange(): void {
    const change = nextChange(this.run, this.frame);
    if (change) this.seek(change.frame);
  }

  toPreviousChange(): void {
    // while playing, a press just after a change goes to the one before it rather than back to the same one
    const from = this.playing ? this.frame - this.run.fps : this.frame;
    const change = previousChange(this.run, Math.max(0, from));
    this.seek(change ? change.frame : 0);
  }

  /** Jumps to three seconds before the nearest change and plays through to three seconds after it. */
  replayTransition(): boolean {
    const change = nearestChange(this.run, this.frame);
    if (!change) return false;
    const pad = Math.round(3 * this.run.fps);
    this.pauseQuietly();
    this.seek(change.frame - pad);
    this.stopAt = this.clamp(change.frame + pad);
    this.play();
    return true;
  }

  dispose(): void {
    if (this.raf && typeof cancelAnimationFrame !== "undefined") cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.playing = false;
    this.video?.pause();
  }
}
