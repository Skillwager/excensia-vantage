/**
 * Measures, in numbers only, whether the tracking and the camera in a run file sit on the video
 * they are meant for, and whether they are in step with it. No picture is shown, saved or
 * printed: the video is opened in a browser with no window, frames are read there, and what
 * comes back is a handful of shares.
 *
 *   node scripts/footage-align.mjs run.json match.mp4
 *
 * The run file needs tracking and a camera (python scripts/public_truth_run.py writes both).
 * Two things are measured on frames spread over the run:
 *
 * 1. Players. Just above the spot where the camera puts a tracked player's feet, the picture
 *    should not be grass, because a player is standing there; a few metres to the side it
 *    should be. The share of not-grass at the players, less the share beside them, is the
 *    score. It is taken with the tracking of the same frame and of frames up to half a second
 *    earlier and later. If tracking, camera and video are in step, the score is highest with
 *    no shift.
 * 2. Pitch lines. Along the lines the camera says are painted on the grass, the picture should
 *    be paler than a little to either side of them.
 *
 * PW_CHROME names a browser that plays the video's format (Chrome or Edge for MP4).
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright-core";

const [runPath, videoPath] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const executablePath = process.env.PW_CHROME;
if (!runPath || !videoPath || !existsSync(runPath) || !existsSync(videoPath)) {
  console.error("Give a run file with a camera and the video it belongs to.");
  process.exit(2);
}
if (!executablePath || !existsSync(executablePath)) {
  console.error("Set PW_CHROME to a browser that plays the video (Chrome or Edge).");
  process.exit(2);
}

const run = JSON.parse(readFileSync(runPath, "utf8"));
if (!run.camera || !run.tracking) {
  console.error("This run file has no camera or no tracking.");
  process.exit(2);
}
const fps = run.match.fps;
const frames = run.match.frames;
const offset = run.video?.offset_frames ?? 0;
const trackingEvery = run.tracking.every;
const cameraEvery = run.camera.every;
const sampleAt = (frame) => run.tracking.samples[Math.min(run.tracking.samples.length - 1, Math.max(0, Math.round(frame / trackingEvery)))];
const matrixAt = (frame) => run.camera.pitch_to_image[Math.min(run.camera.pitch_to_image.length - 1, Math.max(0, Math.round(frame / cameraEvery)))];

const SHIFTS = [-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12];
const PICKS = 36;
const picks = [];
for (let k = 0; k < PICKS; k++) {
  const frame = Math.round(((k + 0.5) / PICKS) * (frames - 1));
  if (frame + SHIFTS[0] < 0 || frame + SHIFTS[SHIFTS.length - 1] > frames - 1) continue;
  if (!matrixAt(frame) || !sampleAt(frame)) continue;
  picks.push(frame);
}

const length = run.match.pitch.length / 2;
const width = run.match.pitch.width / 2;
const lines = [];
const along = (x0, y0, x1, y1, n = 60) => {
  for (let i = 0; i <= n; i++) lines.push([x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * i) / n]);
};
along(-length, -width, length, -width, 120);
along(-length, width, length, width, 120);
along(-length, -width, -length, width);
along(length, -width, length, width);
along(0, -width, 0, width);
for (const side of [-1, 1]) {
  along(side * (length - 16.5), -20.16, side * (length - 16.5), 20.16, 40);
  along(side * length, -20.16, side * (length - 16.5), -20.16, 20);
  along(side * length, 20.16, side * (length - 16.5), 20.16, 20);
}
for (let a = 0; a < 72; a++) lines.push([9.15 * Math.cos((a / 72) * 2 * Math.PI), 9.15 * Math.sin((a / 72) * 2 * Math.PI)]);

const browser = await chromium.launch({ executablePath, headless: true, args: ["--mute-audio"] });
try {
  const page = await browser.newPage();
  await page.setContent('<input type="file" id="pick"><video id="film" muted playsinline preload="auto"></video>');
  // the video is named to the page as a local file, so its frames can be read back as numbers
  const session = await page.context().newCDPSession(page);
  const { root } = await session.send("DOM.getDocument");
  const { nodeId } = await session.send("DOM.querySelector", { nodeId: root.nodeId, selector: "#pick" });
  await session.send("DOM.setFileInputFiles", { files: [resolve(videoPath)], nodeId });
  const size = await page.evaluate(async () => {
    const video = document.getElementById("film");
    video.src = URL.createObjectURL(document.getElementById("pick").files[0]);
    await new Promise((done, fail) => {
      video.onloadeddata = done;
      video.onerror = () => fail(new Error("the video cannot be played in this browser"));
    });
    return [video.videoWidth, video.videoHeight, video.duration];
  });
  const scaleU = size[0] / run.camera.size[0];
  const scaleV = size[1] / run.camera.size[1];

  const totals = SHIFTS.map(() => ({ at: 0, beside: 0, n: 0 }));
  let onLine = 0;
  let offLine = 0;
  let lineCount = 0;
  let used = 0;
  for (const frame of picks) {
    const shifted = SHIFTS.map((shift) => ({ sample: sampleAt(frame + shift), matrix: matrixAt(frame + shift) }));
    if (shifted.some((s) => !s.matrix || !s.sample)) continue;
    const result = await page.evaluate(
      async ({ time, shifted, matrix, lines, scaleU, scaleV }) => {
        const video = document.getElementById("film");
        await new Promise((done) => {
          video.onseeked = done;
          video.currentTime = time;
        });
        const canvas = window.__canvas ?? (window.__canvas = document.createElement("canvas"));
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const g = canvas.getContext("2d", { willReadFrequently: true });
        g.drawImage(video, 0, 0);
        const W = canvas.width;
        const H = canvas.height;
        const data = g.getImageData(0, 0, W, H).data;
        const pixel = (m, x, y) => {
          const w = m[6] * x + m[7] * y + m[8];
          if (!(w > 1e-9)) return null;
          return [((m[0] * x + m[1] * y + m[2]) / w) * scaleU, ((m[3] * x + m[4] * y + m[5]) / w) * scaleV];
        };
        const grass = (u, v) => {
          const i = (Math.round(v) * W + Math.round(u)) * 4;
          return data[i + 1] > data[i] + 10 && data[i + 1] > data[i + 2] + 10;
        };
        // share of not-grass in a small upright patch whose bottom middle is (u, v)
        const patch = (u, v, tall, wide) => {
          let n = 0;
          let other = 0;
          for (let dv = 2; dv <= tall; dv += 2) {
            for (let du = -wide; du <= wide; du += 2) {
              const x = u + du;
              const y = v - dv;
              if (x < 1 || y < 1 || x > W - 2 || y > H - 2) continue;
              n++;
              if (!grass(x, y)) other++;
            }
          }
          return n ? other / n : null;
        };
        const players = shifted.map(({ sample, matrix: m }) => {
          let at = 0;
          let beside = 0;
          let n = 0;
          for (const p of [...sample.h, ...sample.a]) {
            if (!p) continue;
            const foot = pixel(m, p[0], p[1]);
            const next = pixel(m, p[0] + 1, p[1]);
            if (!foot || !next) continue;
            const metre = Math.max(4, Math.hypot(next[0] - foot[0], next[1] - foot[1]));
            if (foot[0] < 40 || foot[1] < 60 || foot[0] > W - 40 || foot[1] > H - 20) continue;
            const tall = Math.max(8, Math.min(40, metre * 1.5));
            const wide = Math.max(2, Math.min(8, metre * 0.25));
            const here = patch(foot[0], foot[1], tall, wide);
            const left = patch(foot[0] - metre * 3, foot[1], tall, wide);
            const right = patch(foot[0] + metre * 3, foot[1], tall, wide);
            if (here === null || left === null || right === null) continue;
            at += here;
            beside += (left + right) / 2;
            n++;
          }
          return { at, beside, n };
        });
        // pitch lines: paler on the line than three pixels either side of it
        let on = 0;
        let off = 0;
        let count = 0;
        const pale = (u, v) => {
          const i = (Math.round(v) * W + Math.round(u)) * 4;
          return (data[i] + data[i + 1] + data[i + 2]) / 3;
        };
        for (const [x, y] of lines) {
          const p = pixel(matrix, x, y);
          if (!p || p[0] < 6 || p[1] < 6 || p[0] > W - 7 || p[1] > H - 7) continue;
          let best = 0;
          for (let d = -1; d <= 1; d++) best = Math.max(best, pale(p[0], p[1] + d), pale(p[0] + d, p[1]));
          on += best;
          off += (pale(p[0] - 5, p[1] - 5) + pale(p[0] + 5, p[1] + 5) + pale(p[0] - 5, p[1] + 5) + pale(p[0] + 5, p[1] - 5)) / 4;
          count++;
        }
        return { players, on, off, count };
      },
      { time: (frame + offset + 0.5) / fps, shifted, matrix: matrixAt(frame), lines, scaleU, scaleV },
    );
    used++;
    result.players.forEach((p, i) => {
      totals[i].at += p.at;
      totals[i].beside += p.beside;
      totals[i].n += p.n;
    });
    onLine += result.on;
    offLine += result.off;
    lineCount += result.count;
  }

  console.log(`video ${size[0]}x${size[1]}, ${size[2].toFixed(1)} s; run of ${frames} frames starting at video frame ${offset}; ${used} frames read`);
  console.log("\nplayers: share of not-grass just above the feet, less the same a few metres to the side");
  const scores = totals.map((t) => (t.n ? (t.at - t.beside) / t.n : NaN));
  SHIFTS.forEach((shift, i) => {
    const t = totals[i];
    console.log(`  tracking ${shift >= 0 ? "+" : ""}${shift} frames: ${scores[i].toFixed(3)}   (at the players ${(t.at / t.n).toFixed(3)}, beside them ${(t.beside / t.n).toFixed(3)}, ${t.n} players)`);
  });
  const best = SHIFTS[scores.indexOf(Math.max(...scores))];
  const zero = scores[SHIFTS.indexOf(0)];
  console.log(`  highest with a shift of ${best} frames; with no shift ${zero.toFixed(3)}, half a second out ${((scores[0] + scores[scores.length - 1]) / 2).toFixed(3)}`);
  console.log(`\npitch lines: brightness on the lines ${(onLine / lineCount).toFixed(1)}, beside them ${(offLine / lineCount).toFixed(1)} (${lineCount} points on ${used} frames)`);
  const ok = Math.abs(best) <= 1 && zero > 0.15 && onLine / lineCount > offLine / lineCount + 8;
  console.log(`\n${ok ? "ok  " : "FAIL"}  the tracking and the camera ${ok ? "sit on this video and are in step with it" : "do not sit on this video as they should"}`);
  process.exitCode = ok ? 0 : 1;
} finally {
  await browser.close();
}
