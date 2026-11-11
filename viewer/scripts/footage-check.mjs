/**
 * Checks that the viewer draws its evidence on the match video in the right place, with a film
 * made here for the purpose. Twelve seconds of the simulated match are filmed by a camera whose
 * position, turn and lens are known: each frame of the film is drawn from the tracking through
 * that camera, players as upright figures in their team colours. The run file given to the
 * viewer carries the same camera, so the outlines, marks and trails the viewer draws on the
 * film have to land on those figures, and that is measured: at a few paused moments the film's
 * own picture is read at the place the viewer's camera puts each player's feet, and the colour
 * there has to be that player's team.
 *
 *   npm run build && node scripts/footage-check.mjs [--shots] [--keep=folder]
 *
 * --shots saves pictures in shots/footage/. --keep leaves the run file and the film in a folder,
 * for scripts/footage-align.mjs to be tried on. Nothing here is a real match.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const url = pathToFileURL(join(root, "dist", "index.html")).href;
const work = join(tmpdir(), `vantage-footage-${process.pid}`);
mkdirSync(work, { recursive: true });
const shots = process.argv.includes("--shots") ? join(root, "shots", "footage") : null;
if (shots) mkdirSync(shots, { recursive: true });
const keep = process.argv.find((a) => a.startsWith("--keep="))?.slice(7) ?? null;

function findChrome() {
  if (process.env.PW_CHROME && existsSync(process.env.PW_CHROME)) return process.env.PW_CHROME;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "ms-playwright");
  if (!existsSync(base)) return null;
  for (const prefix of ["chromium_headless_shell-", "chromium-"]) {
    const dirs = readdirSync(base)
      .filter((d) => d.startsWith(prefix))
      .sort((a, b) => Number(b.split("-").pop()) - Number(a.split("-").pop()));
    for (const dir of dirs) {
      for (const sub of ["chrome-headless-shell-win64/chrome-headless-shell.exe", "chrome-win64/chrome.exe", "chrome-win/chrome.exe", "chrome-linux/chrome"]) {
        if (existsSync(join(base, dir, sub))) return join(base, dir, sub);
      }
    }
  }
  return null;
}

let passed = 0;
const failures = [];
function check(name, ok, detail = "") {
  if (ok) passed++;
  else failures.push(name);
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
}

/* ── the camera: a lens at one spot beside the pitch, turning to follow play ── */

const SIZE = [1280, 720];
const FOCAL = 1050;
const SPOT = [0, -72, 26]; // metres: behind the near touchline, up a stand

function cameraLookingAt(targetX) {
  const target = [targetX, -3, 0];
  const norm = (v) => {
    const n = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / n, v[1] / n, v[2] / n];
  };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const forward = norm([target[0] - SPOT[0], target[1] - SPOT[1], target[2] - SPOT[2]]);
  const right = norm(cross(forward, [0, 0, 1]));
  const down = cross(forward, right);
  const r = [...right, ...down, ...forward]; // rows: right, down, forward
  // ground points (X, Y, 0, 1) to pixels: K [r1 r2 t], t = -R c
  const t = [0, 1, 2].map((row) => -(r[row * 3] * SPOT[0] + r[row * 3 + 1] * SPOT[1] + r[row * 3 + 2] * SPOT[2]));
  const k = [
    [FOCAL, 0, SIZE[0] / 2],
    [0, FOCAL, SIZE[1] / 2],
    [0, 0, 1],
  ];
  const m = [0, 1, 2].map((row) => [r[row * 3], r[row * 3 + 1], t[row]]);
  const h = k.map((krow) => [0, 1, 2].map((col) => krow[0] * m[0][col] + krow[1] * m[1][col] + krow[2] * m[2][col]));
  const scale = h[2][2];
  return { c: SPOT, r, f: FOCAL, matrix: h.flat().map((v) => Number((v / scale).toPrecision(7))) };
}

const browser = await chromium.launch({ executablePath: findChrome(), headless: true, args: ["--hide-scrollbars", "--force-color-profile=srgb", "--autoplay-policy=no-user-gesture-required"] });
try {
  const context = await browser.newContext({ viewport: { width: 1672, height: 941 }, deviceScaleFactor: 1, colorScheme: "dark", acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector(".poss", { timeout: 60_000 });
  await page.waitForFunction(() => !document.querySelector(".notice--busy"));

  // the simulated match, as the run file the viewer itself saves
  await page.getByRole("button", { name: "Files and settings" }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save the simulated match as a run file" }).click();
  const saved = join(work, "simulated.json");
  await (await download).saveAs(saved);
  await page.keyboard.press("Escape");
  const demo = JSON.parse(readFileSync(saved, "utf8"));

  /* ── twelve seconds of it, frame by frame ── */
  const fps = demo.match.fps;
  const seconds = 12;
  const count = seconds * fps;
  const first = Math.round((72 * 60 + 8) * fps);
  const tracking = demo.tracking;
  const lerp = (a, b, u) => (a && b ? [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u] : u < 0.5 ? a ?? null : b ?? null);
  const at = (frame) => {
    const i = Math.min(tracking.samples.length - 1, frame / tracking.every);
    const lo = tracking.samples[Math.floor(i)];
    const hi = tracking.samples[Math.min(tracking.samples.length - 1, Math.floor(i) + 1)];
    const u = i - Math.floor(i);
    const round = (p) => (p ? [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100] : null);
    return { b: round(lerp(lo.b, hi.b, u)), h: lo.h.map((p, n) => round(lerp(p, hi.h[n], u))), a: lo.a.map((p, n) => round(lerp(p, hi.a[n], u))) };
  };
  const samples = Array.from({ length: count }, (_, i) => at(first + i));
  // the camera follows the players' centre, smoothly
  const centre = samples.map((s) => {
    const xs = [...s.h, ...s.a].filter(Boolean).map((p) => p[0]);
    return xs.reduce((sum, x) => sum + x, 0) / Math.max(1, xs.length);
  });
  const cameras = samples.map((_, i) => {
    const lo = Math.max(0, i - 20);
    const hi = Math.min(count, i + 21);
    const smooth = centre.slice(lo, hi).reduce((sum, x) => sum + x, 0) / (hi - lo);
    return cameraLookingAt(smooth * 0.7 + 6 * Math.sin(i / 40));
  });

  // labels of that stretch, cut out of the match's run-length list
  const labels = [];
  let cursor = 0;
  for (const [code, length] of demo.labels) {
    const from = Math.max(first, cursor);
    const to = Math.min(first + count, cursor + length);
    if (to > from) labels.push([code, to - from]);
    cursor += length;
  }
  const run = {
    schema: "excensia.vantage.run/1",
    generator: { name: "synthetic-film", version: "1" },
    match: { title: "Synthetic film", home: demo.match.home, away: demo.match.away, fps, frames: count, pitch: demo.match.pitch },
    labels,
    tracking: { every: 1, samples },
    video: { offset_frames: 0 },
    camera: { size: SIZE, every: 1, pitch_to_image: cameras.map((c) => c.matrix) },
  };
  const runPath = join(work, "synthetic-film.run.json");
  writeFileSync(runPath, JSON.stringify(run));

  /* ── the film: every frame drawn from the tracking through its camera, recorded as it is drawn ── */
  const film = await page.evaluate(
    async ({ frames, pitch, fps, size }) => {
      const [W, H] = size;
      const canvas = document.createElement("canvas");
      canvas.width = W;
      canvas.height = H;
      const g = canvas.getContext("2d");
      const project = (cam, X, Y, Z) => {
        const dx = X - cam.c[0];
        const dy = Y - cam.c[1];
        const dz = Z - cam.c[2];
        const x = cam.r[0] * dx + cam.r[1] * dy + cam.r[2] * dz;
        const y = cam.r[3] * dx + cam.r[4] * dy + cam.r[5] * dz;
        const z = cam.r[6] * dx + cam.r[7] * dy + cam.r[8] * dz;
        return z > 0.1 ? [(cam.f * x) / z + W / 2, (cam.f * y) / z + H / 2] : null;
      };
      const line = (cam, points) => {
        g.beginPath();
        let started = false;
        for (const [X, Y] of points) {
          const p = project(cam, X, Y, 0);
          if (!p) {
            started = false;
            continue;
          }
          if (started) g.lineTo(p[0], p[1]);
          else g.moveTo(p[0], p[1]);
          started = true;
        }
        g.stroke();
      };
      const L = pitch.length / 2;
      const Wd = pitch.width / 2;
      const circle = Array.from({ length: 65 }, (_, a) => [9.15 * Math.cos((a / 64) * 2 * Math.PI), 9.15 * Math.sin((a / 64) * 2 * Math.PI)]);
      const draw = (i) => {
        const frame = frames[i];
        const cam = frame.cam;
        g.fillStyle = "#1c4f27";
        g.fillRect(0, 0, W, H);
        const corners = [[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]].map(([X, Y]) => project(cam, X, Y, 0));
        if (corners.every(Boolean)) {
          g.beginPath();
          corners.forEach((p, k) => (k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
          g.closePath();
          g.fillStyle = "#2f8a3d";
          g.fill();
        }
        g.strokeStyle = "#f0f2ee";
        g.lineWidth = 2;
        line(cam, [[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd], [-L, -Wd]]);
        line(cam, [[0, -Wd], [0, Wd]]);
        line(cam, circle);
        for (const s of [-1, 1]) {
          line(cam, [[s * L, -20.16], [s * (L - 16.5), -20.16], [s * (L - 16.5), 20.16], [s * L, 20.16]]);
          line(cam, [[s * L, -9.16], [s * (L - 5.5), -9.16], [s * (L - 5.5), 9.16], [s * L, 9.16]]);
        }
        const people = [];
        frame.h.forEach((p) => p && people.push([p[0], p[1], "#1e88ff"]));
        frame.a.forEach((p) => p && people.push([p[0], p[1], "#ff3d6e"]));
        people.sort((p, q) => q[1] - p[1]); // the far side first
        for (const [X, Y, colour] of people) {
          const foot = project(cam, X, Y, 0);
          const head = project(cam, X, Y, 1.8);
          if (!foot || !head) continue;
          const tall = foot[1] - head[1];
          const wide = Math.max(4, tall * 0.36);
          g.fillStyle = "rgba(0, 0, 0, 0.28)";
          g.beginPath();
          g.ellipse(foot[0] + wide * 0.5, foot[1], wide, wide * 0.3, 0, 0, Math.PI * 2);
          g.fill();
          g.fillStyle = colour;
          g.beginPath();
          g.roundRect(foot[0] - wide / 2, head[1], wide, tall, wide / 2);
          g.fill();
        }
        if (frame.b) {
          const p = project(cam, frame.b[0], frame.b[1], 0.11);
          if (p) {
            g.fillStyle = "#ffffff";
            g.strokeStyle = "#101418";
            g.lineWidth = 1;
            g.beginPath();
            g.arc(p[0], p[1], 3.4, 0, Math.PI * 2);
            g.fill();
            g.stroke();
          }
        }
        g.fillStyle = "#ffffff";
        g.font = "600 15px monospace";
        g.fillText(`synthetic film, frame ${i}`, 14, 24);
      };
      const recorder = new MediaRecorder(canvas.captureStream(fps), { mimeType: "video/webm;codecs=vp8", videoBitsPerSecond: 8_000_000 });
      const chunks = [];
      recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
      const stopped = new Promise((done) => (recorder.onstop = done));
      draw(0);
      recorder.start();
      const began = performance.now();
      await new Promise((finish) => {
        const tick = () => {
          const t = (performance.now() - began) / 1000;
          draw(Math.min(frames.length - 1, Math.floor(t * fps)));
          if (t < frames.length / fps + 0.3) requestAnimationFrame(tick);
          else finish();
        };
        tick();
      });
      recorder.stop();
      await stopped;
      return Array.from(new Uint8Array(await new Blob(chunks, { type: "video/webm" }).arrayBuffer()));
    },
    { frames: samples.map((s, i) => ({ ...s, cam: cameras[i] })), pitch: demo.match.pitch, fps, size: SIZE },
  );
  const filmPath = join(work, "synthetic-film.webm");
  writeFileSync(filmPath, Buffer.from(film));
  if (keep) {
    mkdirSync(keep, { recursive: true });
    copyFileSync(runPath, join(keep, "synthetic-film.run.json"));
    copyFileSync(filmPath, join(keep, "synthetic-film.webm"));
  }

  /* ── the viewer, with the run and its film ── */
  await page.locator('input[type="file"][accept*="json"]').setInputFiles(runPath);
  await page.waitForFunction(() => document.querySelector(".source__text")?.textContent?.includes("synthetic-film"), null, { timeout: 30_000 });
  check("without the film, the evidence is drawn on a pitch as before", (await page.locator(".shape__canvas").getAttribute("aria-label")) === "Both teams from above" && (await page.locator(".shape .seg").count()) === 0);
  await page.locator('input[type="file"][accept*="video"]').setInputFiles(filmPath);
  await page.waitForSelector("video.match__media", { timeout: 20_000 });
  await page.waitForFunction(() => document.querySelector("video.match__media").readyState >= 2, null, { timeout: 20_000 });
  await page.waitForTimeout(400);
  check("with the film, the team shape is shown on the footage and offers the board back", (await page.locator(".shape__canvas").getAttribute("aria-label")) === "Both teams' outlines on the match video" && (await page.locator(".shape .seg__btn").count()) === 2);
  check("a run that names no model calls its call a label, not a model estimate", (await page.locator(".insp__estimate-key").innerText()).toUpperCase().startsWith("LABEL"));

  const blur = () => page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());
  const frameNow = async () => Number((await page.locator(".insp__frame").innerText()).replace(/\D/g, ""));
  // the film's own picture, read where the viewer's camera puts each player's feet
  const measure = (frame) =>
    page.evaluate(
      async ({ sample, matrix }) => {
        const video = document.querySelector("video.match__media");
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const g = canvas.getContext("2d", { willReadFrequently: true });
        g.drawImage(video, 0, 0);
        const pixel = (x, y) => {
          const w = matrix[6] * x + matrix[7] * y + matrix[8];
          return [(matrix[0] * x + matrix[1] * y + matrix[2]) / w, (matrix[3] * x + matrix[4] * y + matrix[5]) / w];
        };
        const is = { home: (r, gr, b) => b > 150 && r < 130 && b > gr, away: (r, gr, b) => r > 170 && gr < 140 && b < 190 };
        let seen = 0;
        let right = 0;
        for (const [team, list] of [["home", sample.h], ["away", sample.a]]) {
          for (const p of list) {
            if (!p) continue;
            const [u, v] = pixel(p[0], p[1]);
            if (u < 8 || v < 16 || u > canvas.width - 8 || v > canvas.height - 4) continue;
            seen++;
            // a few points up the figure from its feet: any of them in the team's colour will do
            let hit = false;
            for (const up of [4, 7, 10]) {
              const d = g.getImageData(Math.round(u), Math.round(v - up), 1, 1).data;
              if (is[team](d[0], d[1], d[2])) hit = true;
            }
            if (hit) right++;
          }
        }
        return { seen, right, time: video.currentTime };
      },
      { sample: samples[frame], matrix: cameras[frame].matrix },
    );

  await blur();
  await page.keyboard.press("Home");
  for (const target of [50, 125, 225]) {
    while ((await frameNow()) + fps <= target) await page.keyboard.press("Shift+ArrowRight");
    while ((await frameNow()) < target) await page.keyboard.press("ArrowRight");
    await page.waitForFunction(() => !document.querySelector("video.match__media").seeking);
    await page.waitForTimeout(350);
    const here = await measure(target);
    const elsewhere = await measure(Math.min(count - 1, target + 40));
    check(
      `frame ${target}: the camera puts the players' feet on the players in the film`,
      here.seen >= 12 && here.right / here.seen >= 0.8 && here.right > elsewhere.right,
      `${here.right} of ${here.seen} on their own team's colour; with the tracking of 1.6 s later only ${elsewhere.right} of ${elsewhere.seen}; film at ${here.time.toFixed(2)} s`,
    );
    if (shots) {
      await page.screenshot({ path: join(shots, `frame-${target}.png`) });
      await page.locator(".tiles").screenshot({ path: join(shots, `frame-${target}-tiles.png`) });
      await page.locator(".shape").screenshot({ path: join(shots, `frame-${target}-shape.png`) });
    }
  }

  // the evidence chosen in the inspector is also drawn on the large film itself
  const inked = () =>
    page.evaluate(() => {
      const canvas = document.querySelector(".match__overlay");
      if (!canvas) return -1;
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let drawn = 0;
      for (let i = 3; i < data.length; i += 4 * 5) if (data[i] > 0) drawn++;
      return drawn;
    });
  const ringOnly = await inked();
  await page.locator(".chips .chip").nth(1).click();
  await page.waitForTimeout(250);
  const withOutlines = await inked();
  if (shots) await page.screenshot({ path: join(shots, "on-the-film-team-shape.png") });
  await page.locator(".chips .chip").nth(2).click();
  await page.waitForTimeout(250);
  const withRuns = await inked();
  if (shots) await page.screenshot({ path: join(shots, "on-the-film-runs.png") });
  await page.locator(".chips .chip").nth(0).click();
  await page.waitForTimeout(250);
  check(
    "the evidence chosen is drawn on the large film: a ring for the ball, the outlines, the runs",
    ringOnly >= 0 && withOutlines > ringOnly + 300 && withRuns > ringOnly + 150 && (await inked()) === ringOnly,
    `drawn points: ${ringOnly} with the ball alone, ${withOutlines} with team shape, ${withRuns} with the runs`,
  );
  await blur();

  // every tile and the panel are painted from the film, not left blank
  const painted = await page.evaluate(() =>
    [...document.querySelectorAll(".tile__canvas, .shape__canvas")].map((canvas) => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let green = 0;
      let total = 0;
      for (let i = 0; i < data.length; i += 4 * 9, total++) if (data[i + 1] > data[i] + 12 && data[i + 1] > data[i + 2] + 12) green++;
      return green / total;
    }),
  );
  check("the three tiles and the panel show the film's grass", painted.length === 4 && painted.every((share) => share > 0.3), painted.map((share) => share.toFixed(2)).join(", "));

  await page.locator(".shape .seg__btn", { hasText: "Board" }).click();
  await page.waitForTimeout(200);
  check("Board brings the drawn pitch back, Footage the film", (await page.locator(".shape__canvas").getAttribute("aria-label")) === "Both teams from above");
  if (shots) await page.locator(".shape").screenshot({ path: join(shots, "shape-board.png") });
  await page.locator(".shape .seg__btn", { hasText: "Footage" }).click();

  // playing: the picture and the marks keep together
  await page.locator(".tlp .scrub").focus();
  await page.keyboard.press("Home");
  await blur();
  await page.keyboard.press("Space");
  await page.waitForTimeout(2600);
  await page.keyboard.press("Space");
  await page.waitForFunction(() => !document.querySelector("video.match__media").seeking);
  await page.waitForTimeout(350);
  const stopped = await frameNow();
  const after = await measure(stopped);
  const away = await measure(Math.min(count - 1, stopped + 40));
  check(
    "after playing and pausing, the clock is on the film's frame and the marks are still on the players",
    stopped > 30 && stopped < 120 && Math.abs(Math.floor(after.time * fps + 1e-6) - stopped) <= 1 && after.right / Math.max(1, after.seen) >= 0.7 && after.right > away.right,
    `clock at frame ${stopped}, film at ${after.time.toFixed(2)} s; ${after.right} of ${after.seen} on their own team's colour, ${away.right} with the tracking of 1.6 s later`,
  );
  if (shots) await page.screenshot({ path: join(shots, "after-playing.png") });

  check("no errors in the page", errors.length === 0, errors.join(" | ").slice(0, 300));
} finally {
  await browser.close();
  rmSync(work, { recursive: true, force: true });
}
console.log(`\n${passed} checks passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
