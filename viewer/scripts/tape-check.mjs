/**
 * Checks, in numbers only, that a page made with
 *
 *   python -m vantage export --data-dir … --out run.json --page match.html --video match.mp4
 *
 * plays the match video in step with the analysis. It prints durations and clock readings and
 * takes no pictures, so it can be run on a match whose footage may not be shown around.
 *
 * It needs a browser that plays MP4; the Chromium that comes with Playwright does not. Point
 * PW_CHROME at Chrome or Edge:
 *
 *   PW_CHROME="C:/Program Files/Google/Chrome/Application/chrome.exe" node scripts/tape-check.mjs match.html --at=30:00
 */
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const [file, ...rest] = process.argv.slice(2);
const args = Object.fromEntries(rest.map((a) => [a.slice(2).split("=")[0], a.includes("=") ? a.split("=").slice(1).join("=") : true]));
const executablePath = process.env.PW_CHROME;
if (!file || !existsSync(file)) {
  console.error("Give the path of a page made with export --page --video.");
  process.exit(2);
}
if (!executablePath || !existsSync(executablePath)) {
  console.error("Set PW_CHROME to a browser that plays MP4 (Chrome or Edge).");
  process.exit(2);
}

const seconds = (clock) => {
  const parts = String(clock ?? "").trim().split(":").map(Number);
  return parts.length === 2 ? parts[0] * 60 + parts[1] : NaN;
};

const browser = await chromium.launch({ executablePath, headless: true, args: ["--autoplay-policy=no-user-gesture-required", "--mute-audio"] });
let failed = false;
const say = (ok, text) => {
  if (!ok) failed = true;
  console.log(`${ok ? "ok  " : "FAIL"}  ${text}`);
};
try {
  const page = await browser.newPage({ viewport: { width: 1672, height: 941 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const address = pathToFileURL(resolve(file)).href + (typeof args.at === "string" ? "#t=" + args.at : "");
  await page.goto(address, { waitUntil: "load" });
  await page.waitForSelector(".poss", { timeout: 120_000 });

  const video = page.locator("video.match__media");
  const present = await video.waitFor({ state: "attached", timeout: 30_000 }).then(() => true, () => false);
  say(present, "the page shows a video, not the drawn match");
  if (present) {
    await page.waitForFunction(() => document.querySelector("video.match__media").readyState >= 2, null, { timeout: 120_000 }).catch(() => {});
    const read = () =>
      page.evaluate(() => {
        const v = document.querySelector("video.match__media");
        return {
          ready: v.readyState,
          duration: v.duration,
          width: v.videoWidth,
          height: v.videoHeight,
          time: v.currentTime,
          paused: v.paused,
          error: v.error ? v.error.code : 0,
          clock: document.querySelector(".tp__now")?.textContent ?? "",
          total: document.querySelector(".tp__total")?.textContent ?? "",
        };
      });
    const before = await read();
    say(before.error === 0 && before.ready >= 2, `the video can be played (ready state ${before.ready}, error ${before.error}, ${before.width}x${before.height})`);
    const box = await page.evaluate(() => {
      const v = document.querySelector("video.match__media");
      const picture = v.getBoundingClientRect();
      const panel = (v.closest(".match") ?? v.parentElement).getBoundingClientRect();
      return { w: picture.width, h: picture.height, pw: panel.width, ph: panel.height, fit: getComputedStyle(v).objectFit };
    });
    say(
      box.w >= 320 && box.h >= 180 && box.w <= box.pw + 1 && box.h <= box.ph + 1,
      `the picture has room: ${Math.round(box.w)}x${Math.round(box.h)} px inside a panel of ${Math.round(box.pw)}x${Math.round(box.ph)} px (object-fit ${box.fit})`,
    );
    const total = seconds(before.total);
    say(Math.abs(before.duration - total) < 2, `the video is as long as the analysis: ${before.duration.toFixed(1)} s against ${before.total} (${total} s)`);
    say(Math.abs(before.time - seconds(before.clock)) < 1.5, `before playing, video and clock agree: ${before.time.toFixed(2)} s against ${before.clock}`);

    await page.locator(".tp__play").first().click();
    await page.waitForTimeout(5000);
    const during = await read();
    await page.locator(".tp__play").first().click();
    say(!during.paused && during.time - before.time > 2.5, `the video plays: it moved on ${(during.time - before.time).toFixed(2)} s in five seconds`);
    say(Math.abs(during.time - seconds(during.clock)) < 1.5, `while playing, video and clock agree: ${during.time.toFixed(2)} s against ${during.clock}`);

    // jump with the transport's own +5 s button, twice, and check the video followed
    const forward = page.getByRole("button", { name: /forward|5 ?s/i }).last();
    if (await forward.count()) {
      await forward.click();
      await forward.click();
      await page.waitForTimeout(800);
      const after = await read();
      say(Math.abs(after.time - seconds(after.clock)) < 1.5, `after a jump, video and clock agree: ${after.time.toFixed(2)} s against ${after.clock}`);
    }
  }
  say(errors.length === 0, errors.length ? `page errors: ${errors.join(" | ").slice(0, 300)}` : "no page errors");
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
