/**
 * Does a page that names its video (python -m vantage tracking --video-url) open with the video
 * playing? Loads the page over http in a headless Chrome and prints numbers only: the video's
 * size, length and whether time moves once it plays. A picture is taken only with --shot=<png>,
 * which is for open footage alone; private footage is checked in numbers.
 *
 *   node scripts/page-video-check.mjs http://127.0.0.1:5321/index.html [--shot=open.png]
 */
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { chromium } = createRequire(join(here, "..", "package.json"))("playwright-core");
const [url] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const shot = (process.argv.find((a) => a.startsWith("--shot=")) || "").slice(7) || null;
const executable = [process.env.PW_CHROME, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"].find((p) => p && existsSync(p));

const browser = await chromium.launch({ executablePath: executable, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const failures = [];
  page.on("requestfailed", (request) => failures.push(request.url()));
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => {
    const video = document.querySelector("video");
    return video && video.readyState >= 2;
  }, null, { timeout: 30_000 }).catch(() => {});
  const before = await page.evaluate(() => {
    const video = document.querySelector("video");
    return video ? { ready: video.readyState, width: video.videoWidth, height: video.videoHeight, seconds: video.duration } : null;
  });
  // play from the page's own play control, as a person would (space bar), then see time move
  await page.keyboard.press("Space").catch(() => {});
  await page.waitForTimeout(2500);
  const after = await page.evaluate(() => {
    const video = document.querySelector("video");
    const text = document.body.innerText;
    return {
      time: video ? video.currentTime : null,
      paused: video ? video.paused : null,
      notice: (text.match(/(could not be loaded|can't be played|was not opened)[^\n]*/) || [null])[0],
    };
  });
  console.log(JSON.stringify({ video: before, after, failed: failures }, null, 1));
  if (shot) {
    await page.screenshot({ path: shot });
    console.log("picture: " + shot);
  }
} finally {
  await browser.close();
}
