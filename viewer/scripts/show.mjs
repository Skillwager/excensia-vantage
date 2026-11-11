/**
 * Shows a page made with `python -m vantage export --page` as a window of its own: no tabs, no
 * address bar, maximised, and playing from the moment asked for.
 *
 *   node scripts/show.mjs ../runs/match/final/match.html --at=7:20
 *   node scripts/show.mjs page.html --video=/path/to/match.mp4
 *   node scripts/show.mjs --close
 *
 * --video hands the page a video file, the way its own "Add the match video" button does, and
 * starts the two together. Whether the page takes it is the page's decision: a run file of the
 * model's output for the competition match says that its video is withheld, and the viewer then
 * turns the video away (the organisers' rules do not allow output to be checked against the
 * video). A page of the organisers' public five minutes (scripts/public_truth_run.py) takes it.
 *
 * The window is a separate browser with its own empty profile (kept under runs/, which is not
 * in the repository), so no extension of your everyday browser can read the page. The script
 * then asks that window, in numbers only, whether the page is up and the clock is running, and
 * leaves it open. PW_CHROME names the browser; Chrome and Edge are looked for otherwise.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const args = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith("--"))
    .map((a) => [a.slice(2).split("=")[0], a.includes("=") ? a.split("=").slice(1).join("=") : true]),
);
const file = positional[0];
if (!args.close && (!file || !existsSync(file))) {
  console.error("Give the path of a page made with export --page, or --close to close the window.");
  process.exit(2);
}
const videoFile = typeof args.video === "string" ? resolve(args.video) : null;
if (videoFile && !existsSync(videoFile)) {
  console.error(`There is no video at ${videoFile}.`);
  process.exit(2);
}

const browsers = [
  process.env.PW_CHROME,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];
const executable = browsers.find((path) => path && existsSync(path));
if (!executable) {
  console.error("No Chrome or Edge found. Set PW_CHROME to a browser.");
  process.exit(2);
}

const profile = resolve(root, "..", "runs", "show-profile");
mkdirSync(profile, { recursive: true });
const portFile = join(profile, "DevToolsActivePort");
if (existsSync(portFile)) {
  // a window from an earlier call is still open: close it, so there is only ever one
  try {
    const earlier = await chromium.connectOverCDP(`http://127.0.0.1:${Number(readFileSync(portFile, "utf8").split("\n")[0])}`, { timeout: 3000 });
    const session = await earlier.newBrowserCDPSession();
    await session.send("Browser.close").catch(() => {});
    await new Promise((done) => setTimeout(done, 1500));
  } catch {
    // nothing was running
  }
  rmSync(portFile, { force: true });
}
if (args.close) {
  console.log("closed");
  process.exit(0);
}

const at = typeof args.at === "string" ? args.at : "0:00";
// with a video to hand over, the match waits for it; otherwise it starts by itself
const address = pathToFileURL(resolve(file)).href + (videoFile ? `#t=${at}` : `#t=${at}&play=1`);
const child = spawn(
  executable,
  [
    `--user-data-dir=${profile}`,
    `--app=${address}`,
    "--start-maximized",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    // Extensions registered for every profile on this computer (a PDF viewer, a VPN) would
    // otherwise install themselves into this window's empty profile and could read the page.
    "--disable-extensions",
    "--autoplay-policy=no-user-gesture-required",
    "--remote-debugging-port=0",
  ],
  { detached: true, stdio: "ignore" },
);
child.unref();

// the browser writes the port it chose into its profile
let port = null;
for (let i = 0; i < 100 && port === null; i++) {
  await new Promise((done) => setTimeout(done, 200));
  if (existsSync(portFile)) port = Number(readFileSync(portFile, "utf8").split("\n")[0]);
}
if (!port) {
  console.log("The window was started, but it did not answer. Look for it on the screen.");
  process.exit(1);
}

let failed = false;
const say = (ok, text) => {
  if (!ok) failed = true;
  console.log(`${ok ? "ok  " : "FAIL"}  ${text}`);
};
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
try {
  const context = browser.contexts()[0];
  let page = context.pages().find((p) => p.url().startsWith("file:"));
  for (let i = 0; i < 50 && !page; i++) {
    await new Promise((done) => setTimeout(done, 200));
    page = context.pages().find((p) => p.url().startsWith("file:"));
  }
  if (!page) throw new Error("the page did not open");
  await page.waitForSelector(".poss", { timeout: 120_000 });
  await page.waitForFunction(() => !document.querySelector(".notice--busy"), null, { timeout: 120_000 });

  let taken = false;
  if (videoFile) {
    // the file is named to the page's own video input; nothing is copied
    const session = await context.newCDPSession(page);
    const { root: documentNode } = await session.send("DOM.getDocument");
    const { nodeId } = await session.send("DOM.querySelector", { nodeId: documentNode.nodeId, selector: 'input[type="file"][accept*="video"]' });
    await session.send("DOM.setFileInputFiles", { files: [videoFile], nodeId });
    await page
      .waitForFunction(
        () => {
          const video = document.querySelector("video.match__media");
          return (video && video.readyState >= 2) || document.querySelector(".notice[role=alert]");
        },
        null,
        { timeout: 60_000 },
      )
      .catch(() => {});
    taken = (await page.locator("video.match__media").count()) === 1;
    if (taken) await page.locator(".tp__play").first().click();
  }

  const read = () =>
    page.evaluate(() => {
      const video = document.querySelector("video.match__media");
      return {
        clock: document.querySelector(".tp__now")?.textContent ?? "",
        total: document.querySelector(".tp__total")?.textContent ?? "",
        playing: document.querySelector(".tp__play")?.getAttribute("aria-label") === "Pause",
        alerts: document.querySelectorAll(".notice[role=alert]").length,
        videos: document.querySelectorAll("video").length,
        videoTime: video ? video.currentTime : null,
        videoPaused: video ? video.paused : null,
        visible: document.visibilityState === "visible",
        focused: document.hasFocus(),
        width: window.innerWidth,
        height: window.innerHeight,
        screenWidth: window.screen.availWidth,
        screenHeight: window.screen.availHeight,
      };
    });
  const seconds = (clock) => {
    const parts = String(clock).trim().split(":").map(Number);
    return parts.length === 2 ? parts[0] * 60 + parts[1] : NaN;
  };
  await page.waitForTimeout(1500);
  const first = await read();
  await page.waitForTimeout(4000);
  const second = await read();
  if (videoFile && !taken) {
    say(false, `the page did not take the video (${second.alerts} message on the page says why); it shows the match drawn from its tracking`);
  } else {
    say(second.alerts === 0, `the page is up, ${second.total} long, with ${second.alerts} error messages`);
    if (videoFile) {
      say(
        second.videos === 1 && second.videoPaused === false && second.videoTime - first.videoTime >= 3,
        `the video is in the page and playing: ${first.videoTime.toFixed(1)} s to ${second.videoTime.toFixed(1)} s of the video in four seconds`,
      );
    } else say(second.videos === 0, `no video is in the page (${second.videos})`);
    say(second.playing && seconds(second.clock) - seconds(first.clock) >= 3, `the clock is running: from ${first.clock} to ${second.clock} in four seconds`);
    if (videoFile) {
      // where the run has the video's camera, the evidence is drawn on the picture: say how many
      // of the small pictures are (the label on the team-shape panel says which it is showing,
      // and a picture cut from a football video is mostly grass)
      const evidence = await page.evaluate(() => {
        const grass = (canvas) => {
          try {
            const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
            let green = 0;
            let total = 0;
            for (let i = 0; i < data.length; i += 4 * 11, total++) if (data[i + 1] > data[i] + 10 && data[i + 1] > data[i + 2] + 10) green++;
            return total ? green / total : 0;
          } catch {
            return -1;
          }
        };
        return {
          onFootage: document.querySelector(".shape__canvas")?.getAttribute("aria-label")?.includes("match video") ?? false,
          tiles: [...document.querySelectorAll(".tile__canvas")].map(grass),
        };
      });
      if (evidence.onFootage) {
        say(evidence.tiles.length === 3 && evidence.tiles.every((share) => share > 0.25), `the ball close-up, team shape and runs are drawn on the video (grass share of each: ${evidence.tiles.map((share) => share.toFixed(2)).join(", ")})`);
      } else console.log("      this run has no camera for its video, so the evidence stays drawn on a pitch");
    }
  }
  say(second.visible, `the window is showing (${second.width}x${second.height} of a ${second.screenWidth}x${second.screenHeight} screen)${second.focused ? ", in front" : ", not in front yet"}`);
} catch (error) {
  say(false, `the window did not answer as expected: ${error.message}`);
}
// leave the window open: only this script's connection ends
process.exit(failed ? 1 : 0);
