/**
 * Walks every tab of a page made with `python -m vantage export --page` and says, in counts
 * only, whether each one came up: panels, drawn canvases, list rows, insight cards, empty
 * states, error messages. It also follows one link from the possession list and one from an
 * insight card. Nothing the page says about the match is printed and no picture is taken, so it
 * can be run on a match that may not be shown around.
 *
 *   node scripts/page-check.mjs match.html [--refuses-video]
 *
 * With no file the built viewer and its simulated match are walked. --refuses-video also offers
 * the page a test pattern as a video and expects it to be turned away, which is what a page of
 * the competition match has to do. PW_CHROME names another browser than the Chromium that comes
 * with Playwright.
 */
import { existsSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const flags = new Set(process.argv.slice(2).filter((a) => a.startsWith("--")));
const file = positional[0] ?? join(root, "dist", "index.html");
if (!existsSync(file)) {
  console.error("Give the path of a page made with export --page.");
  process.exit(2);
}

function findChrome() {
  if (process.env.PW_CHROME && existsSync(process.env.PW_CHROME)) return process.env.PW_CHROME;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "ms-playwright");
  if (!existsSync(base)) return null;
  const newest = (prefix) =>
    readdirSync(base)
      .filter((d) => d.startsWith(prefix))
      .sort((a, b) => Number(b.split("-").pop()) - Number(a.split("-").pop()));
  for (const dir of newest("chromium-")) {
    for (const sub of ["chrome-win64/chrome.exe", "chrome-win/chrome.exe", "chrome-linux/chrome"]) {
      const exe = join(base, dir, sub);
      if (existsSync(exe)) return exe;
    }
  }
  return null;
}

const executablePath = findChrome();
if (!executablePath) {
  console.error("No browser found. Run `npx playwright install chromium`, or set PW_CHROME.");
  process.exit(2);
}

/** Counts only. A canvas is "drawn" when more than 1% of its sampled pixels differ from its first one. */
const census = () => {
  const shown = (el) => {
    const box = el.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  };
  const count = (selector) => [...document.querySelectorAll(selector)].filter(shown).length;
  const canvases = [...document.querySelectorAll("canvas")].filter(shown);
  let drawn = 0;
  for (const canvas of canvases) {
    try {
      const context = canvas.getContext("2d");
      if (!context || !canvas.width || !canvas.height) continue;
      const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const step = 4 * 7;
      let differ = 0;
      let seen = 0;
      for (let i = 0; i < data.length; i += step, seen++) {
        if (data[i] !== data[0] || data[i + 1] !== data[1] || data[i + 2] !== data[2] || data[i + 3] !== data[3]) differ++;
      }
      if (seen && differ / seen > 0.01) drawn++;
    } catch {
      // a canvas that cannot be read is counted as not drawn
    }
  }
  return {
    panels: count(".panel"),
    empty: count(".panel.empty, .pitch-empty, .match__empty"),
    canvases: canvases.length,
    drawn,
    rows: count(".rowlink"),
    cards: count(".card"),
    alerts: count(".notice[role=alert]"),
    clock: document.querySelector(".tp__now")?.textContent ?? "",
  };
};

const expected = {
  Possession: (c) => c.panels >= 2 && c.drawn >= 3 && c.empty === 0,
  Match: (c) => c.panels >= 3 && c.rows >= 3 && c.empty === 0,
  "Team shape": (c) => c.panels >= 3 && c.drawn >= 1 && c.empty === 0,
  Insights: (c) => c.cards >= 3 && c.empty === 0,
};

const browser = await chromium.launch({ executablePath, headless: true, args: ["--mute-audio", "--hide-scrollbars"] });
let failed = false;
const say = (ok, text) => {
  if (!ok) failed = true;
  console.log(`${ok ? "ok  " : "FAIL"}  ${text}`);
};
try {
  for (const size of [
    { name: "desktop", width: 1672, height: 941 },
    { name: "phone", width: 390, height: 844, mobile: true },
  ]) {
    const context = await browser.newContext({
      viewport: { width: size.width, height: size.height },
      deviceScaleFactor: size.mobile ? 2 : 1,
      isMobile: Boolean(size.mobile),
      hasTouch: Boolean(size.mobile),
      colorScheme: "dark",
    });
    const page = await context.newPage();
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(pathToFileURL(resolve(file)).href + "#t=20:00", { waitUntil: "load" });
    await page.waitForSelector(".poss", { timeout: 120_000 });
    await page.waitForFunction(() => !document.querySelector(".notice--busy"), null, { timeout: 120_000 });
    await page.waitForTimeout(1500);

    for (const tab of Object.keys(expected)) {
      await page.getByRole("tab", { name: tab, exact: true }).click();
      await page.waitForTimeout(900);
      const c = await page.evaluate(census);
      const wide = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
      say(
        expected[tab](c) && c.alerts === 0 && !wide,
        `[${size.name}] ${tab}: ${c.panels} panels, ${c.drawn} of ${c.canvases} canvases drawn, ${c.rows} list rows, ${c.cards} cards, ${c.empty} empty states, ${c.alerts} error messages${wide ? ", PAGE SCROLLS SIDEWAYS" : ""}`,
      );
    }

    // one possession from the list on the Match tab: it should open the Possession tab at that moment
    await page.getByRole("tab", { name: "Match", exact: true }).click();
    await page.waitForTimeout(500);
    const before = (await page.evaluate(census)).clock;
    const rows = page.locator(".rowlink");
    const picked = Math.min(4, (await rows.count()) - 1);
    if (picked >= 0) {
      await rows.nth(picked).click();
      await page.waitForTimeout(900);
      const onPossession = await page.getByRole("tab", { name: "Possession", exact: true }).getAttribute("aria-selected");
      const after = (await page.evaluate(census)).clock;
      say(onPossession === "true" && after !== before, `[${size.name}] a possession from the list opens at its own moment (the clock moved: ${after !== before})`);
    } else say(false, `[${size.name}] the Match tab lists no possessions`);

    // one insight card: its link should do the same
    await page.getByRole("tab", { name: "Insights", exact: true }).click();
    await page.waitForTimeout(500);
    const links = page.locator(".card__link");
    if (await links.count()) {
      const clockBefore = (await page.evaluate(census)).clock;
      await links.first().click();
      await page.waitForTimeout(900);
      const onPossession = await page.getByRole("tab", { name: "Possession", exact: true }).getAttribute("aria-selected");
      const clockAfter = (await page.evaluate(census)).clock;
      say(onPossession === "true", `[${size.name}] an insight's link opens the moment it is about (the clock moved: ${clockAfter !== clockBefore})`);
    } else say(false, `[${size.name}] no insight has a link`);

    if (flags.has("--refuses-video")) {
      // A match whose video is withheld (the competition match is one) has to turn a video away.
      // The video offered is a one-second test pattern recorded in this very browser.
      const bytes = await page.evaluate(async () => {
        const canvas = document.createElement("canvas");
        canvas.width = 160;
        canvas.height = 90;
        const g = canvas.getContext("2d");
        const recorder = new MediaRecorder(canvas.captureStream(25), { mimeType: "video/webm;codecs=vp8" });
        const chunks = [];
        recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
        const stopped = new Promise((done) => (recorder.onstop = done));
        recorder.start();
        const began = performance.now();
        await new Promise((finish) => {
          const draw = () => {
            const t = (performance.now() - began) / 1000;
            g.fillStyle = "#0f1113";
            g.fillRect(0, 0, 160, 90);
            g.fillStyle = "#c6ff00";
            g.fillRect(t * 140, 40, 10, 10);
            if (t < 1.1) requestAnimationFrame(draw);
            else finish();
          };
          draw();
        });
        recorder.stop();
        await stopped;
        return Array.from(new Uint8Array(await new Blob(chunks, { type: "video/webm" }).arrayBuffer()));
      });
      const pattern = join(mkdtempSync(join(tmpdir(), "vantage-check-")), "test-pattern.webm");
      writeFileSync(pattern, Buffer.from(bytes));
      await page.locator('input[type="file"][accept*="video"]').setInputFiles(pattern);
      await page.waitForTimeout(900);
      const state = await page.evaluate(() => ({ videos: document.querySelectorAll("video").length, alerts: document.querySelectorAll(".notice[role=alert]").length }));
      say(state.videos === 0 && state.alerts === 1, `[${size.name}] a video offered to this match is turned away (${state.videos} videos on the page, ${state.alerts} message saying why)`);
      await page.getByRole("button", { name: "Dismiss" }).click().catch(() => {});
    }

    say(errors.length === 0, `[${size.name}] ${errors.length ? "errors: " + errors.join(" | ").slice(0, 300) : "no errors in the page"}`);
    await context.close();
  }
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
