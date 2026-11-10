/**
 * Saves the match view at a list of moments of the simulated match, so the simulation can be looked at:
 * kick-offs, throw-ins, goal kicks, corners, free kicks, goals, half-time.
 *
 *   npm run build && node scripts/moments.mjs
 *
 * Writes shots/moments/<time>.jpg. Simulated data only.
 */
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "shots", "moments");
mkdirSync(out, { recursive: true });
const url = pathToFileURL(join(root, "dist", "index.html")).href;

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

// times given on the command line (node scripts/moments.mjs 12:05 60:30), or this tour of the simulated match
const MOMENTS = process.argv.slice(2).filter((a) => /^\d+:\d\d$/.test(a));
const LIST = MOMENTS.length
  ? MOMENTS.map((time) => [time, ""])
  : [
      ["0:02", "kick-off"],
      ["0:14", "throw-in"],
      ["0:45", "loose-ball"],
      ["9:00", "goal-kick"],
      ["10:01", "free-kick"],
      ["11:43", "corner"],
      ["31:40", "goal"],
      ["32:43", "kick-off-after-goal"],
      ["47:58", "second-half-kick-off"],
      ["60:30", "open-play"],
    ];

const browser = await chromium.launch({ executablePath: findChrome(), headless: true, args: ["--disable-gpu", "--hide-scrollbars", "--force-color-profile=srgb"] });
try {
  const context = await browser.newContext({ viewport: { width: 1672, height: 941 }, deviceScaleFactor: 1, colorScheme: "dark", reducedMotion: "reduce" });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const [moment, what] of LIST) {
    await page.goto(`${url}#t=${moment}`, { waitUntil: "load" });
    // the address only changes after the hash, so make sure the page starts afresh
    await page.reload({ waitUntil: "load" });
    await page.waitForSelector(".poss", { timeout: 30_000 });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(350);
    const name = `${moment.replace(":", "m")}${what ? `-${what}` : ""}`;
    await page.locator(".match").screenshot({ path: join(out, `${name}.jpg`), type: "jpeg", quality: 86 });
    console.log(`saved ${name}`);
  }
  if (errors.length) console.error(errors.join("\n"));
} finally {
  await browser.close();
}
