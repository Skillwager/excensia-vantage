/**
 * Opens the built viewer (dist/index.html) in Chromium and saves screenshots of every tab at three sizes
 * into shots/. Also fails if the page logs an error or asks for anything other than the Google Fonts.
 *
 *   npm run build && npm run shots
 *
 * Options: --only=desktop|laptop|phone   --size=1024x768 (one more size)   --extras (zoom, tooltip, sheets, playing)
 *          --url=http://…
 * Uses a Chromium already installed by Playwright; set PW_CHROME to point at another browser.
 */
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "shots");
mkdirSync(out, { recursive: true });

const args = Object.fromEntries(process.argv.slice(2).map((a) => (a.startsWith("--") ? [a.slice(2).split("=")[0], a.includes("=") ? a.split("=").slice(1).join("=") : true] : [a, true])));
const url = typeof args.url === "string" ? args.url : pathToFileURL(join(root, "dist", "index.html")).href;

function findChrome() {
  if (process.env.PW_CHROME && existsSync(process.env.PW_CHROME)) return process.env.PW_CHROME;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "ms-playwright");
  if (!existsSync(base)) return null;
  const byNewest = (prefix) =>
    readdirSync(base)
      .filter((d) => d.startsWith(prefix))
      .sort((a, b) => Number(b.split("-").pop()) - Number(a.split("-").pop()));
  for (const dir of byNewest("chromium_headless_shell-")) {
    for (const sub of ["chrome-headless-shell-win64/chrome-headless-shell.exe", "chrome-win/headless_shell.exe", "chrome-linux/headless_shell"]) {
      const exe = join(base, dir, sub);
      if (existsSync(exe)) return exe;
    }
  }
  for (const dir of byNewest("chromium-")) {
    for (const sub of ["chrome-win64/chrome.exe", "chrome-win/chrome.exe", "chrome-linux/chrome"]) {
      const exe = join(base, dir, sub);
      if (existsSync(exe)) return exe;
    }
  }
  return null;
}

const VIEWPORTS = [
  { name: "desktop", width: 1672, height: 941, scale: 1 },
  { name: "laptop", width: 1280, height: 800, scale: 1 },
  { name: "phone", width: 390, height: 844, scale: 2, mobile: true },
].filter((v) => !args.only || args.only === v.name);
if (typeof args.size === "string" && /^\d+x\d+$/.test(args.size)) {
  const [width, height] = args.size.split("x").map(Number);
  VIEWPORTS.length = 0;
  VIEWPORTS.push({ name: args.size, width, height, scale: 1, mobile: width < 700 });
}

const TABS = [
  { id: "possession", label: "Possession" },
  { id: "match", label: "Match" },
  { id: "shape", label: "Team shape" },
  { id: "insights", label: "Insights" },
];

const executablePath = findChrome();
if (!executablePath) {
  console.error("No Chromium found. Run `npx playwright install chromium`, or set PW_CHROME to a Chrome or Edge executable.");
  process.exit(2);
}

const browser = await chromium.launch({ executablePath, headless: true, args: ["--disable-gpu", "--hide-scrollbars", "--force-color-profile=srgb"] });
const problems = [];
const requests = new Set();

try {
  for (const vp of VIEWPORTS) {
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: vp.scale,
      isMobile: Boolean(vp.mobile),
      hasTouch: Boolean(vp.mobile),
      colorScheme: "dark",
      reducedMotion: "reduce",
      acceptDownloads: true,
    });
    const page = await context.newPage();
    page.on("console", (message) => {
      if (message.type() === "error") problems.push(`[${vp.name}] console: ${message.text()}`);
    });
    page.on("pageerror", (error) => problems.push(`[${vp.name}] page error: ${error.message}`));
    page.on("request", (request) => {
      const u = request.url();
      if (!u.startsWith("data:") && !u.startsWith("blob:") && u !== url) requests.add(new URL(u).origin === "null" ? u : new URL(u).origin);
    });
    page.on("requestfailed", (request) => {
      if (!request.url().startsWith("blob:")) problems.push(`[${vp.name}] request failed: ${request.url()}`);
    });

    await page.goto(url, { waitUntil: "load" });
    await page.waitForSelector(".poss", { timeout: 30_000 });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(500);

    const overflow = async (label) => {
      const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      if (wide > 0) problems.push(`[${vp.name}] ${label}: the page scrolls sideways by ${wide}px`);
    };

    for (const tab of TABS) {
      await page.getByRole("tab", { name: tab.label, exact: true }).click();
      await page.waitForTimeout(350);
      await overflow(tab.id);
      await page.screenshot({ path: join(out, `${vp.name}-${tab.id}.png`) });
      if (vp.mobile) await page.screenshot({ path: join(out, `${vp.name}-${tab.id}-full.png`), fullPage: true });
      console.log(`saved ${vp.name}-${tab.id}`);
    }

    if (args.extras) {
      await page.getByRole("tab", { name: "Possession", exact: true }).click();
      await page.waitForTimeout(250);
      const canvas = page.locator(".tl__canvas");
      const box = await canvas.boundingBox();
      if (box && !vp.mobile) {
        // hover: the tooltip
        await page.mouse.move(box.x + box.width * 0.42, box.y + 30);
        await page.waitForTimeout(150);
        await page.screenshot({ path: join(out, `${vp.name}-possession-tooltip.png`) });
        // wheel in around the playhead: the zoomed timeline, its range and the way back
        await page.mouse.move(box.x + box.width * 0.76, box.y + 30);
        for (let i = 0; i < 7; i++) await page.mouse.wheel(0, -320);
        await page.waitForTimeout(300);
        await page.mouse.move(box.x + box.width * 0.6, box.y + 52);
        await page.waitForTimeout(150);
        await page.screenshot({ path: join(out, `${vp.name}-possession-zoomed.png`) });
        await page.getByRole("button", { name: "Full match" }).click();
        await page.mouse.move(10, 10);
      }
      // a second chip
      await page.getByRole("button", { name: /Team shape/ }).first().click();
      await page.waitForTimeout(150);
      await page.screenshot({ path: join(out, `${vp.name}-possession-shape-evidence.png`) });
      await page.getByRole("button", { name: /^Ball (seen|hidden)|^The ball/ }).first().click();
      // playing
      await page.getByRole("button", { name: "Play", exact: true }).click();
      await page.waitForTimeout(1600);
      await page.screenshot({ path: join(out, `${vp.name}-possession-playing.png`) });
      await page.getByRole("button", { name: "Pause", exact: true }).click();
      // export
      const download = page.waitForEvent("download", { timeout: 15_000 }).catch(() => null);
      await page.getByRole("button", { name: "Validate export" }).click();
      const file = await download;
      await page.waitForTimeout(300);
      await page.screenshot({ path: join(out, `${vp.name}-export.png`) });
      if (file) console.log(`  export downloaded as ${file.suggestedFilename()}`);
      else problems.push(`[${vp.name}] Validate export did not download a file`);
      await page.getByRole("button", { name: "Done" }).click();
      // settings
      await page.getByRole("button", { name: "Files and settings" }).click();
      await page.waitForTimeout(300);
      await page.screenshot({ path: join(out, `${vp.name}-settings.png`) });
      await page.keyboard.press("Escape");
      console.log(`saved ${vp.name} extras`);
    }
    await context.close();
  }
} finally {
  await browser.close();
}

const outside = [...requests].filter((origin) => !/^https:\/\/fonts\.(googleapis|gstatic)\.com$/.test(origin));
for (const origin of outside) problems.push(`request to ${origin} (only Google Fonts is allowed)`);
console.log(`requests: ${[...requests].join(", ") || "none"}`);
if (problems.length) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log("no console errors, no failed requests, no sideways scroll");
