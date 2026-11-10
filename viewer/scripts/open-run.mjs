/**
 * Opens a run file in the built viewer (dist/index.html) and says whether it loaded.
 *
 *   node scripts/open-run.mjs path/to/match.run.json            # check only: prints totals, no pictures
 *   node scripts/open-run.mjs path/to/match.run.json --shots=out-folder [--at=31:40]
 *
 * The check-only form shows nothing of the match: it prints what the page says about the file
 * (its title line and any refusal), which is how a run of a match that may not be published is
 * tested. `--shots` saves screenshots and is for open data.
 */
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const [file, ...rest] = process.argv.slice(2);
const args = Object.fromEntries(rest.map((a) => [a.slice(2).split("=")[0], a.includes("=") ? a.split("=").slice(1).join("=") : true]));
if (!file || !existsSync(file)) {
  console.error("Give the path of a run file.");
  process.exit(2);
}

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

const executablePath = findChrome();
if (!executablePath) {
  console.error("No Chromium found. Run `npx playwright install chromium`, or set PW_CHROME.");
  process.exit(2);
}

const sizes = [
  { name: "desktop", width: 1672, height: 941, scale: 1 },
  { name: "phone", width: 390, height: 844, scale: 2, mobile: true },
];
const browser = await chromium.launch({ executablePath, headless: true, args: ["--disable-gpu", "--hide-scrollbars", "--force-color-profile=srgb"] });
let failed = false;
try {
  for (const size of args.shots ? sizes : sizes.slice(0, 1)) {
    const context = await browser.newContext({
      viewport: { width: size.width, height: size.height },
      deviceScaleFactor: size.scale,
      isMobile: Boolean(size.mobile),
      hasTouch: Boolean(size.mobile),
      colorScheme: "dark",
      reducedMotion: "reduce",
    });
    const page = await context.newPage();
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(pathToFileURL(join(root, "dist", "index.html")).href, { waitUntil: "load" });
    await page.waitForSelector(".poss", { timeout: 30_000 });
    await page.locator('input[type="file"][accept*="json"]').setInputFiles(resolve(file));
    // the page either shows the file or says why not
    await page.waitForFunction(() => !document.querySelector(".notice--busy"), null, { timeout: 120_000 });
    await page.waitForTimeout(800);
    const notice = await page.locator(".notice[role=alert]").first().textContent({ timeout: 500 }).catch(() => null);
    const footer = await page.locator(".poss").evaluate((node) => {
      const source = node.querySelector("[class*=source], [class*=foot]");
      return source ? source.textContent : null;
    });
    if (notice) {
      failed = true;
      console.log(`[${size.name}] refused: ${notice.trim()}`);
    } else {
      console.log(`[${size.name}] opened${footer ? `: ${footer.trim().slice(0, 120)}` : ""}`);
    }
    if (errors.length) {
      failed = true;
      console.log(`[${size.name}] errors: ${errors.join(" | ").slice(0, 400)}`);
    }
    if (args.shots && !notice) {
      const out = resolve(String(args.shots));
      mkdirSync(out, { recursive: true });
      if (typeof args.at === "string") {
        const [m, s] = args.at.split(":").map(Number);
        const fraction = await page.evaluate(() => 1);
        void fraction;
        // seek by keyboard-free means: click the transport bar at the right share of the match
        const clock = await page.locator(".tp__time, [class*=clock]").first().textContent().catch(() => null);
        const total = clock && /\/\s*(\d+):(\d+)/.exec(clock);
        const bar = await page.locator('input[type="range"]').first().boundingBox().catch(() => null);
        if (total && bar) {
          const share = (m * 60 + s) / (Number(total[1]) * 60 + Number(total[2]));
          await page.mouse.click(bar.x + bar.width * Math.min(0.999, Math.max(0, share)), bar.y + bar.height / 2);
          await page.waitForTimeout(400);
        }
      }
      for (const tab of ["Possession", "Match", "Team shape", "Insights"]) {
        await page.getByRole("tab", { name: tab, exact: true }).click();
        await page.waitForTimeout(450);
        await page.screenshot({ path: join(out, `${size.name}-${tab.toLowerCase().replace(" ", "-")}.png`) });
      }
      console.log(`[${size.name}] screenshots in ${out}`);
    }
    await context.close();
  }
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
