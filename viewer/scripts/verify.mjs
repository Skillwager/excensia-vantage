/**
 * Drives the built viewer (dist/index.html) in Chromium the way a person would and checks that it behaves:
 * the keyboard, scrubbing, zoom, replay, the export's bytes, and opening run files and a video.
 *
 *   npm run build && node scripts/verify.mjs
 *
 * Every file it opens is made here: the simulated run saved by the viewer itself, small hand-written runs,
 * and a three-second test pattern recorded from a canvas. No real match data is involved.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const url = pathToFileURL(join(root, "dist", "index.html")).href;
const work = join(tmpdir(), `vantage-verify-${process.pid}`);
mkdirSync(work, { recursive: true });

function findChrome(full) {
  if (process.env.PW_CHROME && existsSync(process.env.PW_CHROME)) return process.env.PW_CHROME;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || join(process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local"), "ms-playwright");
  if (!existsSync(base)) return null;
  for (const prefix of full ? ["chromium-"] : ["chromium_headless_shell-", "chromium-"]) {
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

const states = process.argv.includes("--shots") ? join(root, "shots", "states") : null;
if (states) mkdirSync(states, { recursive: true });
const snap = async (page, name) => {
  if (!states) return;
  await page.waitForTimeout(250);
  await page.screenshot({ path: join(states, `${name}.png`) });
};

let passed = 0;
const failures = [];
function check(name, ok, detail = "") {
  if (ok) {
    passed++;
    console.log(`  ok    ${name}${detail ? `  (${detail})` : ""}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${detail ? `  (${detail})` : ""}`);
  }
}

const browser = await chromium.launch({
  executablePath: findChrome(false),
  headless: true,
  args: ["--disable-gpu", "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required"],
});
const consoleErrors = [];

try {
  const context = await browser.newContext({ viewport: { width: 1672, height: 941 }, colorScheme: "dark", acceptDownloads: true });
  const page = await context.newPage();
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(`page error: ${e.message}`));

  const open = async () => {
    await page.goto(url, { waitUntil: "load" });
    await page.waitForSelector(".poss", { timeout: 30_000 });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(250);
  };
  const clock = () => page.locator(".tlp .tp__now").innerText();
  const total = () => page.locator(".tlp .tp__total").innerText();
  const frame = async () => Number((await page.locator(".insp__frame").innerText()).replace(/\D/g, ""));
  const estimate = () => page.locator(".insp__estimate-value").innerText();
  const blur = () => page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());

  console.log("\nThe simulated match");
  const started = Date.now();
  await page.goto(url, { waitUntil: "commit" });
  await page.waitForSelector(".hdr");
  const shell = Date.now() - started;
  await page.waitForSelector(".poss", { timeout: 30_000 });
  const ready = Date.now() - started;
  check("shows its header at once and the simulated match soon after", ready < 8000, `header after ${shell} ms, match ready after ${ready} ms`);
  await open();
  check("opens on 72:14 of 95:00", (await clock()) === "72:14" && (await total()) === "95:00", `${await clock()} / ${await total()}`);
  check("the title is Excensia Vantage", (await page.title()) === "Excensia Vantage");
  check("starts at frame 108,350 with Home in possession", (await frame()) === 108350 && (await estimate()).toUpperCase() === "HOME");
  check("shows one Simulated match tag", (await page.getByText("Simulated match", { exact: true }).count()) === 1);
  const background = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  check("has a dark page background", background === "rgb(15, 17, 19)", background);

  console.log("\nKeyboard");
  await blur();
  await page.keyboard.press("ArrowRight");
  check("right arrow steps one frame", (await frame()) === 108351, String(await frame()));
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  check("left arrow steps back one frame", (await frame()) === 108349, String(await frame()));
  await page.keyboard.press("Shift+ArrowRight");
  check("shift + right arrow steps one second", (await frame()) === 108374, String(await frame()));
  await page.keyboard.press("Shift+ArrowLeft");
  await page.keyboard.press("ArrowRight");
  check("shift + left arrow steps back one second", (await frame()) === 108350, String(await frame()));
  await page.keyboard.press("]");
  const nextFrame = await frame();
  check("] jumps to the next change", nextFrame === 108825 && (await clock()) === "72:33", `${nextFrame} ${await clock()}`);
  await page.keyboard.press("[");
  const prevFrame = await frame();
  check("[ jumps to the previous change", prevFrame < 108350 && prevFrame > 107000, String(prevFrame));
  await page.keyboard.press("]");
  await page.keyboard.press("]");
  check("] twice goes on to the change after", (await frame()) > 108825, String(await frame()));
  await page.keyboard.press("[");
  await page.keyboard.press("[");
  await page.keyboard.press("Space");
  await page.waitForTimeout(1300);
  const playing = (await page.getByRole("button", { name: "Pause", exact: true }).count()) === 1;
  await page.keyboard.press("Space");
  const after = await frame();
  check("space plays and pauses", playing && after > prevFrame + 15 && after < prevFrame + 60 && (await page.getByRole("button", { name: "Play", exact: true }).count()) === 1, `${after - prevFrame} frames in 1.3 s`);

  console.log("\nTransport and timeline");
  await page.getByRole("button", { name: "Back 5 seconds" }).click();
  check("−5s goes back five seconds", (await frame()) === after - 125, String((await frame()) - after));
  await page.getByRole("button", { name: "Forward 5 seconds" }).click();
  await page.getByRole("button", { name: "Forward 5 seconds" }).click();
  check("+5s goes forward five seconds", (await frame()) === after + 125);
  await page.getByRole("button", { name: /^Speed/ }).click();
  await page.getByRole("option", { name: "2x" }).click();
  await blur();
  const before2 = await frame();
  await page.keyboard.press("Space");
  await page.waitForTimeout(1000);
  await page.keyboard.press("Space");
  const gained = (await frame()) - before2;
  check("2x plays about twice as fast", gained > 38 && gained < 85, `${gained} frames in 1 s`);
  await page.getByRole("button", { name: /^Speed/ }).click();
  await page.getByRole("option", { name: "1x" }).click();

  const canvas = page.locator(".tl__canvas");
  const box = await canvas.boundingBox();
  await page.mouse.click(box.x + box.width * 0.25, box.y + 30);
  const quarter = (await frame()) / 142500;
  check("a click on the timeline goes to that moment", Math.abs(quarter - 0.25) < 0.004, `${(quarter * 100).toFixed(2)}% of the match`);
  await page.mouse.move(box.x + box.width * 0.25, box.y + 30);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + 40, { steps: 6 });
  await page.mouse.up();
  const half = (await frame()) / 142500;
  check("dragging scrubs", Math.abs(half - 0.5) < 0.004, `${(half * 100).toFixed(2)}%`);
  await page.mouse.move(box.x + box.width * 0.6, box.y + 50);
  await page.waitForTimeout(120);
  const tip = await page.locator(".tip").innerText();
  check("hovering shows a tooltip with the time and the call", /\d\d:\d\d/.test(tip) && /(Home|Away|Dead ball|No answer)/.test(tip), tip.replace(/\n/g, " | "));
  await page.mouse.move(box.x + box.width * 0.5, box.y + 40);
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -300);
  await page.waitForTimeout(200);
  const back = page.getByRole("button", { name: "Full match" });
  check("the wheel zooms in and offers a way back", (await back.count()) === 1 && (await page.locator(".scrub__window").count()) === 1, await page.locator(".tlh__range").innerText());
  const zoomedFrame = await frame();
  check("zooming does not move the playhead", zoomedFrame === Math.round(half * 142500) || Math.abs(zoomedFrame / 142500 - half) < 0.0005);
  await page.mouse.click(box.x + box.width * 0.9, box.y + 30);
  const inWindow = Math.abs((await frame()) / 142500 - 0.5) < 0.06;
  check("a click on the zoomed timeline lands inside the zoomed stretch", inWindow, `${(((await frame()) / 142500) * 100).toFixed(2)}%`);
  // zoomed right in and playing fast, the timeline keeps up with the playhead
  for (let i = 0; i < 8; i++) await page.mouse.wheel(0, -300);
  await page.waitForTimeout(150);
  const rangeBefore = await page.locator(".tlh__range").innerText();
  await page.getByRole("button", { name: /^Speed/ }).click();
  await page.getByRole("option", { name: "4x" }).click();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForTimeout(4500);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const rangeAfter = await page.locator(".tlh__range").innerText();
  const inView = await page.evaluate(() => {
    const seconds = (text) => text.split(":").reduce((m, s) => m * 60 + Number(s), 0);
    const [from, to] = document.querySelector(".tlh__range").textContent.split("–").map((t) => seconds(t.trim()));
    const now = seconds(document.querySelector(".tlp .tp__now").textContent);
    return now >= from - 1 && now <= to + 1;
  });
  check("a zoomed timeline follows the playhead while playing", rangeAfter !== rangeBefore && inView, `${rangeBefore} → ${rangeAfter}`);
  await page.getByRole("button", { name: /^Speed/ }).click();
  await page.getByRole("option", { name: "1x" }).click();
  await back.click();
  check("Full match shows the whole match again", (await page.getByRole("button", { name: "Full match" }).count()) === 0);
  const scrub = await page.locator(".tlp .scrub__track").boundingBox();
  await page.mouse.click(scrub.x + scrub.width * 0.75, scrub.y + scrub.height / 2);
  check("a click on the scrubber goes there", Math.abs((await frame()) / 142499 - 0.75) < 0.004, await clock());

  console.log("\nReplay transition");
  await page.goto(`${url}#t=72:14`);
  await page.reload();
  await page.waitForSelector(".poss");
  await page.getByRole("button", { name: "Replay transition" }).click();
  await page.waitForTimeout(400);
  const replayStart = await frame();
  // 72:14 is nearer the change at 72:01 (frame 108,025) than the one at 72:33; three seconds before it is frame 107,950
  check("jumps to three seconds before the nearest change and plays", replayStart >= 107950 && replayStart < 107990, String(replayStart));
  await page.waitForTimeout(6600);
  const replayEnd = await frame();
  check("plays through the change and stops three seconds after it", replayEnd === 108100 && (await page.getByRole("button", { name: "Play", exact: true }).count()) === 1, String(replayEnd));

  console.log("\nValidate export");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Validate export" }).click();
  const file = await download;
  const csvPath = join(work, file.suggestedFilename());
  await file.saveAs(csvPath);
  const csv = readFileSync(csvPath);
  const text = csv.toString("latin1");
  const lines = text.split("\n");
  check("downloads home-vs-away-possession.csv", file.suggestedFilename() === "home-vs-away-possession.csv");
  check("first line is exactly frame,label", lines[0] === "frame,label" && csv[0] === 0x66);
  check("one row per frame from 0", lines.length === 142500 + 2 && lines[1] === "0,X" && lines[142500] === "142499," + lines[142500].split(",")[1] && lines[142501] === "", `${lines.length - 2} rows`);
  check("every row is frame,H|A|D|X in order", lines.slice(1, -1).every((line, i) => { const c = line.indexOf(","); return line.slice(0, c) === String(i) && "HADX".includes(line.slice(c + 1)) && line.length === c + 2; }));
  check("line ends are \\n only, with one at the end", !csv.includes(0x0d) && csv[csv.length - 1] === 0x0a);
  check("the sheet reports all three checks passed", (await page.locator(".checks__item").count()) === 3 && (await page.locator(".checks__item--bad").count()) === 0);
  const shares = await page.locator(".counts dd").allInnerTexts();
  check("the shares in the sheet add up to 100%", shares.reduce((s, v) => s + parseInt(v, 10), 0) === 100, shares.join(" "));
  await page.getByRole("button", { name: "Done" }).click();

  console.log("\nOpening your own files");
  await page.getByRole("button", { name: "Files and settings" }).click();
  const runDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save the simulated match as a run file" }).click();
  const runFile = await runDownload;
  const runPath = join(work, "saved-simulated.json");
  await runFile.saveAs(runPath);
  const saved = JSON.parse(readFileSync(runPath, "utf8"));
  check("the simulated match saves as a run file in the contract's shape", saved.schema === "excensia.vantage.run/1" && saved.match.frames === 142500 && saved.tracking.samples.length === 28500 && Array.isArray(saved.labels[0]), `${(readFileSync(runPath).length / 1e6).toFixed(1)} MB`);
  check("the settings sheet says files stay in the browser", (await page.getByText("Files are read in this browser only; nothing is uploaded.").count()) === 1);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  const backOn = await page.evaluate(() => document.activeElement?.getAttribute("aria-label"));
  check("Escape closes the sheet and focus goes back to the button that opened it", (await page.locator("dialog").count()) === 0 && backOn === "Files and settings", String(backOn));
  await page.keyboard.press("Space");
  await page.waitForTimeout(200);
  const reopened = (await page.locator("dialog").count()) === 1;
  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  const stillPaused = (await page.getByRole("button", { name: "Play", exact: true }).count()) === 1;
  check("space on a button reached by keyboard presses that button and does not start play", reopened && stillPaused);
  const runInput = page.locator('input[type="file"][accept*="json"]');
  const videoInput = page.locator('input[type="file"][accept*="video"]');
  await runInput.setInputFiles(runPath);
  await page.waitForFunction(() => document.querySelector(".source__text")?.textContent?.includes("saved-simulated.json"), null, { timeout: 30_000 });
  check("opens that run file back", (await clock()) === "00:00" && (await total()) === "95:00" && (await page.getByText("Simulated match", { exact: true }).count()) === 0);
  check("draws the match from its tracking, and says so", (await page.locator("canvas.match__media").count()) === 1 && (await page.getByText("Drawn from tracking").count()) === 1);
  await blur();
  await page.keyboard.press("]");
  check("the opened run has its changes and reasons", (await frame()) === 120 && (await estimate()).toUpperCase() === "HOME");

  // a run with labels only: no spells, no transitions, no tracking
  const plain = { schema: "excensia.vantage.run/1", match: { title: "Reds v Blues", home: { name: "Reds" }, away: { name: "Blues" }, fps: 30, frames: 5400 }, labels: [["X", 90], ["H", 1500], ["D", 600], ["A", 2400], ["H", 810]] };
  const plainPath = join(work, "labels-only.json");
  writeFileSync(plainPath, JSON.stringify(plain));
  await runInput.setInputFiles(plainPath);
  await page.waitForFunction(() => document.querySelector(".tlp .tp__total")?.textContent === "03:00");
  check("opens a run with labels only", (await total()) === "03:00" && (await page.locator(".tl__label--home").innerText()).toUpperCase() === "REDS");
  check("says honestly that there is no tracking", (await page.getByText("No tracking in this run").count()) === 1 && (await page.getByText("No picture for this match yet").count()) === 1);
  await snap(page, "labels-only-run");
  await blur();
  await page.keyboard.press("]");
  await page.keyboard.press("]");
  check("derives the changes from the labels", (await frame()) === 1590 && (await estimate()).toUpperCase() === "DEAD BALL", `${await frame()} ${await estimate()}`);
  check("says there is no confidence in this run", (await page.locator(".conf__value").innerText()) === "–");
  await page.getByRole("tab", { name: "Team shape" }).click();
  check("Team shape shows an empty state without tracking", (await page.getByText("No tracking in this run").count()) === 1);
  await page.getByRole("tab", { name: "Insights" }).click();
  check("Insights still finds things without tracking", (await page.locator(".card").count()) >= 3, `${await page.locator(".card").count()} findings`);
  await page.getByRole("tab", { name: "Match" }).click();
  check("Match summary works on a labels-only run", (await page.locator(".share__figure--home").innerText()) === "49%" && (await page.locator(".share__figure--away").innerText()) === "51%");
  await page.getByRole("tab", { name: "Possession" }).click();
  const csvDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Validate export" }).click();
  const small = await csvDownload;
  const smallPath = join(work, small.suggestedFilename());
  await small.saveAs(smallPath);
  const smallText = readFileSync(smallPath, "utf8");
  check("exports the opened run byte for byte", small.suggestedFilename() === "reds-v-blues-possession.csv" && smallText.startsWith("frame,label\n0,X\n") && smallText.endsWith("\n5399,H\n") && smallText.split("\n").length === 5402);
  await page.getByRole("button", { name: "Done" }).click();

  // files that must be refused, each with words a person can act on
  const refuse = async (name, content, pattern) => {
    const path = join(work, name);
    writeFileSync(path, typeof content === "string" ? content : JSON.stringify(content));
    await runInput.setInputFiles(path);
    await page.waitForFunction((n) => document.querySelector('.notice[role="alert"]')?.textContent?.includes(n), name, { timeout: 10_000 });
    const notice = await page.locator('.notice[role="alert"]').innerText();
    if (name === "short-labels.json") await snap(page, "refused-file");
    check(`refuses ${name}`, pattern.test(notice) && (await total()) === "03:00", notice.replace(/\s+/g, " "));
    await page.getByRole("button", { name: "Dismiss" }).click();
  };
  await refuse("short-labels.json", { ...plain, labels: [["H", 100]] }, /The labels cover 100 frames but the match has 5,400/);
  await refuse("wrong-label.json", { ...plain, labels: [["H", 5000], ["Q", 400]] }, /Label run 2 uses "Q"/);
  await refuse("not-a-run.json", { hello: "world" }, /no "schema" field/);
  await refuse("broken.json", "{ this is not json", /not JSON/);
  await refuse("newer.json", { ...plain, schema: "excensia.vantage.run/2" }, /reads "excensia\.vantage\.run\/1"/);

  // a video: a three-second test pattern recorded from a canvas in this very browser
  const videoBytes = await page.evaluate(async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 180;
    const g = canvas.getContext("2d");
    const stream = canvas.captureStream(25);
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8" });
    const chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise((r) => (recorder.onstop = r));
    recorder.start();
    const t0 = performance.now();
    await new Promise((finish) => {
      const draw = () => {
        const t = (performance.now() - t0) / 1000;
        g.fillStyle = "#0f1113";
        g.fillRect(0, 0, 320, 180);
        g.fillStyle = "#c6ff00";
        g.fillRect((t / 3) * 300, 80, 20, 20);
        g.fillStyle = "#f5f5f2";
        g.font = "20px sans-serif";
        g.fillText(`test pattern ${t.toFixed(1)}`, 20, 40);
        if (t < 3.2) requestAnimationFrame(draw);
        else finish();
      };
      draw();
    });
    recorder.stop();
    await done;
    const buffer = await new Blob(chunks, { type: "video/webm" }).arrayBuffer();
    return Array.from(new Uint8Array(buffer));
  });
  const videoPath = join(work, "test-pattern.webm");
  writeFileSync(videoPath, Buffer.from(videoBytes));
  await videoInput.setInputFiles(videoPath);
  await page.waitForSelector("video.match__media", { timeout: 20_000 });
  check("adds a video beside an opened run and keeps the run", (await total()) === "03:00" && (await page.locator(".notice").count()) === 0);
  await snap(page, "run-with-video");
  await blur();
  await page.keyboard.press("Home");
  await page.locator(".tlp .scrub").focus();
  await page.keyboard.press("Home");
  await blur();
  await page.keyboard.press("Space");
  await page.waitForTimeout(1500);
  await page.keyboard.press("Space");
  const videoTime = await page.locator("video.match__media").evaluate((v) => v.currentTime);
  const runTime = (await frame()) / 30;
  check("the video plays and the clock follows it", videoTime > 0.6 && Math.abs(videoTime - runTime) < 0.12, `video ${videoTime.toFixed(2)} s, clock ${runTime.toFixed(2)} s`);
  await page.keyboard.press("ArrowRight");
  const stepped = await page.locator("video.match__media").evaluate((v) => v.currentTime);
  check("stepping a frame moves the video with it", Math.abs(stepped - ((await frame()) + 0.5) / 30) < 0.02, `${stepped.toFixed(3)} s at frame ${await frame()}`);

  // back to the simulated match, then a video with no run file
  await page.getByRole("button", { name: "Files and settings" }).click();
  await page.getByRole("button", { name: "Back to the simulated match" }).click();
  await page.waitForFunction(() => document.querySelector(".tlp .tp__total")?.textContent === "95:00");
  check("goes back to the simulated match", (await clock()) === "72:14" && (await page.locator("canvas.match__media").count()) === 1);
  await videoInput.setInputFiles(videoPath);
  await page.waitForSelector("video.match__media", { timeout: 20_000 });
  await page.waitForTimeout(200);
  const blankTotal = await total();
  check("a video on its own gets an empty match and asks for its run file", /^00:0[2-4]$/.test(blankTotal) && (await estimate()).toUpperCase() === "NO ANSWER" && /no run file yet/.test(await page.locator(".notice").innerText()), `${blankTotal}; ${await estimate()}`);
  await snap(page, "video-without-run");
  await runInput.setInputFiles(plainPath);
  await page.waitForFunction(() => document.querySelector(".tlp .tp__total")?.textContent === "03:00");
  check("opening the run file then keeps the video", (await page.locator("video.match__media").count()) === 1 && (await page.locator(".notice").count()) === 0);

  // a match whose video is withheld: the video on screen is closed, and none can be added
  const why = "The rules of this match do not allow its video beside the labels.";
  const withheldPath = join(work, "video-withheld.json");
  writeFileSync(withheldPath, JSON.stringify({ ...plain, video: { offset_frames: 0, withheld: why } }));
  await runInput.setInputFiles(withheldPath);
  await page.waitForFunction(() => document.querySelector(".source__text")?.textContent?.includes("video-withheld.json"), null, { timeout: 30_000 });
  await page.waitForTimeout(200);
  const closedNotice = await page.locator('.notice[role="alert"]').innerText().catch(() => "");
  check(
    "a run whose video is withheld closes the video that was open, and says why",
    (await page.locator("video.match__media").count()) === 0 && closedNotice.includes("test-pattern.webm was closed") && closedNotice.includes(why),
    closedNotice.replace(/\s+/g, " "),
  );
  await videoInput.setInputFiles(videoPath);
  await page.waitForTimeout(600);
  const refusedNotice = await page.locator('.notice[role="alert"]').innerText().catch(() => "");
  check(
    "and refuses a video for that match",
    (await page.locator("video.match__media").count()) === 0 && refusedNotice.includes("test-pattern.webm was not opened") && refusedNotice.includes(why),
    refusedNotice.replace(/\s+/g, " "),
  );
  await snap(page, "video-withheld");
  await page.getByRole("button", { name: "Files and settings" }).click();
  check("and offers no way to add one", (await page.getByRole("button", { name: "Add the match video" }).count()) === 0 && (await page.getByText(why).count()) >= 1);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  await page.getByRole("button", { name: "Dismiss" }).click();

  // dropping a file on the page
  await page.getByRole("button", { name: "Files and settings" }).click();
  await page.getByRole("button", { name: "Back to the simulated match" }).click();
  await page.waitForFunction(() => document.querySelector(".tlp .tp__total")?.textContent === "95:00");
  const dropped = await page.evaluateHandle((json) => {
    const data = new DataTransfer();
    data.items.add(new File([json], "dropped.json", { type: "application/json" }));
    return data;
  }, JSON.stringify(plain));
  await page.dispatchEvent("body", "dragenter", { dataTransfer: dropped });
  const overlay = await page.locator(".drop").count();
  await snap(page, "drop-target");
  await page.dispatchEvent("body", "drop", { dataTransfer: dropped });
  await page.waitForFunction(() => document.querySelector(".tlp .tp__total")?.textContent === "03:00");
  check("a run file dropped on the page opens", overlay === 1 && (await page.locator(".drop").count()) === 0 && (await page.locator(".source__text").innerText()).toLowerCase().includes("dropped.json"));

  console.log("\nSmoothness");
  // time the page's own work on each animation frame (a headless browser paints at its own, slower, rate)
  await page.addInitScript(() => {
    const raf = window.requestAnimationFrame.bind(window);
    window.__work = [];
    window.requestAnimationFrame = (callback) =>
      raf((time) => {
        const before = performance.now();
        callback(time);
        window.__work.push(performance.now() - before);
      });
  });
  await open();
  await page.evaluate(() => (window.__work.length = 0));
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForTimeout(3000);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const timing = await page.evaluate(() => {
    const work = window.__work.slice().sort((a, b) => a - b);
    return { frames: work.length, median: work[Math.floor(work.length / 2)], p95: work[Math.floor(work.length * 0.95)], worst: work[work.length - 1] };
  });
  check(
    "draws each frame well inside a 60 fps budget with the whole match loaded",
    timing.frames > 60 && timing.median < 8 && timing.p95 < 14,
    `${timing.frames} frames painted in 3 s; the page's work per frame: median ${timing.median.toFixed(1)} ms, 95th ${timing.p95.toFixed(1)} ms, worst ${timing.worst.toFixed(1)} ms (budget 16.7 ms)`,
  );
  const idle = await page.evaluate(
    () =>
      new Promise((done) => {
        const original = CanvasRenderingContext2D.prototype.clearRect;
        let paints = 0;
        CanvasRenderingContext2D.prototype.clearRect = function (...a) {
          paints++;
          return original.apply(this, a);
        };
        setTimeout(() => {
          CanvasRenderingContext2D.prototype.clearRect = original;
          done(paints);
        }, 800);
      }),
  );
  check("does no drawing while paused", idle === 0, `${idle} repaints in 0.8 s`);

  await context.close();

  console.log("\nPhone");
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: "dark" });
  const mobile = await phone.newPage();
  mobile.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  mobile.on("pageerror", (e) => consoleErrors.push(`page error: ${e.message}`));
  await mobile.goto(url, { waitUntil: "load" });
  await mobile.waitForSelector(".poss");
  await mobile.evaluate(() => document.fonts.ready);
  const order = await mobile.evaluate(() => {
    const top = (sel) => document.querySelector(sel).getBoundingClientRect().top;
    return [top(".match"), top(".tlp__transport"), top(".tlp__rows"), top(".insp"), top(".shape"), top(".actions")];
  });
  check("stacks video, transport, timeline, inspector", order.every((v, i) => i === 0 || v > order[i - 1]), order.map((v) => Math.round(v)).join(" < "));
  const gutters = await mobile.evaluate(() => {
    const r = document.querySelector(".match").getBoundingClientRect();
    return [Math.round(r.left), Math.round(window.innerWidth - r.right), document.documentElement.scrollWidth - document.documentElement.clientWidth];
  });
  check("keeps a 16 px gutter and never scrolls sideways", gutters[0] === 16 && gutters[1] === 16 && gutters[2] === 0, gutters.join(", "));
  const tabs = await mobile.evaluate(() => {
    const el = document.querySelector(".tabs");
    return { overflowX: getComputedStyle(el).overflowX, fits: el.scrollWidth <= el.clientWidth + 1 };
  });
  check("the tabs are a scrollable row", tabs.overflowX === "auto", tabs.fits ? "all four fit at this width" : "scrolls");
  const tl = await mobile.locator(".tl__canvas").boundingBox();
  await mobile.touchscreen.tap(tl.x + tl.width * 0.5, tl.y + 40);
  await mobile.waitForTimeout(100);
  const tapped = await mobile.locator(".tlp__transport .tp__now").innerText();
  check("a tap on the timeline goes to that moment", /^47:\d\d$/.test(tapped), tapped);
  // two fingers moving apart on the timeline zoom it in
  const touch = await phone.newCDPSession(mobile);
  const cy = tl.y + 40;
  const cx = tl.x + tl.width * 0.5;
  const fingers = (spread) => [
    { x: Math.round(cx - spread), y: Math.round(cy), id: 1 },
    { x: Math.round(cx + spread), y: Math.round(cy), id: 2 },
  ];
  await touch.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: fingers(20) });
  for (const spread of [30, 45, 60, 80, 100, 120]) await touch.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: fingers(spread) });
  await touch.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await mobile.waitForTimeout(150);
  const pinched = (await mobile.getByRole("button", { name: "Full match" }).count()) === 1;
  check("a pinch on the timeline zooms in", pinched, pinched ? await mobile.locator(".tlh__range").innerText() : "no zoom");
  if (pinched) {
    await mobile.getByRole("button", { name: "Full match" }).tap();
    check("Full match undoes the pinch", (await mobile.getByRole("button", { name: "Full match" }).count()) === 0);
  }
  await phone.close();
} finally {
  await browser.close();
  rmSync(work, { recursive: true, force: true });
}

check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
console.log(`\n${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  for (const name of failures) console.log(`  failed: ${name}`);
  process.exit(1);
}
