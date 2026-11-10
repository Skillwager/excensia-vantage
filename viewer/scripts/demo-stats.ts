/**
 * Prints the vital signs of the simulated match. Run with: npx vite-node scripts/demo-stats.ts
 * Simulated data only.
 */
import { demoRun } from "../src/model/demo";
import { H, A, D, X } from "../src/model/types";

const t0 = performance.now();
const { run, scene } = demoRun();
const ms = performance.now() - t0;
const fps = run.fps;

const frames = [0, 0, 0, 0];
const spells: number[][] = [[], [], [], []];
for (let r = 0; r < run.runStart.length; r++) {
  const end = r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames;
  const len = end - run.runStart[r];
  frames[run.runLabel[r]] += len;
  spells[run.runLabel[r]].push(len / fps);
}
const pct = (n: number) => `${((100 * n) / run.frames).toFixed(1)}%`;
const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const median = (a: number[]) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] : 0);
const max = (a: number[]) => a.reduce((m, v) => Math.max(m, v), 0);

console.log(`generated in ${ms.toFixed(0)} ms; frames ${run.frames}; runs ${run.runStart.length}; changes ${run.changes.length}`);
console.log(`home ${pct(frames[H])}  away ${pct(frames[A])}  dead ${pct(frames[D])}  none ${pct(frames[X])}`);
const poss = [...spells[H], ...spells[A]];
console.log(`possessions ${poss.length}: mean ${mean(poss).toFixed(1)} s, median ${median(poss).toFixed(1)} s, max ${max(poss).toFixed(0)} s`);
console.log(`  home ${spells[H].length} mean ${mean(spells[H]).toFixed(1)}; away ${spells[A].length} mean ${mean(spells[A]).toFixed(1)}`);
console.log(`dead spells ${spells[D].length}: mean ${mean(spells[D]).toFixed(1)} s, max ${max(spells[D]).toFixed(0)} s; none ${spells[X].length}`);
console.log(`share of ball: home ${((100 * frames[H]) / (frames[H] + frames[A])).toFixed(1)}%`);

const tr = run.tracking!;
let unseen = 0;
for (let i = 0; i < tr.count; i++) if (tr.ball[i * 2] !== tr.ball[i * 2]) unseen++;
console.log(`ball unseen on ${((100 * unseen) / tr.count).toFixed(1)}% of ${tr.count} samples`);

// movement sanity
let fastest = 0;
let over9 = 0;
let outside = 0;
const dt = tr.every / fps;
for (const team of [tr.home, tr.away]) {
  for (let i = 1; i < tr.count; i++) {
    for (let p = 0; p < 11; p++) {
      const a = ((i - 1) * 11 + p) * 2;
      const b = (i * 11 + p) * 2;
      const v = Math.hypot(team[b] - team[a], team[b + 1] - team[a + 1]) / dt;
      if (v > fastest) fastest = v;
      if (v > 9) over9++;
      if (Math.abs(team[b]) > 55 || Math.abs(team[b + 1]) > 36.6) outside++;
    }
  }
}
console.log(`fastest player ${fastest.toFixed(1)} m/s; samples over 9 m/s ${over9}; samples outside the pitch margin ${outside}`);
let ballJump = 0;
let ballFast = 0;
for (let i = 1; i < tr.count; i++) {
  const a = (i - 1) * 2;
  const b = i * 2;
  if (tr.ball[a] !== tr.ball[a] || tr.ball[b] !== tr.ball[b]) continue;
  const v = Math.hypot(tr.ball[b] - tr.ball[a], tr.ball[b + 1] - tr.ball[a + 1]) / dt;
  if (v > ballFast) ballFast = v;
  if (v > 34) ballJump++;
}
console.log(`fastest ball ${ballFast.toFixed(1)} m/s; steps over 34 m/s ${ballJump}`);

const conf = run.segments.filter((s) => s.label !== X && s.confidence !== null).map((s) => s.confidence!);
console.log(`confidence: mean ${mean(conf).toFixed(2)}, below 0.6: ${conf.filter((c) => c < 0.6).length} of ${conf.length}`);
const whys = new Map<string, number>();
for (const c of run.changes) {
  const key = (c.why ?? "(none)").replace(/Home|Away/g, "T").replace(/ (in their own third|in midfield|high up the pitch)/, "").replace(/near|far/, "side");
  whys.set(key, (whys.get(key) ?? 0) + 1);
}
for (const [why, n] of [...whys].sort((a, b) => b[1] - a[1])) console.log(`  ${String(n).padStart(4)}  ${why}`);
console.log("goals", scene.goals.map((g) => `${Math.floor(g.frame / fps / 60)}:${String(Math.floor((g.frame / fps) % 60)).padStart(2, "0")} ${g.side ? "away" : "home"}`).join(", "));

// first half vs second half share
for (const p of run.periods) {
  let h = 0;
  let a = 0;
  for (let r = 0; r < run.runStart.length; r++) {
    const s = Math.max(run.runStart[r], p.start);
    const e = Math.min(r + 1 < run.runStart.length ? run.runStart[r + 1] : run.frames, p.end + 1);
    if (e <= s) continue;
    if (run.runLabel[r] === H) h += e - s;
    if (run.runLabel[r] === A) a += e - s;
  }
  console.log(`${p.name}: home ${((100 * h) / (h + a)).toFixed(1)}% of the ball`);
}
const at = 108350;
const seg = run.segments.find((s) => s.start <= at && s.end >= at)!;
console.log("72:14 →", ["H", "A", "D", "X"][seg.label], "conf", seg.confidence, "seen", seg.ballSeen, `spell ${(seg.start / fps / 60).toFixed(2)}–${(seg.end / fps / 60).toFixed(2)} min`, "ball seen now:", tr.ball[Math.round(at / tr.every) * 2] === tr.ball[Math.round(at / tr.every) * 2]);
