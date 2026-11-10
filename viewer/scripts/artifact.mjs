/**
 * Turns the built viewer (dist/index.html) into the page a Claude Artifact publishes
 * (dist/excensia-vantage.html).
 *
 * A published page is wrapped in its own document at publish time, so this file carries only what
 * goes inside it: the title, the font link, the styles, the mount point and the script. Nothing
 * else changes; the viewer is the same build either way.
 *
 *   npm run build && npm run artifact
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "dist", "index.html"), "utf8");

/** The text between the first `open` and the first `close` after it. An HTML parser ends a script or style the same way. */
function between(text, open, close) {
  const start = text.indexOf(open);
  if (start < 0) throw new Error(`not found in the build: ${open}`);
  const from = start + open.length;
  const end = text.indexOf(close, from);
  if (end < 0) throw new Error(`not closed in the build: ${open}`);
  return text.slice(from, end);
}

const script = between(html, '<script type="module">', "</script>");
const styles = between(html, '<style rel="stylesheet" crossorigin>', "</style>");
const title = between(html, "<title>", "</title>").trim();
const fonts = /href="(https:\/\/fonts\.googleapis\.com\/css2[^"]+)"/.exec(html)?.[1];
if (!fonts) throw new Error("the font link is missing from the build");

const page = [
  `<title>${title}</title>`,
  '<meta name="claude:viewport-fit-sides" content="cover">',
  `<link rel="stylesheet" href="${fonts}">`,
  `<style>${styles}</style>`,
  '<div id="root"></div>',
  `<script type="module">${script}</script>`,
  "",
].join("\n");

// the wrapper adds the document itself; the page's own markup must not carry a second one
const markup = page.slice(0, page.indexOf('<script type="module">')).toLowerCase();
for (const banned of ["<!doctype", "<html", "<head", "<body"]) {
  if (markup.includes(banned)) throw new Error(`the page must not carry ${banned}`);
}
const out = join(root, "dist", "excensia-vantage.html");
writeFileSync(out, page);
console.log(`${out}  ${(Buffer.byteLength(page) / 1000).toFixed(0)} kB`);
