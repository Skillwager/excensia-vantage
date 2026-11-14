"""The technical document as one HTML page, and as a PDF where a browser is at hand.

A Markdown file uploaded on its own shows no figure and no table. This script turns
docs/TECHNICAL_DOCUMENT.md into a page that carries its figures inside it, and prints that page
to a PDF with a Chromium browser (Chrome or Edge) if one is installed. It reads the kinds of
Markdown the document uses and no others: headings, paragraphs, lists (and a list inside an
entry of another, one level deep), tables, fenced code, images, and bold, italic, code and
links inside a line. No package is needed.

    python scripts/document_pdf.py                       # docs/TECHNICAL_DOCUMENT.html and .pdf
    python scripts/document_pdf.py --out submission/Excensia
"""

import argparse
import base64
import html
import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(HERE, "docs", "TECHNICAL_DOCUMENT.md")

STYLE = """
@page { size: A4; margin: 17mm 16mm 15mm 16mm; }
html { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.42;
       color: #111; }
body { margin: 0 auto; max-width: 178mm; }
h1 { font-size: 18pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 16pt; }
li { margin: 0 0 3pt; }
li > ul, li > ol { margin: 3pt 0 0; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 9pt; }
pre { background: #f4f4f1; padding: 6pt 8pt; border-radius: 3pt; overflow-wrap: anywhere; white-space: pre-wrap;
      font-size: 8.6pt; line-height: 1.35; margin: 0 0 8pt; }
pre code { font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 2pt 0 9pt; font-size: 8.7pt; line-height: 1.32; }
th, td { border-bottom: 0.5pt solid #c9c8c0; padding: 3pt 5pt 3pt 0; text-align: left; vertical-align: top; }
th { border-bottom: 0.9pt solid #555; font-weight: 600; }
tr { break-inside: avoid; }
tr.heads { break-after: avoid; }
img { max-width: 100%; display: block; margin: 4pt 0 3pt; }
a { color: #1b4f9c; text-decoration: none; overflow-wrap: anywhere; }
"""


def inline(text):
    """One line's worth of Markdown inside a block: code, bold, italic, links, bare addresses."""
    spans = []

    def keep(match):
        spans.append("<code>{}</code>".format(html.escape(match.group(1), quote=False)))
        return "\x00{}\x00".format(len(spans) - 1)

    text = re.sub(r"`([^`]*)`", keep, text)             # code first: nothing inside it is markup
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"(?<![\"=>])\b(https?://[^\s<)]+[^\s<).,;:])", r'<a href="\1">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub("\x00(\\d+)\x00", lambda match: spans[int(match.group(1))], text)


def image(alt, target, folder):
    """An <img> with the picture inside the page, so the page is one file."""
    path = os.path.normpath(os.path.join(folder, target))
    kind = {"png": "image/png", "svg": "image/svg+xml", "jpg": "image/jpeg"}[path.rsplit(".", 1)[-1].lower()]
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("ascii")
    return '<img alt="{}" src="data:{};base64,{}">'.format(html.escape(alt), kind, data)


def cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def convert(text, folder):
    lines = text.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    item = re.compile(r"^(\s*)(\d+\.|-)\s+(.*)$")
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>{}</code></pre>".format(html.escape("\n".join(body), quote=False)))
        elif re.match(r"^#{1,6} ", line):
            level = len(line) - len(line.lstrip("#"))
            out.append("<h{0}>{1}</h{0}>".format(level, inline(line[level:].strip())))
            i += 1
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{3,}", lines[i + 1]):
            head = cells(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            table = ["<table><thead><tr>{}</tr></thead><tbody>".format(
                "".join("<th>{}</th>".format(inline(cell)) for cell in head))]
            for row in rows:
                # a row that only heads the rows under it stays on their page
                heads = row[0].startswith("**") and row[0].endswith("**") and not any(row[1:])
                table.append("<tr{}>{}</tr>".format(' class="heads"' if heads else "", "".join(
                    "<td>{}</td>".format(inline(cell)) for cell in row)))
            out.append("".join(table) + "</tbody></table>")
        elif re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line):
            match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
            out.append(image(match.group(1), match.group(2), folder))
            i += 1
        elif item.match(line):
            ordered = item.match(line).group(2) != "-"
            entries = []
            while i < len(lines) and item.match(lines[i]) and not lines[i].startswith(" "):
                words = [item.match(lines[i]).group(3)]
                inner, inner_ordered = [], False       # a list inside this entry, one level deep
                i += 1
                while i < len(lines) and lines[i].strip() and lines[i].startswith(" "):
                    nested = item.match(lines[i])
                    if nested:
                        inner_ordered = nested.group(2) != "-" if not inner else inner_ordered
                        inner.append([nested.group(3)])
                    else:
                        (inner[-1] if inner else words).append(lines[i].strip())
                    i += 1
                inside = ""
                if inner:
                    inside = "<{0}>{1}</{0}>".format("ol" if inner_ordered else "ul", "".join(
                        "<li>{}</li>".format(inline(" ".join(part))) for part in inner))
                entries.append("<li>{}{}</li>".format(inline(" ".join(words)), inside))
            out.append("<{0}>{1}</{0}>".format("ol" if ordered else "ul", "".join(entries)))
        else:
            words = []
            while (i < len(lines) and lines[i].strip() and not lines[i].startswith(("```", "|", "#"))
                   and not item.match(lines[i]) and not lines[i].startswith("![")):
                words.append(lines[i].strip())
                i += 1
            out.append("<p>{}</p>".format(inline(" ".join(words))))
    return "\n".join(out)


def browser():
    for name in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if os.path.isfile(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", default=os.path.splitext(SOURCE)[0], help="path without the ending")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    with open(args.source, encoding="utf-8") as handle:
        text = handle.read()
    title = text.split("\n", 1)[0].lstrip("# ").strip()
    page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>{}</title>\n'
            "<style>{}</style></head>\n<body>\n{}\n</body></html>\n").format(
                html.escape(title), STYLE, convert(text, os.path.dirname(os.path.abspath(args.source))))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".html", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    print("wrote " + args.out + ".html")
    program = None if args.no_pdf else browser()
    if program:
        # a profile of its own, so that a browser that is already open is left alone
        profile = tempfile.mkdtemp(prefix="vantage-document-")
        try:
            subprocess.run([program, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            "--user-data-dir=" + profile,
                            "--print-to-pdf=" + os.path.abspath(args.out + ".pdf"),
                            "file:///" + os.path.abspath(args.out + ".html").replace("\\", "/")],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            shutil.rmtree(profile, ignore_errors=True)
        print("wrote " + args.out + ".pdf")
    elif not args.no_pdf:
        print("no Chromium browser found: open the HTML page and print it to a PDF")


if __name__ == "__main__":
    main()
