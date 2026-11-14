"""Figure 2 of the technical document: the earlier and the new models on four kinds of recording.

The decision for the submitted models was made on recordings from the simulator they were
trained on, which is a home ground. Section 6 sets beside it the earlier models' home ground,
a seventh scoring condition and real automatic tracking from another provider. This script
draws those four pairs of numbers; it measures nothing of the provided recording.

    python scripts/figure_checks.py              # draw from docs/figures/cross_checks.json
    python scripts/figure_checks.py --check      # first compare the three cross-validation rows
                                                 # with the stored runs under runs/cv

The numbers in docs/figures/cross_checks.json are the means that other scripts print; each row
names its command. The drawing goes to docs/figures/cross_checks.svg, and the PNG beside it is
that SVG as a browser draws it, at twice the size:

    chrome --headless --window-size=800,384 --force-device-scale-factor=2 \
           --screenshot=docs/figures/cross_checks.png docs/figures/cross_checks.svg
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(HERE, "docs", "figures")

# -- the drawing ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 384
LEFT, RIGHT = 318, 60                       # the axis; the rows' names stand to its left
TOP, STEP = 142, 52
LOW, HIGH = 0.62, 0.72                      # mean spellF1
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#b9b8ad"
EARLIER, NEW = "#eb6834", "#1baf7a"         # as the first and the present simulator in Figure 1
FONT = "'Segoe UI', Arial, Helvetica, sans-serif"


def x_of(value):
    return LEFT + (value - LOW) / (HIGH - LOW) * (WIDTH - LEFT - RIGHT)


def text(x, y, words, size=12.5, fill=INK2, weight=400, anchor="start"):
    return ('<text x="{:.1f}" y="{:.1f}" font-size="{}" fill="{}" font-weight="{}" '
            'text-anchor="{}">{}</text>').format(x, y, size, fill, weight, anchor, words)


def dot(x, y, colour):
    return '<circle cx="{:.1f}" cy="{:.1f}" r="6.5" fill="{}" stroke="#ffffff" stroke-width="2"/>'.format(
        x, y, colour)


def draw(numbers):
    rows = numbers["rows"]
    bottom = TOP + STEP * len(rows) - 26
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}" '
           'font-family="{2}" role="img" aria-labelledby="title desc">'.format(WIDTH, HEIGHT, FONT),
           '<title id="title">The earlier and the new models on four kinds of recording</title>',
           '<desc id="desc">Dot plot of mean spellF1. ' + " ".join(
               "{}: earlier models {:.3f}, new models {:.3f}.".format(row["name"], row["earlier"], row["new"])
               for row in rows) + '</desc>',
           '<rect width="{}" height="{}" fill="#ffffff"/>'.format(WIDTH, HEIGHT)]
    out.append(text(24, 36, "Each set of models is ahead at home; on real tracking the new ones are",
                    size=19, fill=INK, weight=600))
    out.append(text(24, 58, "Mean spellF1 of the earlier models and of the new ones on four kinds of "
                            "recording.", size=13))
    out.append(text(24, 76, "Ten held-out matches under six conditions; one condition for the third "
                            "row; twenty matches for the fourth.", size=13))
    # legend
    out.append(dot(31, 101, EARLIER))
    out.append(text(44, 105.5, "Earlier models, trained on the first simulator’s recordings", size=12.5, fill=INK))
    out.append(dot(431, 101, NEW))
    out.append(text(444, 105.5, "New models, trained on the present simulator’s", size=12.5, fill=INK))
    # grid and axis
    ticks = [round(LOW + 0.02 * i, 2) for i in range(int(round((HIGH - LOW) / 0.02)) + 1)]
    for tick in ticks:
        out.append('<line x1="{0:.1f}" x2="{0:.1f}" y1="{1}" y2="{2}" stroke="{3}" stroke-width="1"/>'.format(
            x_of(tick), TOP - 26, bottom, GRID))
        out.append(text(x_of(tick), bottom + 18, "{:.2f}".format(tick), size=12, anchor="middle"))
    out.append('<line x1="{:.1f}" x2="{:.1f}" y1="{}" y2="{}" stroke="{}" stroke-width="1"/>'.format(
        x_of(LOW), x_of(HIGH), bottom, bottom, AXIS))
    out.append(text((x_of(LOW) + x_of(HIGH)) / 2.0, bottom + 38, "mean spellF1", size=12.5, anchor="middle"))
    # the rows: a name and a line of detail on the left, the two means on the axis
    for index, row in enumerate(rows):
        y = TOP + STEP * index
        out.append(text(24, y - 3, row["name"], size=13, fill=INK, weight=600))
        out.append(text(24, y + 14, row["detail"], size=11.5))
        earlier, new = row["earlier"], row["new"]
        out.append('<line x1="{:.1f}" x2="{:.1f}" y1="{}" y2="{}" stroke="{}" stroke-width="2"/>'.format(
            x_of(min(earlier, new)), x_of(max(earlier, new)), y, y, AXIS))
        for value, colour, other in ((earlier, EARLIER, new), (new, NEW, earlier)):
            out.append(dot(x_of(value), y, colour))
            side = 1 if value > other else -1
            out.append(text(x_of(value) + side * 13, y + 4.5, "{:.3f}".format(value), size=12.5, fill=INK,
                            weight=600, anchor="start" if side > 0 else "end"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def check(numbers):
    """The three cross-validation rows again from the stored runs, beside the saved numbers."""
    from compare_cv import SKIP, chosen, load

    def mean_of(tag, setting_from=None):
        cells, _, _ = load(tag)
        chooser = load(setting_from)[0] if setting_from else cells
        setting = chosen(chooser, [key for key in chooser if key[0] not in SKIP])
        matches = sorted({key[1] for key in cells if key[0] not in SKIP})
        return float(np.mean([np.mean([cells[key][setting] for key in cells if key[1] == match and key[0] not in SKIP])
                              for match in matches]))

    for row in numbers["rows"]:
        runs = row.get("runs")
        if not runs:
            print("{:<34} not checked here: {}".format(row["name"], row["printed_by"]))
            continue
        earlier = mean_of(runs["earlier"], runs.get("earlier_setting_from"))
        new = mean_of(runs["new"], runs.get("new_setting_from"))
        print("{:<34} earlier {:.4f} (saved {:.4f})   new {:.4f} (saved {:.4f})".format(
            row["name"], earlier, row["earlier"], new, row["new"]))
        if abs(earlier - row["earlier"]) > 0.00006 or abs(new - row["new"]) > 0.00006:
            raise SystemExit("the saved numbers of this row are not what the stored runs give")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="compare the cross-validation rows with the stored runs before drawing")
    args = parser.parse_args()
    with open(os.path.join(FIGURES, "cross_checks.json"), encoding="utf-8") as handle:
        numbers = json.load(handle)
    if args.check:
        check(numbers)
    with open(os.path.join(FIGURES, "cross_checks.svg"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(draw(numbers))
    for row in numbers["rows"]:
        print("{:<34} earlier {:.4f}   new {:.4f}   {:+.4f}".format(
            row["name"], row["earlier"], row["new"], row["new"] - row["earlier"]))


if __name__ == "__main__":
    main()
