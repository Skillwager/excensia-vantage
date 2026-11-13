"""Figure 1 of the technical document: what a kick looks like to the tracker.

A kick changes the ball's velocity at once. This script takes every strong change of velocity
in a ball path (vantage/smoothing.py finds them without a label), centres each on its middle and
averages them, which shows over how many frames a tracker spreads a kick. It does so for the
provided recording and for simulated recordings of the ten open matches, with the simulator as
it was first built and as it is now, and draws the three averages.

Of the provided recording it reads the ball's path and keeps 41 averages over the whole match
and the number of changes they are averaged over: no frame, no position and no time.

    python scripts/figure_kick.py --data-dir /path/to/data    # measure everything, then draw
    python scripts/figure_kick.py                               # draw again from the saved numbers

The numbers go to docs/figures/kick_profile.json and the drawing to docs/figures/kick_profile.svg.
The PNG beside them is that SVG as a browser draws it, at twice the size:

    chrome --headless --window-size=800,450 --force-device-scale-factor=2 \
           --screenshot=docs/figures/kick_profile.png docs/figures/kick_profile.svg
"""

import argparse
import json
import os

import numpy as np

from vantage import FPS, competition, external, smoothing

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(HERE, "docs", "figures")
REACH = smoothing.PROFILE_REACH

# -- the drawing ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 450
LEFT, RIGHT, TOP, BOTTOM = 58, 22, 100, 58
LOW, HIGH = -0.02, 0.20                     # the share axis
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#b9b8ad"
RECORDING, FIRST, NOW = "#2a78d6", "#eb6834", "#1baf7a"
FONT = "'Segoe UI', Arial, Helvetica, sans-serif"


def average(balls):
    total, count = np.zeros(2 * REACH + 1), 0
    for ball in balls:
        part, found = smoothing.profile(ball)
        total += part
        count += found
    return {"changes": int(count), "share": [round(float(v), 4) for v in total / max(count, 1)]}


def simulated(generation):
    """The ten open matches as the simulator records them under its typical setting."""
    before = os.environ.get("VANTAGE_SIM")
    os.environ["VANTAGE_SIM"] = generation
    try:
        balls = [external.condition_half(half, match, "typical").ball_xy
                 for match in external.ALL_MATCHES for half in external.labelled_halves(match)]
    finally:
        if before is None:
            del os.environ["VANTAGE_SIM"]
        else:
            os.environ["VANTAGE_SIM"] = before
    return average(balls)


def middle(share, frames):
    """The share of a change that falls in its middle `frames` frames."""
    half = frames // 2
    return float(sum(share[REACH - half:REACH + half + 1]))


width_at_half = smoothing.half_width


def seconds(frames):
    return "{:g} s".format(round(frames / FPS, 2))


def x_of(frame):
    return LEFT + (frame + REACH) / (2.0 * REACH) * (WIDTH - LEFT - RIGHT)


def y_of(share):
    return HEIGHT - BOTTOM - (share - LOW) / (HIGH - LOW) * (HEIGHT - TOP - BOTTOM)


def text(x, y, words, size=12.5, fill=INK2, weight=400, anchor="start"):
    return ('<text x="{:.1f}" y="{:.1f}" font-size="{}" fill="{}" font-weight="{}" '
            'text-anchor="{}">{}</text>').format(x, y, size, fill, weight, anchor, words)


def path(share, colour, width):
    points = " ".join("{}{:.1f} {:.1f}".format("M" if i == 0 else "L", x_of(i - REACH), y_of(v))
                      for i, v in enumerate(share))
    return ('<path d="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round" '
            'stroke-linecap="round"/>').format(points, colour, width)


def draw(numbers):
    recording, first, now = (numbers[key]["share"] for key in ("recording", "first", "now"))
    wide_recording, wide_first = width_at_half(recording), width_at_half(first)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}" '
           'font-family="{2}" role="img" aria-labelledby="title desc">'.format(WIDTH, HEIGHT, FONT),
           '<title id="title">What a kick looks like to the tracker</title>',
           '<desc id="desc">Line chart. The share of a strong change in the ball\'s velocity that '
           'falls in each frame around its middle. In the recording and in the simulator as it is '
           'now, no frame holds more than about {:.0f}% and the curve is {:.0f} frames wide at half its '
           'height. In the simulator as first built the middle frame holds {:.0f}% and the curve is '
           '{:.0f} frames wide.</desc>'.format(100 * max(recording), wide_recording,
                                                100 * max(first), wide_first),
           '<rect width="{}" height="{}" fill="#ffffff"/>'.format(WIDTH, HEIGHT)]

    out.append(text(24, 36, "The recording blurs a kick the way a 21-frame moving average does",
                    size=19, fill=INK, weight=600))
    out.append(text(24, 58, "Every strong change in the ball’s velocity (a kick, a deflection, "
                            "a first touch) is centred on its middle.", size=13))
    out.append(text(24, 76, "The curves show the share of the change that falls in each frame, "
                            "averaged over all of them.", size=13))

    # grid, axes
    for share in (0.05, 0.10, 0.15, 0.20):
        out.append('<line x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="1"/>'.format(
            LEFT, WIDTH - RIGHT, y_of(share), y_of(share), GRID))
    out.append('<line x1="{0:.1f}" x2="{0:.1f}" y1="{1:.1f}" y2="{2:.1f}" stroke="{3}" stroke-width="1" '
               'stroke-dasharray="2 4"/>'.format(x_of(0), y_of(HIGH), y_of(0), AXIS))
    out.append('<line x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="1"/>'.format(
        LEFT, WIDTH - RIGHT, y_of(0), y_of(0), AXIS))
    for share in (0.0, 0.05, 0.10, 0.15, 0.20):
        out.append(text(LEFT - 9, y_of(share) + 4, "{:.0f}%".format(100 * share) if share else "0",
                        size=12, anchor="end"))
    for frame in (-20, -10, 0, 10, 20):
        label = "0" if frame == 0 else "{}{}".format("−" if frame < 0 else "+", abs(frame))
        out.append(text(x_of(frame), HEIGHT - BOTTOM + 17, label, size=12, anchor="middle"))
    out.append(text((LEFT + WIDTH - RIGHT) / 2.0, HEIGHT - 16,
                    "frames before and after the middle of the change ({} frames are one second)".format(FPS),
                    size=12.5, anchor="middle"))

    # the three averages: the simulators as lines, the recording as open circles on top
    out.append(path(first, FIRST, 2.25))
    out.append(path(now, NOW, 2.25))
    for i, share in enumerate(recording):
        out.append('<circle cx="{:.1f}" cy="{:.1f}" r="3.6" fill="none" stroke="{}" stroke-width="1.8"/>'.format(
            x_of(i - REACH), y_of(share), RECORDING))

    def mark(kind, colour, x, y):
        if kind == "circle":
            return '<circle cx="{:.1f}" cy="{:.1f}" r="3.6" fill="none" stroke="{}" stroke-width="1.8"/>'.format(
                x + 9, y, colour)
        return ('<line x1="{:.1f}" x2="{:.1f}" y1="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="2.25" '
                'stroke-linecap="round"/>').format(x, x + 18, y, y, colour)

    # legend, top left, where no curve goes
    rows = [
        ("circle", RECORDING, "The provided recording",
         "{} strong changes, found without labels".format(numbers["recording"]["changes"])),
        ("line", FIRST, "Simulator as first built", "ball path averaged over 5 frames"),
        ("line", NOW, "Simulator now", "averaged over 21 frames, then over 5"),
    ]
    out.append('<rect x="{}" y="{}" width="298" height="{}" fill="#ffffff"/>'.format(
        LEFT - 2, TOP + 4, 38 * len(rows) + 6))
    for row, (kind, colour, name, detail) in enumerate(rows):
        y = TOP + 22 + 38 * row
        out.append(mark(kind, colour, LEFT + 13, y - 4))
        out.append(text(LEFT + 40, y, name, size=12.5, fill=INK, weight=600))
        out.append(text(LEFT + 40, y + 15, detail, size=11.5))

    # the numbers worth reading off, each beside its curve
    notes = [
        ("line", FIRST, x_of(2.4), y_of(max(first)) + 9,
         "{:.0f}% of the change in the middle frame;".format(100 * max(first)), wide_first),
        ("circle", RECORDING, x_of(4.6), y_of(max(recording)) - 36,
         "at most {:.0f}% of the change in any frame;".format(100 * max(recording)), wide_recording),
    ]
    for kind, colour, x, y, words, wide in notes:
        out.append(mark(kind, colour, x, y - 4))
        out.append(text(x + 26, y, words, size=12.5, fill=INK))
        out.append(text(x + 26, y + 16, "{:.0f} frames wide at half its height ({})".format(
            wide, seconds(round(wide))), size=12.5, fill=INK))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="the competition data; without it the saved numbers are drawn again")
    args = parser.parse_args()
    os.makedirs(FIGURES, exist_ok=True)
    stored = os.path.join(FIGURES, "kick_profile.json")
    if args.data_dir:
        numbers = {
            "what": "share of a strong change of the ball's velocity that falls in each frame "
                    "around its middle, averaged over every such change",
            "frames": list(range(-REACH, REACH + 1)),
            "recording": average([half.ball_xy for half in competition.load_halves(args.data_dir)]),
            "first": simulated("g1"),
            "now": simulated("g7"),
        }
        with open(stored, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(numbers, handle, indent=1)
            handle.write("\n")
    else:
        with open(stored, encoding="utf-8") as handle:
            numbers = json.load(handle)
    with open(os.path.join(FIGURES, "kick_profile.svg"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(draw(numbers))
    for key, name in (("recording", "the recording"), ("first", "simulator as first built"),
                      ("now", "simulator now")):
        share = numbers[key]["share"]
        print("{:26s} {:6d} changes; in the middle 5 / 11 / 21 / 31 frames: {}; most in one frame "
              "{:.3f}; {:.1f} frames wide at half its height".format(
                  name, numbers[key]["changes"],
                  " ".join("{:.2f}".format(middle(share, n)) for n in (5, 11, 21, 31)),
                  max(share), width_at_half(share)))


if __name__ == "__main__":
    main()
