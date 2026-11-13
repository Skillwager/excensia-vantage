"""Does the ball's path, extended into a touch from both sides, find where the ball really was?

A label-free check of one reading of the recording: that its ball path is a moving average of a
sharp one (vantage/smoothing.py, vantage/corners.py). If it is, the path outside the reach of
the average is still true, a straight line coming into a touch and a straight line going out.
Extended to the frame of the touch the two lines must meet, and they must meet at somebody's
feet, because that is where a touch happens. So at every touch this script compares

  * how far the reported ball is from the nearest player, and
  * how far the rebuilt corner is from the nearest player,

and says how far apart the two lines end. For simulated recordings of the open matches the
published ball is known as well, and the error of both against it is printed. On a path that
is not smoothed the rebuilding should do harm, and the open matches as published show that too.

Totals over a whole match only: counts, medians and shares. No label, no frame, no position.

    python scripts/corner_check.py --data-dir /path/to/data     # the recording and the open matches
    python scripts/corner_check.py                               # the open matches only
"""

import argparse
import os

import numpy as np

from vantage import FPS, competition, corners, external
from vantage.features import BALL_MAX_SPEED, BALL_VEL_HALFSPAN, _central_velocity

BINS = ((3.0, 6.0), (6.0, 10.0), (10.0, 1e9))
SHIFTS = (-6, -4, -2, 0, 2, 4, 6)


def nearest(point, home, away):
    """Distance from each point to the nearest player of either team at the same frame."""
    out = np.full(len(point), np.inf)
    for xy in (home, away):
        distance = np.sqrt(np.sum((xy - point[:, None, :]) ** 2, axis=2))
        out = np.minimum(out, np.where(np.isfinite(distance), distance, np.inf).min(axis=1))
    return out


def measure(halves, truth=None, reach=corners.REACH):
    keys = ("strength", "ball", "corner", "apart", "ball_error", "corner_error")
    rows = {key: [] for key in keys}
    later = {shift: [] for shift in SHIFTS}
    for index, half in enumerate(halves):
        ball = half.ball_xy.astype(np.float32)
        velocity = _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED)
        frames, strength = corners.touches(velocity)
        frames = frames[(frames >= reach + max(SHIFTS)) & (frames < len(ball) - reach - max(SHIFTS))]
        if not len(frames):
            continue
        line_in, line_out = corners.lines(ball, velocity, frames, FPS, reach)
        whole = np.isfinite(line_in).all(axis=1) & np.isfinite(line_out).all(axis=1)
        frames, line_in, line_out = frames[whole], line_in[whole], line_out[whole]
        corner = 0.5 * (line_in.astype(np.float64) + line_out.astype(np.float64))
        reported = ball[frames].astype(np.float64)
        home, away = half.home_xy.astype(np.float64), half.away_xy.astype(np.float64)
        rows["strength"].append(strength[frames])
        rows["ball"].append(nearest(reported, home[frames], away[frames]))
        rows["corner"].append(nearest(corner, home[frames], away[frames]))
        rows["apart"].append(np.sqrt(np.sum((line_in.astype(np.float64) - line_out) ** 2, axis=1)))
        for shift in SHIFTS:
            later[shift].append(nearest(corner, home[frames + shift], away[frames + shift]))
        if truth is not None:
            true = truth[index].astype(np.float64)[frames]
            rows["ball_error"].append(np.sqrt(np.sum((reported - true) ** 2, axis=1)))
            rows["corner_error"].append(np.sqrt(np.sum((corner - true) ** 2, axis=1)))
    out = {key: np.concatenate(value) if value else np.array([]) for key, value in rows.items()}
    out["later"] = {shift: np.concatenate(value) if value else np.array([]) for shift, value in later.items()}
    return out


def median(values):
    values = values[np.isfinite(values)]
    return "{:.2f}".format(np.median(values)) if len(values) else " n/a"


def report(name, got):
    print("\n== {}: {} touches".format(name, len(got["strength"])))
    if not len(got["strength"]):
        return
    for low, high in BINS:
        pick = (got["strength"] >= low) & (got["strength"] < high)
        if not pick.any():
            continue
        label = "{:.0f} to {:.0f} m/s".format(low, high) if high < 1e8 else "{:.0f} m/s or more".format(low)
        line = ("  change of {:<14} n {:5d}   nearest player, median m: reported ball {}  rebuilt corner {}"
                "   within a metre of a player: {:3.0%} and {:3.0%}   the two lines end {} m apart").format(
                    label, int(pick.sum()), median(got["ball"][pick]), median(got["corner"][pick]),
                    np.mean(got["ball"][pick] < 1.0), np.mean(got["corner"][pick] < 1.0), median(got["apart"][pick]))
        if len(got["ball_error"]):
            line += "   against the published ball, median m: reported {}  rebuilt {}".format(
                median(got["ball_error"][pick]), median(got["corner_error"][pick]))
        print(line)
    hard = got["strength"] >= 6.0
    print("  the players taken s frames later (6 m/s or more), median m from the corner: " + "  ".join(
        "{:+d}: {}".format(shift, median(values[hard])) for shift, values in sorted(got["later"].items())))


def simulated(generation):
    before = os.environ.get("VANTAGE_SIM")
    os.environ["VANTAGE_SIM"] = generation
    try:
        halves, truth = [], []
        for match in external.ALL_MATCHES:
            for half in external.labelled_halves(match):
                halves.append(external.condition_half(half, match, "typical"))
                truth.append(half.ball_xy)
    finally:
        if before is None:
            del os.environ["VANTAGE_SIM"]
        else:
            os.environ["VANTAGE_SIM"] = before
    return halves, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="the competition data; without it only the open matches are read")
    parser.add_argument("--reach", type=int, default=corners.REACH,
                        help="frames either side of a touch at which the lines are taken")
    args = parser.parse_args()
    print("lines taken {} frames either side of a touch".format(args.reach))
    if args.data_dir:
        report("the recording", measure(competition.load_halves(args.data_dir), reach=args.reach))
    halves, truth = simulated("g2")
    report("simulated recordings of the open matches", measure(halves, truth, args.reach))
    halves, truth = simulated("g1")
    report("the same with the simulator as first built", measure(halves, truth, args.reach))
    published = [half for match in external.ALL_MATCHES for half in external.labelled_halves(match)]
    report("the open matches as published", measure(published, [half.ball_xy for half in published], args.reach))


if __name__ == "__main__":
    main()
