"""When SkillCorner's "out of play" begins and ends, read from where the ball is at those moments.

    python scripts/skillcorner_dead.py --matches sc2017461 sc2016236

A stoppage is a gap between phases of play. If the gap starts when the ball leaves the pitch, the
ball is on a line at its first frame; if it ends when the restart is taken, the ball is where
that restart is taken from (on the touchline for a throw-in, by a corner flag for a corner).
Also prints how long the gaps are. Open data, other matches only.
"""

import argparse
import collections
import os

import numpy as np

from vantage import D
from vantage import skillcorner
from vantage.external import ROOT

FOLDER = os.path.join(ROOT, "data", "external", "skillcorner")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="+", default=list(skillcorner.MATCHES[:2]))
    args = parser.parse_args()
    at_end = collections.defaultdict(list)       # kind of restart -> distance of the ball from a line when play stops
    at_start = collections.defaultdict(list)     # kind of restart -> where the ball is when play starts again
    lengths = []
    seen_end = seen_start = total = 0
    for match in args.matches:
        sheet = skillcorner._sheet(match, FOLDER)
        raw = skillcorner._read_tracking(skillcorner._stem(match, FOLDER) + "_tracking_extrapolated.jsonl", sheet)
        phases, possessions = skillcorner._read_events(match, FOLDER, sheet)
        half_l, half_w = sheet["pitch"][0] / 2, sheet["pitch"][1] / 2
        for period, arrays in sorted(raw.items()):
            first = arrays["first"]
            t = len(arrays["ball"])
            labels = skillcorner.labels_at_source_rate(first, first + t - 1, phases, possessions)
            ball = arrays["ball_any"]
            edges = np.diff(np.concatenate(([0], (labels == D).astype(np.int8), [0])))
            starts = {p[0] - first: p[3] for p in possessions if first <= p[0] <= first + t - 1}
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if begin == 0 or end >= t:
                    continue
                lengths.append((end - begin) / 10.0)
                # the first player possession after the gap says what kind of restart it was
                after = next((starts[row] for row in sorted(starts) if row >= end), "")
                kind = after.rsplit("_", 1)[0] if after else "none named"
                total += 1
                last_live, first_live = ball[begin - 1], ball[end]
                if np.isfinite(last_live[0]):
                    seen_end += 1
                    to_line = min(half_l - abs(last_live[0]), half_w - abs(last_live[1]))
                    at_end[kind].append(float(to_line))
                if np.isfinite(first_live[0]):
                    seen_start += 1
                    at_start[kind].append((float(half_w - abs(first_live[1])), float(half_l - abs(first_live[0]))))
    lengths = np.array(lengths)
    print("stoppages: {}; lengths in seconds: shortest {:.1f}, one in ten under {:.1f}, middle {:.1f}, one in ten over {:.1f}".format(
        len(lengths), float(lengths.min()), float(np.percentile(lengths, 10)), float(np.median(lengths)),
        float(np.percentile(lengths, 90))))
    print("  under 1 s: {}, 1-2 s: {}, 2-3 s: {}, 3-5 s: {}".format(
        int((lengths < 1).sum()), int(((lengths >= 1) & (lengths < 2)).sum()),
        int(((lengths >= 2) & (lengths < 3)).sum()), int(((lengths >= 3) & (lengths < 5)).sum())))
    print("ball position known at the last live frame of {} of {}, at the first live frame after of {}".format(
        seen_end, total, seen_start))
    print("\nwhen play stops, metres from the ball to the nearest line (negative = already over it):")
    for kind, values in sorted(at_end.items()):
        values = np.array(values)
        print("  before a {:<12} n={:<4} middle {:5.1f}   a quarter under {:5.1f}, a quarter over {:5.1f}".format(
            kind, len(values), float(np.median(values)), float(np.percentile(values, 25)), float(np.percentile(values, 75))))
    print("\nwhen play starts again, metres from the ball to the touchline / to the goal line:")
    for kind, values in sorted(at_start.items()):
        values = np.array(values)
        print("  after a {:<12} n={:<4} to the touchline: middle {:5.1f}    to the goal line: middle {:5.1f}".format(
            kind, len(values), float(np.median(values[:, 0])), float(np.median(values[:, 1]))))


if __name__ == "__main__":
    main()
