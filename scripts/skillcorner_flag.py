"""Why SkillCorner's per-frame possession flag and the labels built from its events differ.

    python scripts/skillcorner_flag.py --match sc2017461

Prints the agreement at different time shifts, where the disagreements sit (how long after a
change of possession), a stretch of both written out side by side, and how much of the tracking
is missing in live and in dead frames. Open data, other matches only.
"""

import argparse
import os

import numpy as np

from vantage import A, D, H, X
from vantage import skillcorner
from vantage.external import ROOT

FOLDER = os.path.join(ROOT, "data", "external", "skillcorner")
LETTER = {H: "H", A: "A", D: ".", X: "?", -1: "-"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default=skillcorner.MATCHES[0])
    parser.add_argument("--show-from", type=int, default=600, help="first frame of the stretch written out, in the half")
    args = parser.parse_args()
    sheet = skillcorner._sheet(args.match, FOLDER)
    raw = skillcorner._read_tracking(skillcorner._stem(args.match, FOLDER) + "_tracking_extrapolated.jsonl", sheet)
    phases, possessions = skillcorner._read_events(args.match, FOLDER, sheet)
    for period, arrays in sorted(raw.items()):
        first = arrays["first"]
        t = len(arrays["ball"])
        labels = skillcorner.labels_at_source_rate(first, first + t - 1, phases, possessions)
        flag = arrays["group"]
        live = labels != D
        print("== half {}".format(period))
        for shift in (-20, -10, -5, -2, 0, 2, 5, 10, 20):
            moved = np.roll(flag, shift)
            ok = live & (moved >= 0) & (labels != X)
            print("  flag moved {:+3d} frames: agreement {:.1%}".format(shift, float((labels[ok] == moved[ok]).mean())))
        # the flag held: a frame without a flag takes the last flag before it, within live play
        held = flag.copy()
        last = -1
        for i in range(t):
            if not live[i]:
                last = -1
            elif held[i] >= 0:
                last = held[i]
            else:
                held[i] = last
        ok = live & (held >= 0) & (labels != X)
        print("  flag carried forward over its gaps: agreement {:.1%}".format(float((labels[ok] == held[ok]).mean())))
        # where do they disagree: frames since the label last changed
        change = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))
        since = np.arange(t) - change[np.searchsorted(change, np.arange(t), side="right") - 1]
        ok = live & (flag >= 0) & (labels != X)
        wrong = ok & (labels != flag)
        for low, high in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10 ** 9)):
            band = ok & (since >= low) & (since < high)
            if band.any():
                print("  {:>3}-{:<4} frames after the label changed: {:.1%} of frames disagree ({} frames)".format(
                    low, high if high < 10 ** 9 else "", float(wrong[band].mean()), int(band.sum())))
        # runs of the flag: how long, how many a minute
        runs = np.flatnonzero(np.concatenate(([True], flag[1:] != flag[:-1])))
        lengths = np.diff(np.concatenate((runs, [t])))
        codes = flag[runs]
        team_runs = lengths[codes >= 0]
        print("  the flag: {} runs of a team ({:.1f} a minute), middle {:.1f} s".format(
            len(team_runs), len(team_runs) / (t / 600.0), float(np.median(team_runs)) / 10.0))
        start = args.show_from
        for begin in range(start, start + 600, 100):
            print("  {:>6}  labels {}".format(begin, "".join(LETTER[int(v)] for v in labels[begin:begin + 100])))
            print("          flag   {}".format("".join(LETTER[int(v)] for v in flag[begin:begin + 100])))
        players = np.isfinite(arrays["home"][:, :, 0]).sum(axis=1) + np.isfinite(arrays["away"][:, :, 0]).sum(axis=1)
        print("  no player positions at all: {:.1%} of live frames, {:.1%} of dead frames".format(
            float((players[live] == 0).mean()), float((players[~live] == 0).mean())))
        print("  22 player positions: {:.1%} of live frames, {:.1%} of dead frames".format(
            float((players[live] >= 22).mean()), float((players[~live] >= 22).mean())))
        # stoppages by how much of them has tracking
        edges = np.diff(np.concatenate(([0], (labels == D).astype(np.int8), [0])))
        shares = [float((players[a:b] > 0).mean()) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        shares = np.array(shares)
        print("  stoppages: {} in all; tracked throughout {}, not at all {}, partly {}".format(
            len(shares), int((shares > 0.98).sum()), int((shares < 0.02).sum()), int(((shares >= 0.02) & (shares <= 0.98)).sum())))


if __name__ == "__main__":
    main()
