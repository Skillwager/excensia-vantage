"""How SkillCorner's own files agree with each other, and what the labels built from them look like.

    python scripts/skillcorner_labels.py --matches sc2017461 sc2016236

For each match: how much of each half is live, whether player possessions fall inside phases of
play, how the "for / against" wording of interruptions matches the kind of first touch, how often
the labels agree with the phase's team and with the per-frame possession flag, and the lengths
of the possessions and stoppages that come out. Open data, other matches only.
"""

import argparse
import collections
import csv
import os

import numpy as np

from vantage import A, D, H, X
from vantage import skillcorner
from vantage.external import ROOT
from vantage.labels import segments

FOLDER = os.path.join(ROOT, "data", "external", "skillcorner")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="+", default=list(skillcorner.MATCHES[:1]))
    args = parser.parse_args()
    for match in args.matches:
        sheet = skillcorner._sheet(match, FOLDER)
        raw = skillcorner._read_tracking(skillcorner._stem(match, FOLDER) + "_tracking_extrapolated.jsonl", sheet)
        phases, possessions = skillcorner._read_events(match, FOLDER, sheet)
        print("== {}: {} phases, {} player possessions".format(match, len(phases), len(possessions)))

        # how the wording of interruptions goes with the kind of first touch
        pairs = collections.Counter()
        with open(skillcorner._stem(match, FOLDER) + "_dynamic_events.csv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["event_type"] == skillcorner.POSSESSION and row["game_interruption_before"]:
                    pairs[(row["game_interruption_before"], row["start_type"])] += 1
        print("  interruption before -> how the possession started:")
        for (before, start), count in sorted(pairs.items()):
            print("    {:<20} {:<26} {}".format(before, start, count))

        for period, arrays in sorted(raw.items()):
            first = arrays["first"]
            last = first + len(arrays["ball"]) - 1
            labels = skillcorner.labels_at_source_rate(first, last, phases, possessions)
            t = len(labels)
            live = labels != D
            # player possessions that start outside every phase
            outside = sum(1 for start, _, team, _ in possessions
                          if first <= start <= last and not live[start - first])
            total = sum(1 for start, _, _, _ in possessions if first <= start <= last)
            # the phase's team, frame by frame
            phase_team = np.full(t, -1, dtype=np.int8)
            for start, stop, team in phases:
                if stop >= first and start <= last:
                    phase_team[max(start, first) - first:min(stop, last) - first + 1] = team
            both = live & (phase_team >= 0) & (labels != X)
            flag = arrays["group"]
            flagged = live & (flag >= 0) & (labels != X)
            print("  half {}: {} frames at 10 a second; live {:.1%}; no possession named {:.2%}".format(
                period, t, float(live.mean()), float((labels == X).mean())))
            print("    player possessions starting outside every phase: {} of {}".format(outside, total))
            print("    labels agree with the phase's team on {:.1%} of live frames".format(
                float((labels[both] == phase_team[both]).mean())))
            print("    labels agree with the per-frame flag on {:.1%} of the frames that have one ({:.1%} of live frames have one; {:.1%} of dead frames do)".format(
                float((labels[flagged] == flag[flagged]).mean()), float((flag[live] >= 0).mean()),
                float((flag[~live] >= 0).mean()) if (~live).any() else 0.0))
            runs = segments(labels)
            spells = np.array([stop - start for start, stop, code in runs if code in (H, A)]) / 10.0
            stops = np.array([stop - start for start, stop, code in runs if code == D]) / 10.0
            minutes = t / 600.0
            print("    possessions {} ({:.1f} a minute): middle {:.1f} s, under 2 s {:.0%}, under 1 s {:.0%}".format(
                len(spells), len(spells) / minutes, float(np.median(spells)), float((spells < 2).mean()),
                float((spells < 1).mean())))
            print("    stoppages {} ({:.1f} a minute): middle {:.1f} s, under 3 s {:.0%}; dead {:.1%} of the half".format(
                len(stops), len(stops) / minutes, float(np.median(stops)) if len(stops) else 0.0,
                float((stops < 3).mean()) if len(stops) else 0.0, float((labels == D).mean())))
            seen = np.isfinite(arrays["ball"][:, 0])
            anywhere = np.isfinite(arrays["ball_any"][:, 0])
            players = np.isfinite(arrays["home"][:, :, 0]).sum(axis=1) + np.isfinite(arrays["away"][:, :, 0]).sum(axis=1)
            on_screen = arrays["home_seen"].sum(axis=1) + arrays["away_seen"].sum(axis=1)
            print("    ball detected: {:.1%} of live frames, {:.1%} of dead frames; a ball position of any kind on {:.1%}".format(
                float(seen[live].mean()), float(seen[~live].mean()) if (~live).any() else 0.0, float(anywhere.mean())))
            print("    players with a position per frame: middle {:.0f} (none on {:.1%} of frames); on screen: middle {:.0f}".format(
                float(np.median(players)), float((players == 0).mean()), float(np.median(on_screen))))


if __name__ == "__main__":
    main()
