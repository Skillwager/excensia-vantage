"""The regime of each scoring condition: how much of the dead-ball time and how much of the live
time has no tracked ball, from the open matches' own labels.

    python scripts/regime_report.py                       # the six conditions and "mixed"
    python scripts/regime_report.py --conditions mixed typical

A finished file gives the same two shares for the provided recording by its own reading
(scripts/pipeline_check.py prints what they are made of), and scripts/skillcorner_cover.py gives
them for SkillCorner's recordings. Open matches only; totals only.
"""

import argparse

import numpy as np

from vantage import D, external
from vantage.external import ALL_MATCHES, labelled_halves


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conditions", nargs="*",
                        default=["easy", "typical", "watched", "random1", "random2", "harsh", "mixed"])
    args = parser.parse_args()
    print("{:<9} {:>22} {:>22} {:>12}".format(
        "condition", "dead frames untracked", "live frames untracked", "untracked"))
    for name in args.conditions:
        dead_hidden = dead = live_hidden = live = 0
        for match in ALL_MATCHES:
            for half in labelled_halves(match):
                seen = np.isfinite(external.condition_half(half, match, name).ball_xy[:, 0])
                is_dead = half.labels == D
                dead += int(is_dead.sum())
                dead_hidden += int((is_dead & ~seen).sum())
                live += int((~is_dead).sum())
                live_hidden += int((~is_dead & ~seen).sum())
        print("{:<9} {:>22.3f} {:>22.3f} {:>12.3f}".format(
            name, dead_hidden / dead, live_hidden / live, (dead_hidden + live_hidden) / (dead + live)), flush=True)


if __name__ == "__main__":
    main()
