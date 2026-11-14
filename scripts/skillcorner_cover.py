"""How much of SkillCorner's dead-ball time and live time has tracking at all.

    python scripts/skillcorner_cover.py [--matches sc2017461 ...]

The broadcast picture cuts away during stoppages, so this source has frames with no tracked ball
and frames with nobody tracked at all. Prints, over the chosen matches (all twenty by default),
the share of dead-ball frames and of live frames without a ball and without any player, from
the labels built in vantage/skillcorner.py. Open data only; totals only.
"""

import argparse

import numpy as np

from vantage import D
from vantage.external import SKILLCORNER_MATCHES, labelled_halves


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", default=None)
    args = parser.parse_args()
    counts = {"dead": [0, 0, 0], "live": [0, 0, 0]}        # frames, without a ball, without anybody
    for match in (args.matches or SKILLCORNER_MATCHES):
        for half in labelled_halves(match):
            no_ball = ~np.isfinite(half.ball_xy[:, 0])
            players = np.concatenate([half.home_xy[:, :, 0], half.away_xy[:, :, 0]], axis=1)
            nobody = no_ball & ~np.isfinite(players).any(axis=1)
            dead = half.labels == D
            for name, mask in (("dead", dead), ("live", ~dead)):
                counts[name][0] += int(mask.sum())
                counts[name][1] += int((mask & no_ball).sum())
                counts[name][2] += int((mask & nobody).sum())
    for name in ("dead", "live"):
        frames, no_ball, nobody = counts[name]
        print("{} frames {:>8}: no tracked ball {:.3f}; no ball and no player {:.3f}".format(
            name, frames, no_ball / frames, nobody / frames))


if __name__ == "__main__":
    main()
