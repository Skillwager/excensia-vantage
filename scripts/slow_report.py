"""What the ball does while the players are slow: the recording beside the simulator. No labels.

A stoppage that the tracker follows right through has a ball on every frame, so "no ball" cannot
find it. What it does have is players who are close to standing still. This report splits the
frames with a tracked ball by how fast the players are moving on average, and for each band
prints how much of the recording it is and how the ball behaves: its speed, how far inside the
pitch it is, and how far from the nearest player.

The same table is printed for the provided recording (totals only, no labels involved) and for
external matches passed through the simulator, so the simulator's picture of a slow passage can
be compared with the real one.

    python scripts/slow_report.py --data-dir /path/to/data
    python scripts/slow_report.py --matches metrica1 J03WMX --conditions typical watched
"""

import argparse

import numpy as np

from vantage import competition, external
from vantage.features import build

BANDS = ((0.0, 1.0), (1.0, 1.3), (1.3, 1.8), (1.8, 99.0))
COLUMNS = ("b_speed", "b_margin", "b_dmin")


def describe(name, halves):
    tables = [build(half) for half in halves]
    names = tables[0][0]
    table = np.concatenate([t for _, t in tables])
    col = {n: table[:, i] for i, n in enumerate(names)}
    seen = np.isfinite(col["b_x"])
    speed = col["t_speed"]
    print("{}: ball tracked on {:.2f} of frames".format(name, seen.mean()))
    print("  {:<12}{:>8}{:>26}{:>26}{:>26}{:>10}".format(
        "players m/s", "frames", "ball speed p10/50/90", "inside pitch p10/50/90",
        "to nearest p10/50/90", "at rest"))
    for low, high in BANDS:
        mask = seen & (speed >= low) & (speed < high)
        if not mask.any():
            continue
        cells = []
        for column in COLUMNS:
            values = col[column][mask]
            values = values[np.isfinite(values)]
            cells.append("{:.2f}/{:.2f}/{:.2f}".format(*np.percentile(values, [10, 50, 90])))
        rest = np.mean(np.nan_to_num(col["b_speed"][mask], nan=99.0) < 0.5)
        print("  {:<12}{:>8.3f}{:>26}{:>26}{:>26}{:>10.2f}".format(
            "{:.1f} to {:.1f}".format(low, min(high, 9.9)), mask.mean(), *cells, rest))
    hidden = ~seen
    print("  ball untracked: {:.3f} of frames; of those, players under 1.3 m/s: {:.2f}".format(
        hidden.mean(), np.mean(np.nan_to_num(speed[hidden], nan=0.0) < 1.3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--matches", nargs="*", default=[])
    parser.add_argument("--conditions", nargs="*", default=["typical"])
    args = parser.parse_args()
    if args.data_dir:
        describe("the recording", competition.load_halves(args.data_dir))
    for match in args.matches:
        halves = external.labelled_halves(match)
        for condition in args.conditions:
            describe("{} {}".format(match, condition),
                     [external.condition_half(half, match, condition) for half in halves])


if __name__ == "__main__":
    main()
