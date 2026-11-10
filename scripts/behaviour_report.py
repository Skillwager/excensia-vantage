"""How the method behaves on held-out external matches, in totals that need no ground truth.

For every held-out match and simulated recording condition of a cross-validation run, this
prints the same totals scripts/pipeline_check.py prints for the competition file: how much of the
time the ball is untracked, how much of that is called dead ball, possessions per minute of live
play, how long possessions and stoppages are. Next to each is the same total for the true labels.

Two uses. It shows where the method is biased when the truth is known (it finds fewer possessions
than there are). And it gives the competition file something to be compared with: if the file
sits outside everything seen here, the recording differs from what was simulated, and that is
visible without a single label of the match.

    python scripts/behaviour_report.py --tag v7 --decode '<json>'
"""

import argparse
import glob
import json
import os

import numpy as np

from vantage import decode as decoding
from vantage.external import CONDITIONS, ROOT, condition_half, labelled_halves
from vantage.labels import behaviour

COLUMNS = (("hidden", "untracked"), ("hidden_called_dead", "untr>dead"),
           ("visible_called_dead", "trk>dead"), ("dead", "dead"),
           ("per_live_minute", "poss/min"), ("possession_median", "poss med"),
           ("possession_under3", "under 3s"), ("stoppage_median", "stop med"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--decode", default="{}", help="JSON of DecodeConfig fields")
    parser.add_argument("--settings", nargs="*", default=None)
    args = parser.parse_args()
    config = decoding.DecodeConfig(**json.loads(args.decode))

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "cv", args.tag, "prob_*.npz"))):
        match, setting_name = os.path.basename(path)[5:-4].rsplit("_", 1)
        if setting_name not in CONDITIONS or (args.settings and setting_name not in args.settings):
            continue
        predicted, truth = [], []
        with np.load(path) as data:
            for index, half in enumerate(labelled_halves(match)):
                seen = np.isfinite(condition_half(half, match, setting_name).ball_xy[:, 0])
                boundary = data["b{}".format(index)] if "b{}".format(index) in data else None
                labels = decoding.decode(data["p{}".format(index)].astype(np.float64), config,
                                         boundary=boundary)
                predicted.append((labels, seen))
                truth.append((data["y{}".format(index)], seen))
        rows.append((setting_name, match, behaviour(predicted), behaviour(truth)))

    header = "{:<9}{:<10}".format("condition", "match") + "".join(
        "{:>16}".format(title) for _, title in COLUMNS)
    print("each cell: the method / the truth\n" + header)
    for setting_name in sorted({row[0] for row in rows}):
        chosen = [row for row in rows if row[0] == setting_name]
        for _, match, mine, true in chosen:
            print("{:<9}{:<10}".format(setting_name, match) + "".join(
                "{:>16}".format("{:.2f} / {:.2f}".format(mine[key], true[key]))
                for key, _ in COLUMNS))
        print("{:<9}{:<10}".format(setting_name, "mean") + "".join(
            "{:>16}".format("{:.2f} / {:.2f}".format(
                np.mean([row[2][key] for row in chosen]), np.mean([row[3][key] for row in chosen])))
            for key, _ in COLUMNS))
    for label, position in (("the method", 2), ("the truth", 3)):
        print("\nrange and mean over all cells, " + label)
        for key, title in COLUMNS:
            values = [row[position][key] for row in rows]
            print("  {:<10} {:.3f} to {:.3f}   mean {:.3f}".format(
                title, min(values), max(values), float(np.mean(values))))


if __name__ == "__main__":
    main()
