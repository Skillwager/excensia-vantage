"""Can a classifier tell the provided recording from the simulated training data?

A label-free check on the simulator. Feature rows from the competition match and rows from the
simulated external matches are mixed and a classifier is asked which is which. If it cannot
(AUC near 0.5) the simulator is doing its job. Where it can, the features it leans on are exactly
the ones that look different in the real recording, and those are either a flaw in the simulator
to fix or a feature the possession model should not be given.

Prints totals only: an AUC, and for the most telling features their 10th / 50th / 90th
percentiles in the two sets. No frame, position or time of the match is printed, and no
possession label of the match is involved at any point.

    python scripts/adversarial.py --data-dir /path/to/data --data-tag v5
"""

import argparse
import glob
import os

import lightgbm
import numpy as np

from vantage import competition, external
from vantage.features import build


def sample_rows(table, count, rng):
    if len(table) <= count:
        return table
    return table[rng.choice(len(table), size=count, replace=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pseudo", default=None,
                        help="use this external match in place of the recording (a baseline)")
    parser.add_argument("--data-tag", default="v5")
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--drop", nargs="*", default=[], help="features to leave out")
    parser.add_argument("--fresh", type=int, default=0,
                        help="simulate this many external matches now instead of using the cache")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    real, names = [], None
    if args.pseudo:
        # A baseline: how separable is one *external* match from the others? Any single match
        # has habits of its own, so this is the level the real recording should be compared to.
        from vantage.degrade import DegradeConfig
        for half in external.labelled_halves(args.pseudo):
            names, table = build(external.observed(half, rng, DegradeConfig()))
            real.append(table)
    else:
        for half in competition.load_halves(args.data_dir):
            names, table = build(half)
            real.append(table)
    real = sample_rows(np.concatenate(real), args.rows, rng)

    simulated = []
    if args.fresh:
        # Simulate now, with the code as it stands, instead of reading cached tables.
        from vantage.degrade import sample_config
        pool = [match for match in external.ALL_MATCHES if match != args.pseudo]
        picks = rng.permutation(len(pool))[:args.fresh]
        for match in [pool[i] for i in picks]:
            for half in external.labelled_halves(match):
                noisy = external.observed(half, rng, sample_config(rng))
                simulated.append(sample_rows(build(noisy)[1], args.rows // (2 * args.fresh), rng))
    else:
        paths = sorted(glob.glob(os.path.join(external.CACHE, "train_" + args.data_tag,
                                              "*_own.npz")))
        per_file = max(1, args.rows // len(paths))
        for path in paths:
            with np.load(path) as data:
                if [str(name) for name in data["names"]] != list(names):
                    raise SystemExit("feature names differ: rebuild the tables")
                simulated.append(sample_rows(data["table"], per_file, rng))
    simulated = np.concatenate(simulated)

    keep = [index for index, name in enumerate(names) if name not in set(args.drop)]
    kept_names = [names[index] for index in keep]
    table = np.concatenate([real[:, keep], simulated[:, keep]])
    target = np.concatenate([np.ones(len(real)), np.zeros(len(simulated))])
    order = rng.permutation(len(table))
    table, target = table[order], target[order]

    folds = np.arange(len(table)) % 4
    score = np.zeros(len(table))
    gain = np.zeros(len(kept_names))
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 200,
              "feature_fraction": 0.5, "verbosity": -1, "num_threads": 4, "seed": 0}
    for fold in range(4):
        train = folds != fold
        booster = lightgbm.train(params, lightgbm.Dataset(table[train], label=target[train],
                                                          feature_name=kept_names),
                                 num_boost_round=120)
        score[~train] = booster.predict(table[~train])
        gain += booster.feature_importance("gain")

    ranks = np.argsort(np.argsort(score))
    positives = target == 1
    auc = (ranks[positives].sum() - positives.sum() * (positives.sum() - 1) / 2.0) / (
        positives.sum() * (~positives).sum())
    print("rows: {} real, {} simulated; features: {}".format(len(real), len(simulated), len(kept_names)))
    print("AUC real vs simulated: {:.4f}   (0.5 = cannot tell them apart)".format(auc))

    def spread(values):
        values = values[np.isfinite(values)]
        if not len(values):
            return "all missing"
        return "{:>8.2f} {:>8.2f} {:>8.2f}".format(*np.percentile(values, (10, 50, 90)))

    print("\n{:<22} {:>6} {:>7} {:>7}   {:<27} {:<27}".format(
        "feature", "gain%", "nan r", "nan s", "real p10/50/90", "simulated p10/50/90"))
    for index in np.argsort(-gain)[:args.top]:
        column = keep[index]
        print("{:<22} {:>6.1f} {:>7.3f} {:>7.3f}   {:<27} {:<27}".format(
            kept_names[index], 100.0 * gain[index] / gain.sum(),
            float(np.mean(~np.isfinite(real[:, column]))),
            float(np.mean(~np.isfinite(simulated[:, column]))),
            spread(real[:, column]), spread(simulated[:, column])))


if __name__ == "__main__":
    main()
