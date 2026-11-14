"""Can a classifier tell the provided recording from the simulated training data?

A label-free check on the simulator. Feature rows from the competition match and rows from
simulated recordings of the open matches are given to a classifier that is asked which is which.
Where it can tell, the features it leans on are the ones that look different in the real
recording, and those are either a flaw in the simulator to fix or a feature the possession
model should not be given.

The classifier is fitted on one half of the match against one set of open matches and asked
about the other half against other open matches, then the other way round. A split by row
would measure nothing: the features change slowly (some are constant over a whole gap in the
ball's track), so rows a few frames apart are near copies, and a classifier that has seen one
knows the other. Split that way, any open match is told from the others without a single
error.

Read the number against a baseline, not against 0.5. Any one match has habits of its own (its
pitch, the two teams' shapes), so an open match set in the recording's place (--pseudo) is told
from the others too; the recording should not be told apart much better than that. --only
keeps the check to one family of features, for example the ball's own (b_ gap_ tch_).

Prints totals only: the two AUCs, and for the most telling features their 10th / 50th / 90th
percentiles in the two sets. No frame, position or time of the match is printed, and no
possession label of the match is involved at any point.

    python scripts/adversarial.py --data-dir /path/to/data --fresh 4
    python scripts/adversarial.py --pseudo metrica1 --fresh 4          # the baseline
    python scripts/adversarial.py --data-dir /path/to/data --fresh 4 \
        --families edges=ed_ corners=rb_,tcr_,rq_ shape=h_,a_,t_,sp_     # one AUC for each family
"""

import argparse
import glob
import os

import lightgbm
import numpy as np

from vantage import competition, epts, external, tagging
from vantage.features import build


def sample_rows(table, count, rng):
    if len(table) <= count:
        return table
    return table[rng.choice(len(table), size=count, replace=False)]


def area_under_curve(score, positive):
    ranks = np.argsort(np.argsort(score))
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() - 1) / 2.0)
                 / (positive.sum() * (~positive).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pseudo", default=None,
                        help="use this external match in place of the recording (a baseline)")
    parser.add_argument("--data-tag", default="g7")
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--drop", nargs="*", default=[], help="features to leave out")
    parser.add_argument("--only", nargs="*", default=None,
                        help="prefixes of the features to keep (default: all)")
    parser.add_argument("--families", nargs="*", default=None,
                        help="name=prefix,prefix ...: an AUC for each family of features, from the "
                             "same recordings (the table of features is not printed)")
    parser.add_argument("--fresh", type=int, default=0,
                        help="simulate this many external matches for each of the two sets now, "
                             "instead of using the cache")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    # -- the recording (or the open match standing in for it), one table for each half ----------
    real, names = [], None
    if args.pseudo:
        typical = external.conditions()["typical"]
        for half in external.labelled_halves(args.pseudo):
            # with imitated set-piece tags, as the training tables have them
            names, table = build(external.with_tags(external.observed(half, rng, typical), rng))
            real.append(sample_rows(table, args.rows // 2, rng))
    else:
        # the tagged set pieces are attached as the run command attaches them
        paths = competition.tactical_paths(args.data_dir)
        home_name, away_name = epts.read_team_names(paths["meta"])
        tagged = tagging.restarts(tagging.load(paths["tagging"]), home_name, away_name)
        for half in competition.load_halves(args.data_dir):
            half.meta["restarts"] = [(frame - half.first_frame, team) for frame, team in tagged
                                     if 0 <= frame - half.first_frame < half.n_frames]
            names, table = build(half)
            real.append(sample_rows(table, args.rows // 2, rng))
    if len(real) != 2:
        raise SystemExit("two halves are needed, one to fit on and one to ask about")

    # -- simulated recordings of the open matches, in two sets that share no match ---------------
    simulated = [[], []]
    if args.fresh:
        pool = [match for match in external.ALL_MATCHES if match != args.pseudo]
        if 2 * args.fresh > len(pool):
            raise SystemExit("--fresh {} needs {} open matches and there are {}".format(
                args.fresh, 2 * args.fresh, len(pool)))
        picks = [pool[i] for i in rng.permutation(len(pool))[:2 * args.fresh]]
        for position, match in enumerate(picks):
            for half in external.labelled_halves(match):
                noisy = external.with_tags(external.observed(half, rng, external.sampled(rng)), rng)
                simulated[position % 2].append(sample_rows(build(noisy)[1], args.rows // (4 * args.fresh), rng))
        made_by = external.generation()
    else:
        paths = sorted(glob.glob(os.path.join(external.CACHE, "train_" + args.data_tag, "*_own.npz")))
        matches = sorted({os.path.basename(path).split("_p")[0] for path in paths} - {args.pseudo})
        per_file = max(1, args.rows // max(len(paths), 1))
        made_by = set()
        for path in paths:
            match = os.path.basename(path).split("_p")[0]
            if match == args.pseudo:
                continue
            with np.load(path) as data:
                if [str(name) for name in data["names"]] != list(names):
                    raise SystemExit("feature names differ: rebuild the tables")
                made_by.add(str(data["generation"]) if "generation" in data.files else "g1")
                simulated[matches.index(match) % 2].append(sample_rows(data["table"], per_file, rng))
        made_by = "/".join(sorted(made_by)) + " (the tables of " + args.data_tag + ")"
    simulated = [np.concatenate(part) for part in simulated]

    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 200,
              "feature_fraction": 0.5, "verbosity": -1, "num_threads": 4, "seed": 0}
    if args.families:
        print("{}; simulated by {}; rows {} and {}".format(
            "open match " + args.pseudo if args.pseudo else "the recording", made_by,
            sum(len(part) for part in real), sum(len(part) for part in simulated)))
        for family in ["everything="] + list(args.families):
            label, _, prefixes = family.partition("=")
            prefixes = tuple(p for p in prefixes.split(",") if p)
            keep = [index for index, name in enumerate(names) if not prefixes or name.startswith(prefixes)]
            if not keep:
                print("   {:<12} no such feature".format(label))
                continue
            kept_names = [names[index] for index in keep]
            areas = []
            for fit_on, ask in ((0, 1), (1, 0)):
                table = np.concatenate([real[fit_on][:, keep], simulated[fit_on][:, keep]])
                target = np.concatenate([np.ones(len(real[fit_on])), np.zeros(len(simulated[fit_on]))])
                order = np.random.default_rng(1).permutation(len(table))
                booster = lightgbm.train(params, lightgbm.Dataset(table[order], label=target[order],
                                                                  feature_name=kept_names), num_boost_round=120)
                asked = np.concatenate([real[ask][:, keep], simulated[ask][:, keep]])
                positive = np.concatenate([np.ones(len(real[ask]), dtype=bool),
                                           np.zeros(len(simulated[ask]), dtype=bool)])
                areas.append(area_under_curve(booster.predict(asked), positive))
            print("   {:<12} {:3d} features: AUC {:.3f} and {:.3f}, mean {:.3f}".format(
                label, len(keep), areas[0], areas[1], float(np.mean(areas))))
        return
    keep = [index for index, name in enumerate(names) if name not in set(args.drop)
            and (args.only is None or any(name.startswith(prefix) for prefix in args.only))]
    kept_names = [names[index] for index in keep]
    gain = np.zeros(len(kept_names))
    areas = []
    for fit_on, ask in ((0, 1), (1, 0)):
        table = np.concatenate([real[fit_on][:, keep], simulated[fit_on][:, keep]])
        target = np.concatenate([np.ones(len(real[fit_on])), np.zeros(len(simulated[fit_on]))])
        order = rng.permutation(len(table))
        booster = lightgbm.train(params, lightgbm.Dataset(table[order], label=target[order],
                                                          feature_name=kept_names), num_boost_round=120)
        gain += booster.feature_importance("gain")
        asked = np.concatenate([real[ask][:, keep], simulated[ask][:, keep]])
        positive = np.concatenate([np.ones(len(real[ask]), dtype=bool), np.zeros(len(simulated[ask]), dtype=bool)])
        areas.append(area_under_curve(booster.predict(asked), positive))

    real_rows, simulated_rows = np.concatenate(real), np.concatenate(simulated)
    print("{}: {} rows; simulated by {}: {} rows; features: {}".format(
        "open match " + args.pseudo if args.pseudo else "the recording", len(real_rows),
        made_by, len(simulated_rows), len(kept_names)))
    print("AUC, fitted on one half and asked about the other against other open matches: {:.3f} and "
          "{:.3f}, mean {:.3f}".format(areas[0], areas[1], float(np.mean(areas))))

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
            float(np.mean(~np.isfinite(real_rows[:, column]))),
            float(np.mean(~np.isfinite(simulated_rows[:, column]))),
            spread(real_rows[:, column]), spread(simulated_rows[:, column])))


if __name__ == "__main__":
    main()
