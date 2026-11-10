"""Cross-validation on other matches: train on some, score whole held-out matches.

Every number this script prints comes from matches other than the competition match. It is the
only place model and decoding choices are made.

    python scripts/cv.py prepare                      # build and cache the training tables
    python scripts/cv.py fold --test J03WOH metrica1  # train on the rest, score these two
    python scripts/cv.py fold --test metrica3 --tag exp1 --rounds 300

Ten external matches are used: seven from the open DFL dataset and Metrica's three public sample
games (see vantage/dfl.py and vantage/metrica_sample.py for sources and licences).

Held-out matches are scored several ways, because the real match's noise level is not known:
`clean` (no simulated noise), `typical` (the published averages), `harsh`, and two random draws.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from vantage import evaluate, external, model
from vantage import decode as decoding
from vantage.degrade import DegradeConfig, sample_config
from vantage.external import ALL_MATCHES, ROOT, labelled_halves, observed, seed_for, source_of

EVAL_SETTINGS = {
    "clean": None,
    "typical": DegradeConfig(),
    "easy": DegradeConfig(live_loss_rate=0.2, long_miss=0.0, player_missing=0.03, blackout=0.0,
                          extra_share=0.03, false_rate=0.0),
    "harsh": DegradeConfig(dead_hidden=0.7, lose_shift=-1.0, lose_spread=1.5, find_lag=1.5,
                           long_miss=0.12, live_loss_rate=1.6, live_loss_median=10.0,
                           until_restart=0.5, false_rate=0.15, player_missing=0.1,
                           player_drift=0.35, blackout=0.5, extra_share=0.2),
    "random1": "random",
    "random2": "random",
}


def prepare(args):
    for match in (args.matches or ALL_MATCHES):
        started = time.time()
        external.prepare_match(match, args.data_tag, args.variants, args.stride)
        print("{} prepared in {:.0f}s".format(match, time.time() - started), flush=True)


def eval_probabilities(booster, names, match, setting_name, setting):
    """[(labels, probabilities)] per half for one held-out match under one noise setting."""
    out = []
    for half in labelled_halves(match):
        rng = np.random.default_rng(seed_for("eval", match, half.period, setting_name))
        config = sample_config(rng) if setting == "random" else setting
        noisy = observed(half, rng, config)
        out.append((half.labels, model.predict_half(booster, noisy, names)))
    return out


def decode_grid():
    grid = []
    for cost_team in (2.0, 4.0, 8.0, 14.0):
        for cost_to_dead in (4.0, 8.0, 14.0, 22.0):
            for cost_from_dead in (4.0, 8.0, 14.0, 22.0):
                for min_possession in (1.0, 1.5, 2.5):
                    for min_dead in (0.5, 1.5, 3.0):
                        grid.append(decoding.DecodeConfig(
                            cost_team=cost_team, cost_to_dead=cost_to_dead,
                            cost_from_dead=cost_from_dead, min_possession=min_possession,
                            min_dead=min_dead))
    return grid


def grid_scores(halves, grid):
    """Score every decoding setting on one match. Viterbi runs once per distinct cost triple."""
    truth = [labels for labels, _ in halves]
    log_probs = [np.log(np.clip(prob, grid[0].floor, 1.0)) for _, prob in halves]
    triples = sorted({(c.cost_team, c.cost_to_dead, c.cost_from_dead) for c in grid})
    position = {triple: index for index, triple in enumerate(triples)}
    unique = [decoding.DecodeConfig(cost_team=a, cost_to_dead=b, cost_from_dead=c)
              for a, b, c in triples]
    paths = [decoding.viterbi_many(log_prob, unique) for log_prob in log_probs]
    scores = []
    for config in grid:
        index = position[(config.cost_team, config.cost_to_dead, config.cost_from_dead)]
        predicted = [decoding.enforce_minimums(path[index], config) for path in paths]
        scores.append(evaluate.quick(truth, predicted))
    return scores


def fold(args):
    test = list(args.test)
    train = [match for match in ALL_MATCHES if match not in test]
    if args.train_sources:
        train = [match for match in train if source_of(match) in args.train_sources]
    names, table, target = external.load_training(args.data_tag, train, args.variants)
    print("train {} rows x {} features from {}".format(len(table), len(names), train))
    started = time.time()
    booster = model.fit(table, target, names, seed=args.seed, rounds=args.rounds)
    print("fit in {:.0f}s".format(time.time() - started))
    sys.stdout.flush()
    del table

    out_dir = os.path.join(ROOT, "runs", "cv", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "model_{}.txt".format("-".join(test))))

    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)[:25]
    print("top features: " + ", ".join("{}({:.0f}%)".format(names[i], 100 * gain[i] / gain.sum())
                                       for i in order))

    grid = decode_grid()
    results = {}
    for setting_name, setting in EVAL_SETTINGS.items():
        for match in test:
            halves = eval_probabilities(booster, names, match, setting_name, setting)
            np.savez_compressed(
                os.path.join(out_dir, "prob_{}_{}.npz".format(match, setting_name)),
                **{"p{}".format(i): prob for i, (_, prob) in enumerate(halves)},
                **{"y{}".format(i): labels for i, (labels, _) in enumerate(halves)})
            truth = [labels for labels, _ in halves]
            frame_best = [np.argmax(prob, axis=1).astype(np.int8) for _, prob in halves]
            raw = evaluate.quick(truth, frame_best)
            scores = grid_scores(halves, grid)
            results[(setting_name, match)] = {"raw": raw, "grid": scores}
            best = int(np.argmax([s["spellF1"] for s in scores]))
            print("{:<8} {:<9} argmax spellF1 {:.3f} macroF1 {:.3f} | best decode spellF1 {:.3f} "
                  "macroF1 {:.3f} n {}/{} [{}]".format(
                      setting_name, match, raw["spellF1"], raw["macroF1"],
                      scores[best]["spellF1"], scores[best]["macroF1"], scores[best]["n_pred"],
                      scores[best]["n_true"], best))
            sys.stdout.flush()

    # One decoding setting for everything: the one with the best mean over matches and noise.
    mean = np.mean([[s["spellF1"] for s in value["grid"]] for value in results.values()], axis=0)
    winner = int(np.argmax(mean))
    print("\nbest single decode setting #{}: {}".format(winner, grid[winner]))
    for (setting_name, match), value in results.items():
        s = value["grid"][winner]
        print("  {:<8} {:<9} spellF1 {:.3f} macroF1 {:.3f} poss {}/{}".format(
            setting_name, match, s["spellF1"], s["macroF1"], s["n_pred"], s["n_true"]))
    print("  mean spellF1 over settings and matches: {:.4f}".format(float(mean[winner])))

    with open(os.path.join(out_dir, "summary_{}.json".format("-".join(test))), "w") as handle:
        json.dump({"test": test, "train": train, "winner": winner,
                   "mean_by_config": mean.tolist(),
                   "results": {"{}|{}".format(*key): value for key, value in results.items()}},
                  handle, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fold"):
        p = sub.add_parser(name)
        p.add_argument("--data-tag", default="v3")
        p.add_argument("--variants", type=int, default=3)
        p.add_argument("--stride", type=int, default=6)
        if name == "prepare":
            p.add_argument("--matches", nargs="*", default=None)
        if name == "fold":
            p.add_argument("--test", nargs="+", required=True)
            p.add_argument("--train-sources", nargs="*", default=None,
                           help="restrict training to these sources (dfl, metrica)")
            p.add_argument("--tag", default="dev")
            p.add_argument("--rounds", type=int, default=model.ROUNDS)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"prepare": prepare, "fold": fold}[args.command](args)


if __name__ == "__main__":
    main()
