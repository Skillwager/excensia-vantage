"""Cross-validation on other matches, with the boundary model: train on some, score held-out ones.

Every number this script prints comes from matches other than the competition match. It is the
only place model and decoding choices are made.

    python scripts/cv.py prepare --data-tag v4                 # build the training tables
    python scripts/cvb.py fold --tag v4 --test J03WMX metrica1 # train on the rest, score these
    python scripts/cvb.py summary --tag v4                     # one decoding setting for all folds

Held-out matches are scored under several simulated recording conditions, because how the real
recording behaves when the ball is lost cannot be measured without labels.
"""

import argparse
import glob
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
GRID_FIELDS = ("cost_team", "cost_to_dead", "cost_from_dead", "boundary_weight",
               "min_possession", "min_dead")


def decode_grid(with_boundary):
    grid = []
    for cost_team in (2.0, 4.0, 8.0, 14.0):
        for cost_to_dead in (4.0, 8.0, 14.0):
            for cost_from_dead in (4.0, 8.0, 14.0):
                for weight in ((0.0, 0.5, 1.0, 2.0) if with_boundary else (0.0,)):
                    for min_possession in (1.0, 1.5, 2.5):
                        for min_dead in (0.5, 1.5, 3.0):
                            grid.append(decoding.DecodeConfig(
                                cost_team=cost_team, cost_to_dead=cost_to_dead,
                                cost_from_dead=cost_from_dead, boundary_weight=weight,
                                min_possession=min_possession, min_dead=min_dead))
    return grid


def grid_scores(halves, grid):
    """Score every decoding setting on one match. Viterbi runs once per distinct cost setting."""
    truth = [labels for labels, _, _ in halves]
    log_probs = [np.log(np.clip(prob, grid[0].floor, 1.0)) for _, prob, _ in halves]
    odds = [decoding.boundary_log_odds(boundary) for _, _, boundary in halves]

    def key(c):
        return (c.cost_team, c.cost_to_dead, c.cost_from_dead, c.boundary_weight)

    unique_keys = sorted({key(c) for c in grid})
    position = {k: index for index, k in enumerate(unique_keys)}
    unique = [decoding.DecodeConfig(cost_team=a, cost_to_dead=b, cost_from_dead=c,
                                    boundary_weight=w) for a, b, c, w in unique_keys]
    paths = [decoding.viterbi_many(log_prob, unique, odd) for log_prob, odd in zip(log_probs, odds)]
    scores = []
    for config in grid:
        index = position[key(config)]
        predicted = [decoding.enforce_minimums(path[index], config) for path in paths]
        scores.append(evaluate.quick(truth, predicted))
    return scores


def eval_match(state_model, boundary_model, names, match, setting_name, setting):
    """[(labels, state probabilities, boundary probabilities)] per half."""
    out = []
    for half in labelled_halves(match):
        rng = np.random.default_rng(seed_for("eval", match, half.period, setting_name))
        config = sample_config(rng) if setting == "random" else setting
        noisy = observed(half, rng, config)
        state, boundary = model.predict_half_both(state_model, boundary_model, noisy, names)
        out.append((half.labels, state, boundary))
    return out


def fold(args):
    test = list(args.test)
    train = [match for match in ALL_MATCHES if match not in test]
    if args.train_sources:
        train = [match for match in train if source_of(match) in args.train_sources]
    names, table, target, change = external.load_training(args.data_tag, train, args.variants,
                                                          with_change=True)
    print("train {} rows x {} features from {}".format(len(table), len(names), train), flush=True)
    started = time.time()
    state_model = model.fit(table, target, names, seed=args.seed, rounds=args.rounds)
    boundary_model = None
    if not args.no_boundary:
        boundary_model = model.fit_boundary(table, change, names, seed=args.seed,
                                            rounds=args.boundary_rounds)
    print("fit in {:.0f}s".format(time.time() - started), flush=True)
    del table

    out_dir = os.path.join(ROOT, "runs", "cv", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    name = "-".join(test)

    grid = decode_grid(boundary_model is not None)
    results = {}
    for setting_name, setting in EVAL_SETTINGS.items():
        for match in test:
            halves = eval_match(state_model, boundary_model, names, match, setting_name, setting)
            payload = {}
            for index, (labels, state, boundary) in enumerate(halves):
                payload["y{}".format(index)] = labels
                payload["p{}".format(index)] = state.astype(np.float32)
                if boundary is not None:
                    payload["b{}".format(index)] = boundary.astype(np.float32)
            np.savez_compressed(os.path.join(out_dir, "prob_{}_{}.npz".format(match, setting_name)),
                                **payload)
            scores = grid_scores(halves, grid)
            results["{}|{}".format(setting_name, match)] = scores
            best = int(np.argmax([s["spellF1"] for s in scores]))
            print("{:<8} {:<9} best spellF1 {:.3f} macroF1 {:.3f} poss {}/{}".format(
                setting_name, match, scores[best]["spellF1"], scores[best]["macroF1"],
                scores[best]["n_pred"], scores[best]["n_true"]), flush=True)

    with open(os.path.join(out_dir, "summary_{}.json".format(name)), "w") as handle:
        json.dump({"test": test, "train": train,
                   "grid": [{field: getattr(config, field) for field in GRID_FIELDS}
                            for config in grid],
                   "results": results}, handle)


def summary(args):
    rows, grid = {}, None
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "cv", args.tag, "summary_*.json"))):
        with open(path) as handle:
            data = json.load(handle)
        if grid is None:
            grid = data["grid"]
        elif grid != data["grid"]:
            raise SystemExit("folds were run with different decoding grids")
        rows.update(data["results"])
    if not rows:
        raise SystemExit("no folds found for tag " + args.tag)

    keys = [key for key in rows if key.split("|")[0] not in args.skip]
    spell = np.array([[s["spellF1"] for s in rows[key]] for key in keys])
    macro = np.array([[s["macroF1"] for s in rows[key]] for key in keys])
    mean = spell.mean(axis=0)
    order = np.argsort(-mean)
    print("{} match x condition cells, {} decoding settings".format(len(keys), len(grid)))
    for index in order[:8]:
        print("#{:<4} mean spellF1 {:.4f} macroF1 {:.4f} | {}".format(
            index, mean[index], macro[:, index].mean(),
            " ".join("{}={}".format(field, grid[index][field]) for field in GRID_FIELDS)))
    by_weight = {}
    for index, config in enumerate(grid):
        weight = config["boundary_weight"]
        by_weight[weight] = max(by_weight.get(weight, 0.0), float(mean[index]))
    print("best mean spellF1 by boundary weight: " + "  ".join(
        "{}: {:.4f}".format(weight, value) for weight, value in sorted(by_weight.items())))

    winner = int(order[0])
    settings = sorted({key.split("|")[0] for key in rows})
    matches = sorted({key.split("|")[1] for key in rows})
    print("\nwith setting #{}:".format(winner))
    print("{:<10}".format("") + "".join("{:>9}".format(name) for name in settings) + "{:>10}".format("poss"))
    for match in matches:
        cells = []
        for name in settings:
            value = rows.get("{}|{}".format(name, match))
            cells.append("{:>9.3f}".format(value[winner]["spellF1"]) if value else "{:>9}".format("-"))
        typical = rows.get("typical|{}".format(match))
        poss = "{}/{}".format(typical[winner]["n_pred"], typical[winner]["n_true"]) if typical else ""
        print("{:<10}".format(match) + "".join(cells) + "{:>10}".format(poss))
    for source in ("dfl", "metrica"):
        chosen = [key for key in keys if source_of(key.split("|")[1]) == source]
        if chosen:
            print("mean over {} cells ({}): {:.4f}".format(
                len(chosen), source, float(np.mean([rows[key][winner]["spellF1"] for key in chosen]))))
    print("decode json: " + json.dumps(grid[winner]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fold")
    p.add_argument("--data-tag", default="v4")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--train-sources", nargs="*", default=None)
    p.add_argument("--tag", default="dev")
    p.add_argument("--rounds", type=int, default=model.ROUNDS)
    p.add_argument("--boundary-rounds", type=int, default=300)
    p.add_argument("--no-boundary", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("summary")
    p.add_argument("--tag", required=True)
    p.add_argument("--skip", nargs="*", default=["clean"],
                   help="conditions left out of the choice of decoding setting")
    args = parser.parse_args()
    {"fold": fold, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
