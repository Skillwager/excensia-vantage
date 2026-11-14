"""Cross-validation on other matches: train on some, score whole held-out ones.

Every number this script prints comes from matches other than the competition match. It is the
only place model and decoding choices are made.

    python scripts/cv.py prepare                               # build the training tables
    python scripts/cv.py fold --tag v8 --test J03WMX metrica1  # train on the rest, score these
    python scripts/cv.py summary --tag v8                      # one decoding setting for all folds
    python scripts/cv.py rescore --tag v8 --out-tag v8b        # after a change to the decoding grid

Ten external matches are used: seven from the open DFL dataset and Metrica's three public sample
games (vantage/dfl.py and vantage/metrica_sample.py give sources and licences). Held-out matches
are scored under several simulated recording conditions (vantage/external.py), because how the
real recording behaves when the ball is lost cannot be measured without labels.
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
from vantage.external import ALL_MATCHES, ROOT, labelled_halves, source_of

GRID_FIELDS = ("cost_team", "cost_to_dead", "cost_from_dead", "boundary_weight", "boundary_bias",
               "min_possession", "min_dead")


def decode_grid(with_boundary, wide=False):
    """The decoding settings that are compared. A stoppage costs the same to enter and to leave.

    `wide` is a second grid that starts where the first one ends: the setting the first one
    chooses is at its edge in four numbers, and this looks beyond that edge.
    """
    grid = []
    if wide:
        for cost_team in (2.0, 4.0, 8.0):
            for cost_dead in (14.0, 20.0, 28.0):
                for weight in (0.0, 8.0, 12.0, 16.0, 24.0):
                    for bias in ((0.0, -0.5, -1.0) if weight else (0.0,)):
                        for min_possession in (0.4, 0.8, 1.2):
                            grid.append(decoding.DecodeConfig(
                                cost_team=cost_team, cost_to_dead=cost_dead,
                                cost_from_dead=cost_dead, boundary_weight=weight,
                                boundary_bias=bias, min_possession=min_possession, min_dead=0.5))
        return grid
    if not with_boundary:
        for cost_team in (0.5, 2.0, 6.0, 12.0):
            for cost_dead in (10.0, 14.0, 20.0):
                for min_possession in (0.2, 0.4, 0.8):
                    grid.append(decoding.DecodeConfig(
                        cost_team=cost_team, cost_to_dead=cost_dead, cost_from_dead=cost_dead,
                        min_possession=min_possession, min_dead=0.5))
        return grid
    for cost_team in (0.5, 2.0):
        for cost_dead in (10.0, 14.0):
            for weight in (0.0, 4.0, 8.0, 12.0):
                # a bias only matters where the boundary model is used
                for bias in ((0.0, -0.5, -1.0, -1.5, -2.0) if weight else (0.0,)):
                    for min_possession in (0.2, 0.4, 0.8):
                        grid.append(decoding.DecodeConfig(
                            cost_team=cost_team, cost_to_dead=cost_dead,
                            cost_from_dead=cost_dead, boundary_weight=weight,
                            boundary_bias=bias, min_possession=min_possession, min_dead=0.5))
    return grid


def grid_scores(halves, grid):
    """Score every decoding setting on one match. Viterbi runs once per distinct cost setting."""
    truth = [labels for labels, _, _ in halves]
    log_probs = [np.log(np.clip(prob, grid[0].floor, 1.0)) for _, prob, _ in halves]
    odds = [decoding.boundary_log_odds(boundary) for _, _, boundary in halves]

    def key(c):
        return (c.cost_team, c.cost_to_dead, c.cost_from_dead, c.boundary_weight, c.boundary_bias)

    unique_keys = sorted({key(c) for c in grid})
    position = {k: index for index, k in enumerate(unique_keys)}
    unique = [decoding.DecodeConfig(cost_team=a, cost_to_dead=b, cost_from_dead=c,
                                    boundary_weight=w, boundary_bias=s)
              for a, b, c, w, s in unique_keys]
    paths = [decoding.viterbi_many(log_prob, unique, odd) for log_prob, odd in zip(log_probs, odds)]
    scores = []
    for config in grid:
        index = position[key(config)]
        predicted = [decoding.enforce_minimums(path[index], config) for path in paths]
        scores.append(evaluate.quick(truth, predicted))
    return scores


def prepare(args):
    for match in (args.matches or ALL_MATCHES):
        started = time.time()
        external.prepare_match(match, args.data_tag, args.variants, args.stride,
                               extended=not args.first_features)
        print("{} prepared in {:.0f}s".format(match, time.time() - started), flush=True)


def eval_match(state_model, boundary_model, names, match, condition):
    """[(labels, state probabilities, boundary probabilities)] per half."""
    out = []
    for half in labelled_halves(match):
        noisy = external.condition_half(half, match, condition)
        state, boundary = model.predict_half_both(state_model, boundary_model, noisy, names)
        out.append((half.labels, state, boundary))
    return out


def fold(args):
    test = list(args.test)
    train = [match for match in ALL_MATCHES if match not in test]
    if args.train_sources:
        train = [match for match in train if source_of(match) in args.train_sources]
    if args.train_every > 1:
        # fewer matches, to see how much their number matters: every Nth, starting with the
        # first or (--train-offset) a later one, so that the halves can be compared
        train = train[args.train_offset::args.train_every]
    training = external.load_training(args.data_tag, train, args.variants)
    own_rows = len(training["target"])
    weight = external.row_weights(training, args.metrica_weight, args.short_weight, args.short_seconds)
    boundary_weight = external.row_weights(training, args.metrica_weight)
    if args.skillcorner:
        # SkillCorner's matches join the frame classifier's rows as they were recorded (one
        # copy each: nothing is simulated on them). The boundary model is fitted without them,
        # because their stoppages begin about half a second early (scripts/skillcorner_dead.py).
        added = list(external.SKILLCORNER_MATCHES[:args.skillcorner])
        extra = external.load_training(args.data_tag, added, 1)
        if extra["names"] != training["names"]:
            raise SystemExit("the SkillCorner tables were built with other features")
        extra_weight = external.row_weights(extra, 1.0, args.short_weight, args.short_seconds)
        if extra_weight is None:
            extra_weight = np.ones(len(extra["target"]), dtype=np.float32)
        if weight is None:
            weight = np.ones(own_rows, dtype=np.float32)
        weight = np.concatenate([weight, extra_weight * np.float32(args.skillcorner_weight)])
        for key in ("table", "target", "run", "metrica"):
            training[key] = np.concatenate([training[key], extra[key]])
        print("added {} rows from {} SkillCorner matches at weight {}".format(
            len(extra["target"]), len(added), args.skillcorner_weight), flush=True)
        del extra
    names, table = training["names"], training["table"]
    print("train {} rows x {} features from {}".format(len(table), len(names), train), flush=True)
    made_by, recorded_by = training["generation"], external.generation()
    print("tables made by simulator {}, held-out matches recorded by simulator {}".format(
        made_by, recorded_by), flush=True)
    started = time.time()
    state_model = model.fit(
        table, training["target"], names, seed=args.seed, rounds=args.rounds,
        params=json.loads(args.state_params) or None, weight=weight)
    boundary_model = None
    if not args.no_boundary:
        boundary_model = model.fit_boundary(
            table[:own_rows], training["change"], names, seed=args.seed, rounds=args.boundary_rounds,
            weight=boundary_weight)
    print("fit in {:.0f}s".format(time.time() - started), flush=True)
    del table, training

    out_dir = os.path.join(ROOT, "runs", "cv", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    name = "-".join(test)

    grid = decode_grid(boundary_model is not None)
    results = {}
    for setting_name in external.CONDITIONS:
        for match in test:
            halves = eval_match(state_model, boundary_model, names, match, setting_name)
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
        json.dump({"test": test, "train": train, "generation": recorded_by, "tables": made_by,
                   "grid": [{field: getattr(config, field) for field in GRID_FIELDS}
                            for config in grid],
                   "results": results}, handle)


def recorded_with(tag):
    """The simulator generation a run's held-out matches were recorded with, where the run says
    so: a fold's summary does, and a rescored or averaged run passes on what its source said.
    None where nothing says or the parts disagree (compare_cv.py reads that as the first)."""
    folder = os.path.join(ROOT, "runs", "cv", tag)
    said = set()
    for path in (glob.glob(os.path.join(folder, "summary_*.json"))
                 + glob.glob(os.path.join(folder, "averaged.json"))):
        with open(path) as handle:
            said.add(json.load(handle).get("generation"))
    said.discard(None)
    return said.pop() if len(said) == 1 else None


def rescore(args):
    """Score the saved predictions of a run again, with the decoding grid as it is now.

    A fold saves the probabilities it predicted for each held-out match and condition, so a
    change to the grid needs no retraining. The new scores go to their own tag.
    """
    source_dir = os.path.join(ROOT, "runs", "cv", args.tag)
    out_dir = os.path.join(ROOT, "runs", "cv", args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    generation = recorded_with(args.tag)
    found = sorted(glob.glob(os.path.join(source_dir, "prob_*.npz")))
    matches = sorted({os.path.basename(path)[5:-4].rsplit("_", 1)[0] for path in found})
    for match in matches:
        if args.matches and match not in args.matches:
            continue
        results, grid = {}, None
        for condition in external.CONDITIONS:
            path = os.path.join(source_dir, "prob_{}_{}.npz".format(match, condition))
            if not os.path.isfile(path):
                continue
            with np.load(path) as data:
                count = sum(1 for key in data.files if key.startswith("y"))
                halves = [(data["y{}".format(i)], data["p{}".format(i)].astype(np.float64),
                           data["b{}".format(i)].astype(np.float64)
                           if "b{}".format(i) in data.files else None) for i in range(count)]
            if args.relabel:
                # the labels as they are defined now, not as they were when the fold was run
                truth = [half.labels for half in labelled_halves(match)]
                halves = [(labels, state, boundary)
                          for labels, (_, state, boundary) in zip(truth, halves)]
            grid = decode_grid(halves[0][2] is not None, wide=getattr(args, "wide", False))
            scores = grid_scores(halves, grid)
            results["{}|{}".format(condition, match)] = scores
            best = int(np.argmax([s["spellF1"] for s in scores]))
            print("{:<8} {:<9} best spellF1 {:.3f} macroF1 {:.3f} poss {}/{}".format(
                condition, match, scores[best]["spellF1"], scores[best]["macroF1"],
                scores[best]["n_pred"], scores[best]["n_true"]), flush=True)
        if results:
            record = {"test": [match], "rescored_from": args.tag,
                      "grid": [{field: getattr(config, field) for field in GRID_FIELDS}
                               for config in grid],
                      "results": results}
            if generation:
                record["generation"] = generation
            with open(os.path.join(out_dir, "summary_{}.json".format(match)), "w") as handle:
                json.dump(record, handle)


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

    keys = [key for key in rows if key.split("|")[0] not in args.skip
            and (not args.sources or source_of(key.split("|")[1]) in args.sources)]
    spell = np.array([[s["spellF1"] for s in rows[key]] for key in keys])
    macro = np.array([[s["macroF1"] for s in rows[key]] for key in keys])
    mean = spell.mean(axis=0)
    order = np.argsort(-mean)
    print("{} match x condition cells, {} decoding settings".format(len(keys), len(grid)))
    for index in order[:8]:
        print("#{:<4} mean spellF1 {:.4f} macroF1 {:.4f} | {}".format(
            index, mean[index], macro[:, index].mean(),
            " ".join("{}={}".format(field, grid[index][field]) for field in GRID_FIELDS)))
    for field in ("boundary_weight", "boundary_bias", "min_possession"):
        best = {}
        for index, config in enumerate(grid):
            value = config.get(field, 0.0)
            best[value] = max(best.get(value, 0.0), float(mean[index]))
        print("best mean spellF1 by {}: ".format(field) + "  ".join(
            "{}: {:.4f}".format(value, score) for value, score in sorted(best.items())))

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

    # The same setting on every match, whichever matches chose it: one row per source.
    for metric in ("spellF1", "macroF1"):
        print("\nmean {} by source and condition, setting #{} (mean* leaves out: {})".format(
            metric, winner, ", ".join(args.skip) or "nothing"))
        print("{:<13}".format("") + "".join("{:>9}".format(name) for name in settings)
              + "{:>9}".format("mean*"))
        for source in ("metrica", "dfl", None):
            chosen = [match for match in matches if source is None or source_of(match) == source]
            if not chosen:
                continue
            cells, kept = [], []
            for name in settings:
                values = [rows["{}|{}".format(name, match)][winner][metric] for match in chosen
                          if "{}|{}".format(name, match) in rows]
                cells.append("{:>9.3f}".format(np.mean(values)) if values else "{:>9}".format("-"))
                if name not in args.skip:
                    kept += values
            print("{:<13}".format("{} ({})".format(source or "all", len(chosen))) + "".join(cells)
                  + "{:>9.3f}".format(np.mean(kept)))
    print("\ndecode json: " + json.dumps(grid[winner]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-tag", default="g7")
    p.add_argument("--variants", type=int, default=6)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--matches", nargs="*", default=None)
    p.add_argument("--first-features", action="store_true",
                   help="tables with the first 255 columns only, as the first models' tables had")
    p = sub.add_parser("fold")
    p.add_argument("--data-tag", default="g7")
    p.add_argument("--variants", type=int, default=6)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--train-sources", nargs="*", default=None)
    p.add_argument("--skillcorner", type=int, default=0,
                   help="add the first so many SkillCorner matches to the frame classifier's rows")
    p.add_argument("--skillcorner-weight", type=float, default=1.0)
    p.add_argument("--train-every", type=int, default=1,
                   help="train on every Nth of the fold's training matches only")
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--tag", default="dev")
    p.add_argument("--rounds", type=int, default=model.ROUNDS)
    p.add_argument("--boundary-rounds", type=int, default=300)
    p.add_argument("--no-boundary", action="store_true")
    p.add_argument("--metrica-weight", type=float, default=1.0,
                   help="weight of rows from Metrica's sample games relative to DFL rows")
    p.add_argument("--short-weight", type=float, default=4.0,
                   help="largest weight given to the rows of a short possession (1 = none)")
    p.add_argument("--short-seconds", type=float, default=8.0,
                   help="possessions at least this long keep weight one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-params", default="{}",
                   help="JSON of LightGBM settings that replace the frame classifier's own")
    p = sub.add_parser("rescore")
    p.add_argument("--tag", required=True, help="the run whose saved predictions are scored")
    p.add_argument("--out-tag", required=True, help="where the new scores are written")
    p.add_argument("--relabel", action="store_true",
                   help="score against the labels as defined now, not the ones saved with the fold")
    p.add_argument("--wide", action="store_true",
                   help="score the wider grid that starts where the ordinary one ends")
    p.add_argument("--matches", nargs="*", default=None)
    p = sub.add_parser("summary")
    p.add_argument("--tag", required=True)
    p.add_argument("--sources", nargs="*", default=None,
                   help="choose the decoding setting on these sources only (dfl, metrica)")
    p.add_argument("--skip", nargs="*", default=["clean"],
                   help="conditions left out of the choice of decoding setting")
    args = parser.parse_args()
    {"prepare": prepare, "fold": fold, "rescore": rescore, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
