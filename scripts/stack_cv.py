"""A second model on top of the first, tried by cross-validation on the external matches.

The frame classifier looks at one frame's features. This second model looks at what the frame
classifier and the boundary model said over the seconds either side of a frame, and gives the
three states again. What it learns from are the predictions each external match received while
it was held out (runs/cv/<tag>/prob_*.npz), so nothing it sees was fitted to the match it is
looking at. It is tested fold by fold in the same way: trained on the eight matches outside a
fold, applied to the two inside it, and its output written as a new set of prediction files
that `cv.py rescore` and `cv.py summary` score like any other.

    python scripts/stack_cv.py --tag v10w --out-tag v13s
    python scripts/cv.py rescore --tag v13s --out-tag v13sb
    python scripts/cv.py summary --tag v13sb --sources metrica

No competition data is read.
"""

import argparse
import glob
import os

import numpy as np

from vantage.external import CONDITIONS, ROOT

FOLDS = (("J03WMX", "metrica1"), ("J03WN1", "metrica2"), ("J03WOH", "metrica3"),
         ("J03WOY", "J03WPY"), ("J03WQQ", "J03WR9"))
# frames either side of now at which the first models' answers are read
OFFSETS = (-75, -50, -37, -25, -18, -12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12, 18, 25, 37, 50, 75)
WINDOWS = (6, 12, 25, 62)       # half-widths of the running means, in frames


def shifted(values, offset):
    """values (T, C) moved so that row t holds what was at t + offset; NaN past either end."""
    out = np.full_like(values, np.nan)
    if offset == 0:
        out[:] = values
    elif offset > 0:
        out[:-offset] = values[offset:]
    else:
        out[-offset:] = values[:offset]
    return out


def running_mean(values, half):
    """Mean over [t - half, t + half], shortened at the ends."""
    total = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), values]), axis=0)
    t = np.arange(len(values))
    lo = np.maximum(t - half, 0)
    hi = np.minimum(t + half + 1, len(values))
    return ((total[hi] - total[lo]) / (hi - lo)[:, None]).astype(np.float32)


def table_of(state, boundary):
    """The second model's inputs for one half: (T, F) float32."""
    channels = np.hstack([state, boundary]).astype(np.float32)        # 3 + 4 columns
    parts = [shifted(channels, offset) for offset in OFFSETS]
    parts += [running_mean(channels, half) for half in WINDOWS]
    return np.hstack(parts)


def swap_teams(table, state_columns=3, block=7):
    """The same inputs with home and away exchanged (columns 0 and 1 of every block of seven)."""
    out = table.copy()
    for start in range(0, table.shape[1], block):
        out[:, start], out[:, start + 1] = table[:, start + 1], table[:, start]
    return out


def load(tag, match, condition):
    path = os.path.join(ROOT, "runs", "cv", tag, "prob_{}_{}.npz".format(match, condition))
    halves = []
    with np.load(path) as data:
        index = 0
        while "p{}".format(index) in data:
            halves.append((data["y{}".format(index)], data["p{}".format(index)].astype(np.float32),
                           data["b{}".format(index)].astype(np.float32)))
            index += 1
    return halves


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the run whose held-out predictions are read")
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--stride", type=int, default=4, help="train on one frame in this many")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    import lightgbm

    out_dir = os.path.join(ROOT, "runs", "cv", args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    have = glob.glob(os.path.join(ROOT, "runs", "cv", args.tag, "prob_*.npz"))
    if len(have) != len(FOLDS) * 2 * len(CONDITIONS):
        raise SystemExit("{} prediction files under {}, expected {}".format(
            len(have), args.tag, len(FOLDS) * 2 * len(CONDITIONS)))

    for number, test in enumerate(FOLDS):
        if args.folds and number not in args.folds:
            continue
        tables, targets = [], []
        for fold in FOLDS:
            if fold == test:
                continue
            for match in fold:
                for condition in CONDITIONS:
                    for labels, state, boundary in load(args.tag, match, condition):
                        table = table_of(state, boundary)[::args.stride]
                        target = labels[::args.stride].astype(np.int64)
                        keep = target < 3
                        # both views of every half: the model cannot favour the side called home
                        tables += [table[keep], swap_teams(table[keep])]
                        swapped = target[keep].copy()
                        swapped[target[keep] == 0] = 1
                        swapped[target[keep] == 1] = 0
                        targets += [target[keep], swapped]
        table = np.vstack(tables)
        target = np.concatenate(targets)
        del tables, targets
        print("fold {}: {} rows x {} inputs".format(number, len(table), table.shape[1]), flush=True)
        booster = lightgbm.train(
            {"objective": "multiclass", "num_class": 3, "learning_rate": 0.06, "num_leaves": 31,
             "min_data_in_leaf": 400, "feature_fraction": 0.6, "bagging_fraction": 0.7,
             "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 127, "num_threads": args.threads,
             "deterministic": True, "force_row_wise": True, "seed": 0, "verbosity": -1},
            lightgbm.Dataset(table, label=target), num_boost_round=args.rounds)
        del table, target

        for match in test:
            for condition in CONDITIONS:
                payload = {}
                for index, (labels, state, boundary) in enumerate(load(args.tag, match, condition)):
                    inputs = table_of(state, boundary)
                    straight = booster.predict(inputs)
                    mirrored = booster.predict(swap_teams(inputs))[:, [1, 0, 2]]
                    payload["y{}".format(index)] = labels
                    payload["p{}".format(index)] = ((straight + mirrored) / 2).astype(np.float32)
                    payload["b{}".format(index)] = boundary
                np.savez_compressed(os.path.join(out_dir, "prob_{}_{}.npz".format(match, condition)),
                                    **payload)
        print("fold {} written".format(number), flush=True)


if __name__ == "__main__":
    main()
