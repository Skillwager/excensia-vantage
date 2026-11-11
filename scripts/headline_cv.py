"""The official score's parts, match by match, for one saved set of predictions and one decoding.

    python scripts/headline_cv.py --tag v10w --setting typical
    python scripts/headline_cv.py --tag sc_shipped --setting clean

Reads the probabilities a fold (or scripts/skillcorner_check.py) saved, decodes them with the
shipped setting unless --decode gives another, and prints spellF1, macroF1, live-or-dead F1, the
share of frames given to the right team while both truth and prediction are live, and how many
changes of state are found within a second. External matches only.
"""

import argparse
import glob
import json
import os

import numpy as np

from vantage import decode as decoding
from vantage import evaluate
from vantage.cli import DEFAULT_MODEL
from vantage.external import ROOT, source_of

KEYS = ("spellF1", "macroF1", "alive_dead", "team_acc", "trans25")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--setting", default="typical")
    parser.add_argument("--decode", default=None, help="a decoding setting as JSON; the shipped one otherwise")
    args = parser.parse_args()
    if args.decode:
        config = decoding.DecodeConfig(**json.loads(args.decode))
    else:
        with open(DEFAULT_MODEL) as handle:
            config = decoding.DecodeConfig(**json.load(handle)["decode"])
    rows = {}
    print("{:<11} {:>8} {:>8} {:>10} {:>9} {:>8}   {}".format(
        "match", "spellF1", "macroF1", "live/dead", "team", "trans25", "possessions found/true"))
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "cv", args.tag, "prob_*_{}.npz".format(args.setting)))):
        match = os.path.basename(path)[5:-4].rsplit("_", 1)[0]
        truth, predicted = [], []
        with np.load(path) as data:
            index = 0
            while "y{}".format(index) in data:
                boundary = data["b{}".format(index)] if "b{}".format(index) in data else None
                truth.append(data["y{}".format(index)])
                predicted.append(decoding.decode(data["p{}".format(index)].astype(np.float64), config, boundary))
                index += 1
        result = evaluate.headline(evaluate.score(truth, predicted))
        rows.setdefault(source_of(match), []).append(result)
        print("{:<11} {:>8.3f} {:>8.3f} {:>10.3f} {:>9.3f} {:>8.3f}   {}/{}".format(
            match, *(result[key] for key in KEYS), result["n_pred"], result["n_true"]), flush=True)
    for source, results in sorted(rows.items()):
        print("{:<11} {:>8.3f} {:>8.3f} {:>10.3f} {:>9.3f} {:>8.3f}   {:.2f}".format(
            "mean " + source[:6], *(float(np.mean([r[key] for r in results])) for key in KEYS),
            float(np.mean([r["n_pred"] / max(1, r["n_true"]) for r in results]))))


if __name__ == "__main__":
    main()
