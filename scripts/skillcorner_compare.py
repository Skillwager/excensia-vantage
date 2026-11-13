"""Two sets of saved predictions on SkillCorner's matches, side by side, match by match.

    python scripts/skillcorner_compare.py --base sc_shipped --other sc_try [--decode '<json>'] [--other-decode '<json>']

Each is decoded with the settings given (default: the shipped model card's) and scored with the
official metric code against SkillCorner's own labels. External matches only.
"""
import argparse
import json
import os

import numpy as np

from vantage import decode as decoding
from vantage import evaluate
from vantage.external import ROOT, SKILLCORNER_MATCHES


def scores(tag, config):
    out = {}
    for match in SKILLCORNER_MATCHES:
        path = os.path.join(ROOT, "runs", "cv", tag, "prob_{}_clean.npz".format(match))
        if not os.path.isfile(path):
            continue
        with np.load(path) as data:
            truth, predicted, index = [], [], 0
            while "y{}".format(index) in data:
                boundary = data["b{}".format(index)].astype(np.float64) if "b{}".format(index) in data else None
                truth.append(data["y{}".format(index)])
                predicted.append(decoding.decode(data["p{}".format(index)].astype(np.float64), config, boundary))
                index += 1
        out[match] = evaluate.headline(evaluate.score(truth, predicted))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--decode", default=None)
    parser.add_argument("--other-decode", default=None)
    args = parser.parse_args()
    with open(os.path.join(ROOT, "models", "vantage.json")) as handle:
        card = json.load(handle)["decode"]
    base_config = decoding.DecodeConfig(**(json.loads(args.decode) if args.decode else card))
    other_config = decoding.DecodeConfig(**(json.loads(args.other_decode) if args.other_decode
                                            else (json.loads(args.decode) if args.decode else card)))
    base, other = scores(args.base, base_config), scores(args.other, other_config)
    matches = [m for m in SKILLCORNER_MATCHES if m in base and m in other]
    print("{:<11} {:>9} {:>9} {:>8}   {:>9} {:>9}   {:>9} {:>9}".format(
        "match", args.base[:9], args.other[:9], "change", "team", "team", "live/dead", "live/dead"))
    for m in matches:
        print("{:<11} {:>9.3f} {:>9.3f} {:>+8.3f}   {:>9.3f} {:>9.3f}   {:>9.3f} {:>9.3f}".format(
            m, base[m]["spellF1"], other[m]["spellF1"], other[m]["spellF1"] - base[m]["spellF1"],
            base[m]["team_acc"], other[m]["team_acc"], base[m]["alive_dead"], other[m]["alive_dead"]))
    def mean(rows, key):
        return float(np.mean([rows[m][key] for m in matches]))
    gains = sum(1 for m in matches if other[m]["spellF1"] - base[m]["spellF1"] > 0.0005)
    losses = sum(1 for m in matches if other[m]["spellF1"] - base[m]["spellF1"] < -0.0005)
    print("{:<11} {:>9.4f} {:>9.4f} {:>+8.4f}   {:>9.3f} {:>9.3f}   {:>9.3f} {:>9.3f}   {} gain, {} lose of {}".format(
        "mean", mean(base, "spellF1"), mean(other, "spellF1"), mean(other, "spellF1") - mean(base, "spellF1"),
        mean(base, "team_acc"), mean(other, "team_acc"), mean(base, "alive_dead"), mean(other, "alive_dead"),
        gains, losses, len(matches)))
    print("possessions found / true: {} {}/{}   {} {}/{}".format(
        args.base, sum(base[m]["n_pred"] for m in matches), sum(base[m]["n_true"] for m in matches),
        args.other, sum(other[m]["n_pred"] for m in matches), sum(other[m]["n_true"] for m in matches)))


if __name__ == "__main__":
    main()
