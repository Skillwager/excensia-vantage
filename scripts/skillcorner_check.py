"""The shipped models on SkillCorner's matches: real automatic tracking from another provider.

Every other external match is corrected tracking put through this project's own simulator of
automatic tracking, so the models have only ever been scored on noise they were built for. These
twenty matches are tracking as a different company's system recorded it from the television
picture. Nothing is fitted here: the models and the decoding are the shipped ones.

    python scripts/skillcorner_check.py                      # all twenty, into runs/cv/sc_shipped
    python scripts/skillcorner_check.py --matches sc2017461 --tag sc_try

The predictions are saved like a cross-validation fold's (prob_<match>_clean.npz), so
`cv.py rescore`, `cv.py summary --sources skillcorner` and `error_report.py --setting clean`
read them. Two things differ from the challenge's labels and are in the numbers (see
vantage/skillcorner.py): possessions under a second are rare in this source, and its stoppages
begin when the last player lets the ball go, not when it crosses the line.
"""

import argparse
import json
import os
import time

import numpy as np

from vantage import decode as decoding
from vantage import evaluate, external, model, smoothing
from vantage.cli import DEFAULT_MODEL, load_model
from vantage.external import ROOT, SKILLCORNER_MATCHES, labelled_halves, seed_for


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tag", default="sc_shipped")
    parser.add_argument("--smooth", action="store_true",
                        help="bring every half's ball path to the reference smoothness first, "
                             "whatever the model card asks for")
    parser.add_argument("--as-recorded", action="store_true",
                        help="leave every half's ball path as it is, whatever the model card asks for")
    parser.add_argument("--decode", default=None, help="JSON of decoding settings, else the card's")
    args = parser.parse_args()
    state_model, boundary_model, card = load_model(args.model)
    names = card["features"]
    config = decoding.DecodeConfig(**(json.loads(args.decode) if args.decode else card["decode"]))
    out_dir = os.path.join(ROOT, "runs", "cv", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    print("{:<11} {:>8} {:>8} {:>10} {:>9} {:>8}   {}".format(
        "match", "spellF1", "macroF1", "live/dead", "team", "trans25", "possessions found/true"))
    for match in (args.matches or SKILLCORNER_MATCHES):
        started = time.time()
        truth, predicted, payload = [], [], {}
        for index, half in enumerate(labelled_halves(match)):
            if args.smooth:
                half, _ = smoothing.to_reference(half)
            elif not args.as_recorded:
                half, _ = smoothing.for_model(half, card)
            tagged = external.with_tags(half, np.random.default_rng(
                seed_for("tags", "eval", match, half.period, "clean")))
            state, boundary = model.predict_half_both(state_model, boundary_model, tagged, names)
            payload["y{}".format(index)] = half.labels
            payload["p{}".format(index)] = state.astype(np.float32)
            if boundary is not None:
                payload["b{}".format(index)] = boundary.astype(np.float32)
            truth.append(half.labels)
            predicted.append(decoding.decode(state.astype(np.float64), config, boundary))
        np.savez_compressed(os.path.join(out_dir, "prob_{}_clean.npz".format(match)), **payload)
        result = evaluate.headline(evaluate.score(truth, predicted))
        rows.append(result)
        print("{:<11} {:>8.3f} {:>8.3f} {:>10.3f} {:>9.3f} {:>8.3f}   {}/{}   ({:.0f}s)".format(
            match, result["spellF1"], result["macroF1"], result["alive_dead"], result["team_acc"],
            result["trans25"], result["n_pred"], result["n_true"], time.time() - started), flush=True)
    if len(rows) > 1:
        print("{:<11} {:>8.3f} {:>8.3f} {:>10.3f} {:>9.3f} {:>8.3f}".format(
            "mean", *(float(np.mean([row[key] for row in rows]))
                      for key in ("spellF1", "macroF1", "alive_dead", "team_acc", "trans25"))))


if __name__ == "__main__":
    main()
