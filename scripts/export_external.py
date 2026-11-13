"""A run file for the viewer from one of the open external matches.

The match is passed through the tracking simulator under a named recording condition, labelled
by the models in models/ (or any model card), and written in the viewer's format, so the whole
path from tracking to the viewer can be seen on open data.

The models in models/ were trained on all ten external matches, this one included, so the result
flatters them. For an honest look, pass a model trained without the match.

    python scripts/export_external.py --match metrica1 --out runs/viewer/metrica1.run.json
"""

import argparse
import os
from dataclasses import replace

import numpy as np

from vantage import X, cli, evaluate, external, model, smoothing, viewer_export
from vantage import decode as decoding
from vantage.features import build

GAP = 1500   # frames of half-time between the two halves on the run file's timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", required=True, choices=external.ALL_MATCHES)
    parser.add_argument("--condition", default="typical", choices=list(external.CONDITIONS))
    parser.add_argument("--model", default=cli.DEFAULT_MODEL)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--page", default=None,
                        help="also write the viewer with this match built in, as one .html file")
    args = parser.parse_args()

    state_model, boundary_model, card = cli.load_model(args.model)
    config = decoding.DecodeConfig(**card["decode"])

    halves, states, evidence, pieces, truth, predicted = [], [], [], [], [], []
    offset = 0
    for half in external.labelled_halves(args.match):
        noisy = replace(external.condition_half(half, args.match, args.condition),
                        first_frame=offset)
        noisy, _ = smoothing.for_model(noisy, card)
        state, boundary = model.predict_half_both(state_model, boundary_model, noisy,
                                                  card["features"])
        codes = decoding.decode(state, config, boundary)
        built, table = build(noisy)
        names = card["features"]                     # the columns this model was trained on
        table = table[:, [built.index(name) for name in names]]
        evidence.append(viewer_export.evidence_shares(state_model, table, names,
                                                      codes.astype(np.int64), every=args.every))
        halves.append(noisy)
        states.append(state)
        truth.append(half.labels)
        predicted.append(codes)
        pieces += [codes, np.full(GAP, X, dtype=np.int8)]
        offset += half.n_frames + GAP
    labels = np.concatenate(pieces[:-1])

    source = "Metrica sample game" if external.source_of(args.match) == "metrica" else "DFL open data"
    run_file = viewer_export.build("{} {} ({} tracking)".format(source, args.match, args.condition),
                                   "Home", "Away", labels, halves, states, evidence,
                                   every=args.every)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    viewer_export.write(args.out, run_file)
    print("wrote {}  ({:.1f} MB)".format(args.out, os.path.getsize(args.out) / 1e6))
    if args.page:
        viewer_export.write_page(args.page, run_file, cli.DEFAULT_VIEWER,
                                 name=os.path.basename(args.out))
        print("wrote {}  (the viewer with this match built in)".format(args.page))
    for key, value in viewer_export.summary(run_file).items():
        print("  {}: {}".format(key, value))
    score = evaluate.headline(evaluate.score(truth, predicted))
    print("  against this match's labels: spellF1 {:.3f}, macroF1 {:.3f}, {} possessions for {}".format(
        score["spellF1"], score["macroF1"], score["n_pred"], score["n_true"]))


if __name__ == "__main__":
    main()
