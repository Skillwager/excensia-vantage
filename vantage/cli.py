"""Command line: regenerate the submission file from the provided data bundle.

    python -m vantage run --data-dir /path/to/data --out excensia.csv

`--data-dir` is the organisers' bundle (the folder that contains `tactical/`). It is read, never
copied. The only thing written is the label file, plus a small receipt next to it that records
what produced it (model, settings, package versions, hashes of the inputs and of the output) so
that a second run can be compared with the first.

Nothing about the match is printed beyond totals: label shares and the number of possessions.
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

from . import A, D, H, LABELS, X, __version__, competition, tagging
from . import decode as decoding
from . import epts, model
from .labels import spell_stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(os.path.dirname(HERE), "models", "vantage.txt")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path):
    import lightgbm

    booster = lightgbm.Booster(model_file=path)
    with open(os.path.splitext(path)[0] + ".json") as handle:
        card = json.load(handle)
    return booster, card


def run(args):
    booster, card = load_model(args.model)
    config = decoding.DecodeConfig(**card["decode"])
    paths = competition.tactical_paths(args.data_dir)

    halves = competition.load_halves(args.data_dir)
    vote = None
    home_name = away_name = ""
    if not args.no_tagging:
        meta = epts.read_team_names(paths["meta"])
        home_name, away_name = meta
        instances = tagging.load(paths["tagging"])
        vote = tagging.votes(instances, competition.N_FRAMES, home_name, away_name)

    decoded = []
    for half in halves:
        probability = model.predict_half(booster, half, card["features"])
        log_prob = np.log(np.clip(probability, config.floor, 1.0))
        if vote is not None:
            log_prob = log_prob + vote[half.first_frame:half.first_frame + half.n_frames]
        path = decoding.enforce_minimums(decoding.viterbi(log_prob, config), config)
        decoded.append((half.first_frame, path))

    labels = competition.assemble(decoded)
    competition.write_submission(args.out, labels)

    graded = labels != X
    counts = {name: int(np.sum(labels == code)) for code, name in enumerate(LABELS)}
    n_spells, mean_seconds, _, _ = spell_stats(labels)
    receipt = {
        "package": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": __import__("lightgbm").__version__,
        "model": {"file": os.path.basename(args.model), "sha256": sha256(args.model)},
        "decode": card["decode"],
        "tagging": not args.no_tagging,
        "inputs": {name: sha256(paths[name]) for name in ("meta", "raw", "tagging")},
        "output": {"file": os.path.basename(args.out), "sha256": sha256(args.out),
                   "rows": int(len(labels)), "counts": counts,
                   "possessions": int(n_spells), "mean_possession_seconds": round(mean_seconds, 2)},
    }
    with open(os.path.splitext(args.out)[0] + ".receipt.json", "w", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("wrote {}  ({} rows)".format(args.out, len(labels)))
    print("  graded frames {}: H {:.1%}  A {:.1%}  D {:.1%}".format(
        int(graded.sum()), *(np.mean(labels[graded] == code) for code in (H, A, D))))
    print("  {} possessions, mean {:.1f} s".format(n_spells, mean_seconds))
    print("  sha256 {}".format(receipt["output"]["sha256"]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vantage", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="write the submission file")
    runner.add_argument("--data-dir", required=True, help="folder containing tactical/")
    runner.add_argument("--out", required=True, help="where to write the frame,label CSV")
    runner.add_argument("--model", default=DEFAULT_MODEL)
    runner.add_argument("--no-tagging", action="store_true",
                        help="ignore the Smart Tagging file (tracking only)")
    args = parser.parse_args(argv)
    return {"run": run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
