"""Command line: regenerate the submission file from the provided data bundle.

    python -m vantage run --data-dir /path/to/data --out Excensia.csv

`--data-dir` is the organisers' bundle (the folder that contains `tactical/`). It is read, never
copied. The only thing written is the label file, plus a small receipt next to it that records
what produced it (model, settings, package versions, hashes of the inputs and of the output) so
that a second run can be compared with the first.

Nothing about the match is printed beyond totals: label shares and the number of possessions.

    python -m vantage export --data-dir /path/to/data --out runs/match/run.json

writes the same analysis as a run file for the viewer (viewer/): the labels, how sure the model
was of each possession and on what evidence, and thinned-out tracking. That file is derived
match data and stays on the machine that has the match.
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
from . import epts, model, viewer_export
from .features import build as build_features
from .labels import spell_stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(os.path.dirname(HERE), "models", "vantage.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(card_path):
    """(state model, boundary model or None, card). The card names the model files beside it and
    their hashes; a file that does not match its hash is refused."""
    import lightgbm

    with open(card_path) as handle:
        card = json.load(handle)
    folder = os.path.dirname(os.path.abspath(card_path))
    boosters = {}
    for role in ("state", "boundary"):
        entry = card["models"].get(role)
        if entry is None:
            boosters[role] = None
            continue
        path = os.path.join(folder, entry["file"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("{} does not match the hash in the model card".format(path))
        boosters[role] = lightgbm.Booster(model_file=path)
    return boosters["state"], boosters["boundary"], card


def analyse(args):
    """Label the match. Returns (labels, halves, state probabilities per half, models, card)."""
    state_model, boundary_model, card = load_model(args.model)
    config = decoding.DecodeConfig(**card["decode"])
    paths = competition.tactical_paths(args.data_dir)

    halves = competition.load_halves(args.data_dir)
    vote = None
    if not args.no_tagging:
        home_name, away_name = epts.read_team_names(paths["meta"])
        instances = tagging.load(paths["tagging"])
        vote = tagging.votes(instances, competition.N_FRAMES, home_name, away_name)
        # tagged set pieces: when play resumed and for whom, as features for the model
        tagged = tagging.restarts(instances, home_name, away_name)
        for half in halves:
            half.meta["restarts"] = [(frame - half.first_frame, team) for frame, team in tagged
                                     if 0 <= frame - half.first_frame < half.n_frames]

    decoded, states = [], []
    for half in halves:
        state, boundary = model.predict_half_both(state_model, boundary_model, half,
                                                  card["features"])
        extra = None
        if vote is not None:
            extra = vote[half.first_frame:half.first_frame + half.n_frames]
        decoded.append((half.first_frame, decoding.decode(state, config, boundary, extra)))
        states.append(state)

    return competition.assemble(decoded), halves, states, state_model, card


def run(args):
    labels, _, _, _, card = analyse(args)
    paths = competition.tactical_paths(args.data_dir)
    competition.write_submission(args.out, labels)

    graded = labels != X
    counts = {name: int(np.sum(labels == code)) for code, name in enumerate(LABELS)}
    n_spells, mean_seconds, _, _ = spell_stats(labels)
    receipt = {
        "package": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": __import__("lightgbm").__version__,
        "model": {"card": os.path.basename(args.model), "card_sha256": sha256(args.model),
                  "files": card["models"]},
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


def export(args):
    labels, halves, states, state_model, card = analyse(args)
    paths = competition.tactical_paths(args.data_dir)
    home_name, away_name = epts.read_team_names(paths["meta"])
    evidence = None
    if not args.no_evidence:
        evidence = []
        for half in halves:
            built, table = build_features(half)
            names = card["features"]                 # the columns this model was trained on
            table = table[:, [built.index(name) for name in names]]
            codes = labels[half.first_frame:half.first_frame + half.n_frames]
            evidence.append(viewer_export.evidence_shares(
                state_model, table, names, np.minimum(codes, 2).astype(np.int64),
                every=args.every))
    run_file = viewer_export.build("{} v {}".format(home_name, away_name), home_name, away_name,
                                   labels, halves, states, evidence, every=args.every)
    folder = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(folder, exist_ok=True)
    viewer_export.write(args.out, run_file)
    print("wrote {}  ({:.1f} MB)".format(args.out, os.path.getsize(args.out) / 1e6))
    for key, value in viewer_export.summary(run_file).items():
        print("  {}: {}".format(key, value))
    print("  this file holds match tracking: keep it on this machine")
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
    exporter = sub.add_parser("export", help="write a run file for the viewer")
    exporter.add_argument("--data-dir", required=True, help="folder containing tactical/")
    exporter.add_argument("--out", required=True, help="where to write the run file (.json)")
    exporter.add_argument("--model", default=DEFAULT_MODEL)
    exporter.add_argument("--no-tagging", action="store_true")
    exporter.add_argument("--every", type=int, default=5,
                          help="keep one tracking sample in this many frames")
    exporter.add_argument("--no-evidence", action="store_true",
                          help="leave out the per-possession evidence shares (faster)")
    args = parser.parse_args(argv)
    return {"run": run, "export": export}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
