"""Train the models that are shipped in models/, on all ten external matches.

    python scripts/cv.py prepare                    # once: builds the cached training tables
    python scripts/train_final.py --decode '<json>' # writes models/vantage_state.txt,
                                                    # vantage_boundary.txt and vantage.json

The competition match is not read here or anywhere else during training. The decoding settings
written into the model card are the ones chosen by cross-validation (scripts/cv.py summary);
pass them with --decode.
"""

import argparse
import hashlib
import json
import os
import platform

import lightgbm
import numpy as np

from vantage import __version__, external, model
from vantage import decode as decoding
from vantage.labels import CONTROL_SECONDS

ROOT = external.ROOT


def save(booster, path):
    """Write a model with LF line ends, so the file hashes the same on every platform."""
    booster.save_model(path)
    with open(path, "rb") as handle:
        content = handle.read().replace(b"\r\n", b"\n")
    with open(path, "wb") as handle:
        handle.write(content)
    return {"file": os.path.basename(path), "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content)}


def write_card(path, card):
    with open(path, "w", newline="\n") as handle:
        json.dump(card, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-tag", default="v10")
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=model.ROUNDS)
    parser.add_argument("--boundary-rounds", type=int, default=300)
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--metrica-weight", type=float, default=1.0)
    parser.add_argument("--short-weight", type=float, default=4.0,
                        help="largest weight given to the rows of a short possession "
                             "in the frame classifier (1 = none)")
    parser.add_argument("--short-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-params", default="{}",
                        help="JSON of LightGBM settings that replace the frame classifier's own")
    parser.add_argument("--decode", default="{}", help="JSON of DecodeConfig fields")
    parser.add_argument("--card-only", action="store_true",
                        help="rewrite the decoding settings of an existing model card, no training")
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "vantage"))
    args = parser.parse_args()

    config = decoding.DecodeConfig(**json.loads(args.decode))
    decode = {key: getattr(config, key) for key in config.__dataclass_fields__}
    if args.card_only:
        with open(args.out + ".json") as handle:
            card = json.load(handle)
        card["decode"] = decode
        write_card(args.out + ".json", card)
        print("decoding settings rewritten in " + args.out + ".json")
        return

    training = external.load_training(args.data_tag, external.ALL_MATCHES, args.variants)
    names, table, target = training["names"], training["table"], training["target"]
    print("training on {} rows x {} features from {} matches".format(
        len(table), len(names), len(external.ALL_MATCHES)), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    state_params = json.loads(args.state_params)
    state = model.fit(table, target, names, seed=args.seed, rounds=args.rounds,
                      params=state_params or None,
                      weight=external.row_weights(training, args.metrica_weight, args.short_weight,
                                                  args.short_seconds))
    models = {"state": save(state, args.out + "_state.txt"), "boundary": None}
    if not args.no_boundary:
        boundary = model.fit_boundary(table, training["change"], names, seed=args.seed,
                                      rounds=args.boundary_rounds,
                                      weight=external.row_weights(training, args.metrica_weight))
        models["boundary"] = save(boundary, args.out + "_boundary.txt")

    card = {
        "package": __version__,
        "features": names,
        "models": models,
        "decode": decode,
        "training": {
            "matches": list(external.ALL_MATCHES),
            "rows": int(len(table)),
            "class_counts": np.bincount(target, minlength=3).tolist(),
            "data_tag": args.data_tag,
            "variants": args.variants,
            "rounds": args.rounds,
            "boundary_rounds": args.boundary_rounds,
            "boundary_width": model.BOUNDARY_WIDTH,
            "seed": args.seed,
            "metrica_weight": args.metrica_weight,
            "short_weight": args.short_weight,
            "short_seconds": args.short_seconds,
            "params": dict(model.PARAMS, **state_params),
            "control_seconds": CONTROL_SECONDS,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    write_card(args.out + ".json", card)
    for role, entry in models.items():
        if entry:
            print("wrote {} model {} ({} bytes, sha256 {})".format(
                role, entry["file"], entry["bytes"], entry["sha256"]))


if __name__ == "__main__":
    main()
