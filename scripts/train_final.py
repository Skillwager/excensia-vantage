"""Train the model that is shipped in models/, on all ten external matches.

    python scripts/cv.py prepare            # once: builds the cached training tables
    python scripts/train_final.py           # writes models/vantage.txt and models/vantage.json

The competition match is not read here or anywhere else during training. The decoding settings
written into the model card are the ones chosen by cross-validation (scripts/cv.py); pass them
with --decode once that has been run.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-tag", default="v4")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=model.ROUNDS)
    parser.add_argument("--boundary-rounds", type=int, default=300)
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode", default="{}", help="JSON of DecodeConfig fields")
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "vantage"))
    args = parser.parse_args()

    names, table, target, change = external.load_training(
        args.data_tag, external.ALL_MATCHES, args.variants, with_change=True)
    print("training on {} rows x {} features from {} matches".format(
        len(table), len(names), len(external.ALL_MATCHES)), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    models = {"state": save(model.fit(table, target, names, seed=args.seed, rounds=args.rounds),
                            args.out + "_state.txt"), "boundary": None}
    if not args.no_boundary:
        models["boundary"] = save(
            model.fit_boundary(table, change, names, seed=args.seed, rounds=args.boundary_rounds),
            args.out + "_boundary.txt")

    config = decoding.DecodeConfig(**json.loads(args.decode))
    card = {
        "package": __version__,
        "features": names,
        "models": models,
        "decode": {key: getattr(config, key) for key in config.__dataclass_fields__},
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
            "params": model.PARAMS,
            "control_seconds": CONTROL_SECONDS,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    with open(args.out + ".json", "w", newline="\n") as handle:
        json.dump(card, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for role, entry in models.items():
        if entry:
            print("wrote {} model {} ({} bytes, sha256 {})".format(
                role, entry["file"], entry["bytes"], entry["sha256"]))


if __name__ == "__main__":
    main()
