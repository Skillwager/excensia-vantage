"""A copy of one model card that asks for the same preparation of a recording as another card.

    python scripts/equal_input_card.py --card /path/to/earlier/models/vantage.json \
        --like models/vantage.json --out runs/models/earlier_ref
    python scripts/skillcorner_check.py --model runs/models/earlier_ref/vantage.json --tag sc_earlier_ref
    python scripts/skillcorner_check.py                      # the shipped models, into sc_shipped
    python scripts/skillcorner_compare.py --base sc_earlier_ref --other sc_shipped \
        --base-model runs/models/earlier_ref/vantage.json

The models of the first simulator read a recording as it is; the present ones first bridge the
gaps the reference tracker never reports and bring the ball's path to the reference smoothness
(the card's "smoothing" entry; vantage/smoothing.py). To compare the two sets of models on
recordings prepared the same way, which is how the provided recording comes (it is already that
smooth and has no such gap), this writes a copy of the first card with the second card's
"smoothing" entry. The model files are copied unchanged, and their hashes stay in the card.
The earlier models are in this repository's history (tag `earlier-models`; `git worktree add
../earlier earlier-models` puts them beside this folder). Only the ball's path is prepared;
the players' paths stay as recorded. Open data only.
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--card", required=True, help="the card to copy (its model files lie beside it)")
    parser.add_argument("--like", required=True, help="the card whose preparation step is taken")
    parser.add_argument("--out", required=True, help="folder for the copy")
    args = parser.parse_args()
    with open(args.card) as handle:
        card = json.load(handle)
    with open(args.like) as handle:
        wanted = json.load(handle).get("smoothing")
    if not wanted:
        raise SystemExit(args.like + " asks for no preparation")
    source = os.path.dirname(os.path.abspath(args.card))
    os.makedirs(args.out, exist_ok=True)
    card["smoothing"] = wanted
    for entry in card["models"].values():
        if entry:
            shutil.copyfile(os.path.join(source, entry["file"]), os.path.join(args.out, entry["file"]))
    with open(os.path.join(args.out, "vantage.json"), "w", newline="\n") as handle:
        json.dump(card, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote {} with the preparation {}".format(os.path.join(args.out, "vantage.json"), wanted))


if __name__ == "__main__":
    main()
