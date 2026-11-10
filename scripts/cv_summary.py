"""Combine the folds of a cross-validation run and pick one decoding setting for all of them.

    python scripts/cv_summary.py --tag v3
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv  # noqa: E402

from vantage.external import ROOT, source_of  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--skip", nargs="*", default=["clean"], help="settings left out of the choice")
    args = parser.parse_args()

    grid = cv.decode_grid()
    rows = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "cv", args.tag, "summary_*.json"))):
        with open(path) as handle:
            summary = json.load(handle)
        for key, value in summary["results"].items():
            rows[key] = value
    if not rows:
        raise SystemExit("no folds found")

    keys = [key for key in rows if key.split("|")[0] not in args.skip]
    table = np.array([[s["spellF1"] for s in rows[key]["grid"]] for key in keys])
    macro = np.array([[s["macroF1"] for s in rows[key]["grid"]] for key in keys])
    mean = table.mean(axis=0)
    order = np.argsort(-mean)
    print("{} match x setting cells, {} decoding settings".format(len(keys), len(grid)))
    for rank, index in enumerate(order[:8]):
        c = grid[index]
        print("#{:<3} mean spellF1 {:.4f} macroF1 {:.4f} | team {:>4} to_dead {:>4} from_dead {:>4} "
              "min_poss {:.1f} min_dead {:.1f}".format(
                  index, mean[index], macro[:, index].mean(), c.cost_team, c.cost_to_dead,
                  c.cost_from_dead, c.min_possession, c.min_dead))
    winner = int(order[0])
    print("\nwith setting #{}:".format(winner))
    settings = sorted({key.split("|")[0] for key in rows})
    matches = sorted({key.split("|")[1] for key in rows})
    print("{:<10}".format("") + "".join("{:>9}".format(name) for name in settings) + "{:>9}".format("poss"))
    for match in matches:
        cells = []
        for name in settings:
            value = rows.get("{}|{}".format(name, match))
            cells.append("{:>9.3f}".format(value["grid"][winner]["spellF1"]) if value else "{:>9}".format("-"))
        typical = rows.get("typical|{}".format(match))
        poss = "{}/{}".format(typical["grid"][winner]["n_pred"], typical["grid"][winner]["n_true"]) if typical else ""
        print("{:<10}".format(match) + "".join(cells) + "{:>9}".format(poss))
    for source in ("dfl", "metrica"):
        chosen = [key for key in keys if source_of(key.split("|")[1]) == source]
        if chosen:
            values = [rows[key]["grid"][winner]["spellF1"] for key in chosen]
            print("mean over {} cells ({}): {:.4f}".format(len(chosen), source, float(np.mean(values))))
    print("decode json: " + json.dumps({k: getattr(grid[winner], k) for k in
                                       ("cost_team", "cost_to_dead", "cost_from_dead", "min_possession", "min_dead")}))


if __name__ == "__main__":
    main()
