"""Two cross-validation runs side by side, on the matches both of them scored.

    python scripts/compare_cv.py --base v11b --other v13sc

Each run is read at its own best decoding setting, chosen on the Metrica cells of the matches
both runs hold (the "clean" condition is left out, as everywhere). Prints the mean by source and,
match by match, the mean over the simulated conditions and the difference. External matches only.
"""

import argparse
import glob
import json
import os

import numpy as np

from vantage.external import ROOT, source_of

SKIP = ("clean",)


def load(tag):
    """({(condition, match): [score per setting]}, the grid) of one run."""
    cells, grid = {}, None
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "cv", tag, "summary_*.json"))):
        with open(path) as handle:
            data = json.load(handle)
        grid = data["grid"]
        for key, scores in data["results"].items():
            condition, match = key.split("|")
            cells[(condition, match)] = np.array([score["spellF1"] for score in scores])
    return cells, grid


def best(cells, matches):
    """(index of the setting best on the Metrica cells of `matches`, {match: mean over conditions})."""
    keys = [key for key in cells if key[1] in matches and key[0] not in SKIP]
    metrica = [key for key in keys if source_of(key[1]) == "metrica"] or keys
    setting = int(np.argmax(np.mean([cells[key] for key in metrica], axis=0)))
    per_match = {match: float(np.mean([cells[key][setting] for key in keys if key[1] == match]))
                 for match in matches}
    return setting, per_match


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    args = parser.parse_args()
    base_cells, base_grid = load(args.base)
    other_cells, other_grid = load(args.other)
    matches = sorted({key[1] for key in base_cells} & {key[1] for key in other_cells})
    missing = sorted(({key[1] for key in base_cells} | {key[1] for key in other_cells}) - set(matches))
    base_setting, base = best(base_cells, matches)
    other_setting, other = best(other_cells, matches)
    print("{} matches in both{}".format(len(matches), "; only in one: " + " ".join(missing) if missing else ""))
    print("{:<10} {:>9} {:>9} {:>8}".format("match", args.base, args.other, "change"))
    for match in matches:
        print("{:<10} {:>9.3f} {:>9.3f} {:>+8.3f}".format(match, base[match], other[match], other[match] - base[match]))
    for source in ("metrica", "dfl"):
        rows = [match for match in matches if source_of(match) == source]
        if rows:
            a = float(np.mean([base[m] for m in rows]))
            b = float(np.mean([other[m] for m in rows]))
            print("{:<10} {:>9.4f} {:>9.4f} {:>+8.4f}   ({} matches)".format("mean " + source[:5], a, b, b - a, len(rows)))
    a = float(np.mean([base[m] for m in matches]))
    b = float(np.mean([other[m] for m in matches]))
    gains = sum(1 for m in matches if other[m] - base[m] > 0.0005)
    losses = sum(1 for m in matches if other[m] - base[m] < -0.0005)
    print("{:<10} {:>9.4f} {:>9.4f} {:>+8.4f}   {} gain, {} lose, {} unchanged".format(
        "mean all", a, b, b - a, gains, losses, len(matches) - gains - losses))
    print("settings: {} -> {}".format(json.dumps(base_grid[base_setting]), json.dumps(other_grid[other_setting])))


if __name__ == "__main__":
    main()
