"""Two cross-validation runs side by side, on the cells both of them scored.

    python scripts/compare_cv.py --base v11b --other v13sc
    python scripts/compare_cv.py --base v11b --other v13sc --held-out-setting
    python scripts/compare_cv.py --base v11b --other v13sc --setting '<json>'

Each run is read at its own best decoding setting, chosen on Metrica's games (the "clean"
condition is left out, as everywhere). Prints the mean by source and, match by match, the mean
over the simulated conditions and the difference. External matches only.

By default the setting is chosen once, on the Metrica cells of the matches both runs hold, which
is how the shipped setting is chosen. With --held-out-setting it is chosen for every fold on the
Metrica cells of the other folds, so that no match is scored at a setting it helped to choose;
the difference between two runs should not depend on which of the two is used. With --setting
both runs are read at the one setting given, which takes the setting out of the comparison: two
runs each at its own setting differ by their models and by their settings.
"""

import argparse
import glob
import json
import os

import numpy as np

from vantage.external import ROOT, source_of

SKIP = ("clean",)
RECORDED = {}       # tag -> the simulator generations its held-out matches were recorded with


def load(tag):
    """({(condition, match): [score per setting]}, the grid, {match: its fold}) of one run."""
    cells, grid, fold_of = {}, None, {}
    for fold, path in enumerate(sorted(glob.glob(os.path.join(ROOT, "runs", "cv", tag, "summary_*.json")))):
        with open(path) as handle:
            data = json.load(handle)
        if grid is not None and data["grid"] != grid:
            raise SystemExit("the folds of {} were scored on different grids of settings".format(tag))
        grid = data["grid"]
        # summaries from before the generation was written down are the first generation's
        RECORDED.setdefault(tag, set()).add(data.get("generation", "g1"))
        for key, scores in data["results"].items():
            condition, match = key.split("|")
            cells[(condition, match)] = np.array([score["spellF1"] for score in scores])
            fold_of[match] = fold
    return cells, grid, fold_of


def chosen(cells, keys):
    """The index of the setting that is best on the Metrica cells among `keys` (on all of them
    if none is Metrica's)."""
    metrica = [key for key in keys if source_of(key[1]) == "metrica"] or list(keys)
    return int(np.argmax(np.mean([cells[key] for key in metrica], axis=0)))


def given(setting, grid, tag):
    """The index, in the grid of the run `tag`, of the decoding setting given on the command line
    (None if none was). Stops if the run was not scored at that setting."""
    if setting is None:
        return None
    if not grid or setting not in grid:
        raise SystemExit("the setting {} is not in the grid of {}".format(json.dumps(setting), tag))
    return grid.index(setting)


def per_match(cells, keys, fold_of, held_out, fixed=None):
    """({match: mean over its conditions}, the settings used). With `fixed`, the index of a
    setting, every match is read at that one and none is chosen."""
    matches = sorted({key[1] for key in keys})
    if not held_out:
        setting = chosen(cells, keys) if fixed is None else fixed
        return {match: float(np.mean([cells[key][setting] for key in keys if key[1] == match]))
                for match in matches}, [setting]
    out, used = {}, []
    for match in matches:
        others = [key for key in keys if fold_of[key[1]] != fold_of[match]]
        if not others:
            raise SystemExit("one fold only: there is nothing to choose the setting on")
        setting = chosen(cells, others)
        used.append(setting)
        out[match] = float(np.mean([cells[key][setting] for key in keys if key[1] == match]))
    return out, sorted(set(used))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--held-out-setting", action="store_true",
                        help="choose each fold's decoding setting on the other folds")
    parser.add_argument("--setting", type=json.loads, metavar="JSON",
                        help="read both runs at this one decoding setting in place of their own")
    args = parser.parse_args()
    if args.setting is not None and args.held_out_setting:
        raise SystemExit("--setting gives the setting and --held-out-setting chooses it: use one or the other")
    base_cells, base_grid, base_fold = load(args.base)
    other_cells, other_grid, other_fold = load(args.other)
    if not base_cells or not other_cells:
        raise SystemExit("no results yet for {}".format(args.base if not base_cells else args.other))
    keys = sorted(key for key in set(base_cells) & set(other_cells) if key[0] not in SKIP)
    matches = sorted({key[1] for key in keys})
    everything = {key[1] for key in base_cells} | {key[1] for key in other_cells}
    missing = sorted(everything - set(matches))
    conditions = sorted({key[0] for key in keys})
    uneven = [m for m in matches if sum(1 for key in keys if key[1] == m) != len(conditions)]
    base, base_settings = per_match(base_cells, keys, base_fold, args.held_out_setting,
                                    given(args.setting, base_grid, args.base))
    other, other_settings = per_match(other_cells, keys, other_fold, args.held_out_setting,
                                      given(args.setting, other_grid, args.other))
    print("{} matches in both, conditions {}{}".format(
        len(matches), " ".join(conditions), "; only in one: " + " ".join(missing) if missing else ""))
    if uneven:
        print("NOT EVERY CONDITION for: " + " ".join(uneven))
    if RECORDED.get(args.base) != RECORDED.get(args.other):
        print("NOT THE SAME TEST: {} was scored on recordings of simulator {}, {} on {}".format(
            args.base, "/".join(sorted(RECORDED.get(args.base, []))),
            args.other, "/".join(sorted(RECORDED.get(args.other, [])))))
    print("{:<10} {:>9} {:>9} {:>8}".format("match", args.base[:9], args.other[:9], "change"))
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
    if args.held_out_setting:
        print("settings chosen on the other folds: {} for {}, {} for {}".format(
            len(base_settings), args.base, len(other_settings), args.other))
    elif args.setting is not None:
        print("both read at the given setting: {}".format(json.dumps(args.setting)))
    else:
        print("settings: {} -> {}".format(json.dumps(base_grid[base_settings[0]]), json.dumps(other_grid[other_settings[0]])))


if __name__ == "__main__":
    main()
