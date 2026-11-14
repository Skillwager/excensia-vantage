"""Runs scored under one more condition, each at the decoding setting it was given elsewhere.

    python scripts/condition_compare.py --pairs G1x_g7:G1xM_g7 S7base_g7:S7baseM_g7 S7all_g7:S7allM_g7
    python scripts/condition_compare.py --pairs S7all_g7:S7allM_g7 R8_g7:R8M_g7 --setting '<json>'

Each pair is "the run whose cells choose the setting : the run to be read at that setting". The
first run of a pair is the model's cross-validation under the six conditions; its best setting,
chosen on Metrica's games as everywhere, is the one its model is read at in the second run, which
holds the same models' answers under another condition (`cv_variants.py --conditions mixed`). So
the added condition chooses nothing, not even a decoding setting. Prints the mean by source and
match by match, and the differences to the first pair. With --setting every second run is read
at the one setting given instead. External matches only.
"""

import argparse
import json

import numpy as np

from compare_cv import SKIP, chosen, given, load
from vantage.external import source_of


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--setting", type=json.loads, metavar="JSON",
                        help="read every run at this one decoding setting in place of the chosen ones")
    args = parser.parse_args()
    columns = []
    for pair in args.pairs:
        chooser, reader = pair.split(":")
        cells, grid, _ = load(chooser)
        setting = chosen(cells, [key for key in cells if key[0] not in SKIP])
        read, read_grid, _ = load(reader)
        if read_grid != grid:
            raise SystemExit("{} and {} were scored on different grids".format(chooser, reader))
        if args.setting is not None:
            setting = given(args.setting, read_grid, reader)
        conditions = sorted({key[0] for key in read})
        columns.append((reader, {match: float(np.mean([read[(c, m)][setting] for c, m in read if m == match]))
                                 for match in sorted({key[1] for key in read})}, conditions))
    matches = sorted(set.intersection(*(set(scores) for _, scores, _ in columns)))
    print("{} matches, conditions {}".format(len(matches), ", ".join(sorted({c for _, _, cs in columns for c in cs}))))
    print("{:<10}".format("match") + "".join("{:>12}".format(name[:11]) for name, _, _ in columns)
          + "".join("{:>12}".format("- first") for _ in columns[1:]))
    for match in matches:
        values = [scores[match] for _, scores, _ in columns]
        print("{:<10}".format(match) + "".join("{:>12.3f}".format(v) for v in values)
              + "".join("{:>+12.3f}".format(v - values[0]) for v in values[1:]))
    for label, keep in (("metrica", lambda m: source_of(m) == "metrica"), ("dfl", lambda m: source_of(m) == "dfl"),
                        ("all", lambda m: True)):
        chosen_matches = [m for m in matches if keep(m)]
        if not chosen_matches:
            continue
        means = [float(np.mean([scores[m] for m in chosen_matches])) for _, scores, _ in columns]
        print("{:<10}".format("mean " + label) + "".join("{:>12.4f}".format(v) for v in means)
              + "".join("{:>+12.4f}".format(v - means[0]) for v in means[1:]))
    first = columns[0][1]
    for name, scores, _ in columns[1:]:
        gains = sum(scores[m] > first[m] for m in matches)
        print("{} over {}: {} gain, {} lose of {}".format(
            name, columns[0][0], gains, sum(scores[m] < first[m] for m in matches), len(matches)))
    if args.setting is not None:
        print("every run read at the given setting: {}".format(json.dumps(args.setting)))


if __name__ == "__main__":
    main()
