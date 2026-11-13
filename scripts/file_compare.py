"""Two finished files side by side: where they agree, and what kind of frame they differ on.

Label-free: two files of the method's own output and, with --data-dir, whether the ball is
tracked on each frame. Totals only. It chooses nothing; it says how far apart two versions of
the method are on the match, which the cross-validation on other matches cannot.

    python scripts/file_compare.py old/Excensia.csv new/Excensia.csv --data-dir /path/to/data
"""

import argparse

import numpy as np

from vantage import A, D, H, X, competition, epts
from vantage import _official_metrics as official
from vantage.labels import behaviour

NAMES = {H: "home", A: "away", D: "dead ball"}


def read(path):
    table = official.read_label_csv(path)
    reason = official.validate_submission(table, competition.N_FRAMES)
    if reason:
        raise SystemExit("{} is INVALID: {}".format(path, reason))
    return official.label_codes(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    first, second = read(args.first), read(args.second)
    seen = None
    if args.data_dir:
        paths = competition.tactical_paths(args.data_dir)
        seen = np.isfinite(epts.read(paths["meta"], paths["raw"]).ball[:, 0])
    graded = np.zeros(len(first), dtype=bool)
    for period in (1, 2):
        graded[competition.KICKOFF[period]:competition.GRADED[period][1] + 1] = True
    graded &= (first != X) & (second != X)
    a, b = first[graded], second[graded]
    print("{} frames from each kick-off to the end of its half; the two files agree on {:.1%}".format(
        int(graded.sum()), np.mean(a == b)))
    print("   first file: home {:.1%}, away {:.1%}, dead ball {:.1%}; second: home {:.1%}, away {:.1%}, dead ball {:.1%}".format(
        np.mean(a == H), np.mean(a == A), np.mean(a == D), np.mean(b == H), np.mean(b == A), np.mean(b == D)))
    for one in (H, A, D):
        for other in (H, A, D):
            if one != other:
                print("   {} in the first and {} in the second: {:.1%}".format(
                    NAMES[one], NAMES[other], np.mean((a == one) & (b == other))))
    live_both = (a != D) & (b != D)
    print("   where both call play live they name the same team on {:.1%}".format(np.mean(a[live_both] == b[live_both])))
    if seen is not None and np.any(a != b):
        s = seen[graded]
        differ = a != b
        print("   of the frames they differ on, the ball is tracked on {:.0%} (of all these frames: {:.0%}); they differ on "
              "{:.1%} of the frames with a tracked ball and on {:.1%} of those without".format(
                  np.mean(s[differ]), np.mean(s), np.mean(differ[s]), np.mean(differ[~s])))
    for name, labels in (("first", first), ("second", second)):
        halves = [(labels[competition.KICKOFF[period]:competition.GRADED[period][1] + 1], None) for period in (1, 2)]
        totals = behaviour(halves)
        print("   {} file: {} possessions ({:.2f} a live minute, median {:.1f} s, under 3 s {:.0%}); {} stoppages "
              "(median {:.1f} s)".format(name, totals["possessions"], totals["per_live_minute"],
                                         totals["possession_median"], totals["possession_under3"],
                                         totals["stoppages"], totals["stoppage_median"]))
    # stretches on which they differ, by length
    differ_all = np.zeros(len(first), dtype=bool)
    differ_all[graded] = a != b
    edges = np.diff(np.concatenate(([0], differ_all.astype(np.int8), [0])))
    lengths = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / 25.0
    if len(lengths):
        print("   they differ in {} stretches: seconds p25/50/75 {}; {} of them over five seconds, holding {:.0%} of the "
              "frames they differ on".format(len(lengths), "/".join("{:.1f}".format(v) for v in np.percentile(lengths, (25, 50, 75))),
                                             int(np.sum(lengths > 5)), lengths[lengths > 5].sum() / lengths.sum()))


if __name__ == "__main__":
    main()
