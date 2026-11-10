"""The external labels under different control thresholds: how many possessions, how long.

For each source (the DFL matches, Metrica's sample games) and each threshold, spells shorter than
the threshold are merged into the possession before them (vantage/labels.py) and the result is
described: possessions per minute of live play and the share of them under 1, 2, 3 and 4 seconds.
Threshold 0 is the labels as the source gives them.

Only external matches are read.

    python scripts/label_stats.py
"""

import argparse

import numpy as np

from vantage import A, FPS, H, dfl, external, metrica_sample
from vantage.labels import harmonise, segments

THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def raw_halves(match):
    """The match's halves with the source's own labels, nothing merged."""
    if external.source_of(match) == "metrica":
        return metrica_sample.load_game(int(match[-1]), external.METRICA_DIR)
    return dfl.load_match(match, external.DFL_DIR, external.CACHE)


def describe(label_arrays):
    lengths, live = [], 0
    for labels in label_arrays:
        live += int(np.sum((labels == H) | (labels == A)))
        lengths += [(stop - start) / FPS for start, stop, code in segments(labels) if code in (H, A)]
    lengths = np.array(lengths)
    return (len(lengths) / (live / (60.0 * FPS)), np.mean(lengths < 1), np.mean(lengths < 2),
            np.mean(lengths < 3), np.mean(lengths < 4), np.median(lengths), len(lengths))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-match", action="store_true")
    args = parser.parse_args()

    groups = {"dfl": [m for m in external.ALL_MATCHES if external.source_of(m) == "dfl"],
              "metrica": list(external.METRICA_MATCHES)}
    print("{:<10}{:>6}{:>10}{:>8}{:>8}{:>8}{:>8}{:>9}{:>8}".format(
        "source", "merge", "per min", "<1 s", "<2 s", "<3 s", "<4 s", "median", "spells"))
    for source, matches in groups.items():
        loaded = {match: [half.labels for half in raw_halves(match)] for match in matches}
        units = [(source, [labels for match in matches for labels in loaded[match]])]
        if args.per_match:
            units += [(match, loaded[match]) for match in matches]
        for name, arrays in units:
            for threshold in THRESHOLDS:
                merged = [harmonise(labels, threshold) if threshold else labels for labels in arrays]
                per_minute, s1, s2, s3, s4, median, count = describe(merged)
                print("{:<10}{:>6.1f}{:>10.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>9.1f}{:>8}".format(
                    name, threshold, per_minute, s1, s2, s3, s4, median, count))


if __name__ == "__main__":
    main()
