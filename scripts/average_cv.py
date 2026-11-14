"""Several sets of models answering together: the average of what each said.

Each cross-validation run saves what its models said about every held-out match
(runs/cv/<tag>/prob_*.npz). This writes the average of several runs' answers as a new set of
prediction files, which `cv.py rescore` and `cv.py summary` score like any other.

Runs that differ only in their seed are the same method drawn twice: their average is what a
set of such copies would say if it were shipped. Runs of two different sets of models on the
same held-out recordings (the earlier models and the submitted ones) give the hedge between the
two simulators that section 6 of the technical document reports as tried and not adopted.

    python scripts/average_cv.py --tags v10w v12s1 v12s2 --out-tag v12avg
    python scripts/cv.py rescore --tag v12avg --out-tag v12avgb
    python scripts/cv.py summary --tag v12avgb --sources metrica

    python scripts/average_cv.py --tags S7all_g7 G1x_g7 --out-tag Havg_g7
    python scripts/cv.py rescore --tag Havg_g7 --out-tag H_g7
    python scripts/compare_cv.py --base S7all_g7 --other H_g7

The same works for runs of scripts/skillcorner_check.py (scripts/skillcorner_compare.py reads
the result). No competition data is read.
"""

import argparse
import glob
import json
import os

import numpy as np

from vantage.external import ROOT

import cv as cvlib


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tags", nargs="+", required=True,
                        help="runs that hold the same held-out recordings")
    parser.add_argument("--out-tag", required=True)
    args = parser.parse_args()
    first = os.path.join(ROOT, "runs", "cv", args.tags[0])
    out_dir = os.path.join(ROOT, "runs", "cv", args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for path in sorted(glob.glob(os.path.join(first, "prob_*.npz"))):
        name = os.path.basename(path)
        others = [os.path.join(ROOT, "runs", "cv", tag, name) for tag in args.tags[1:]]
        if not all(os.path.isfile(other) for other in others):
            print("skipped {}: not in every run".format(name))
            continue
        copies = [dict(np.load(each)) for each in [path] + others]
        out = {}
        for key in copies[0]:
            if key.startswith("y"):
                for copy in copies[1:]:
                    if not np.array_equal(copy[key], copies[0][key]):
                        raise SystemExit("{}: the runs do not hold the same frames".format(name))
                out[key] = copies[0][key]
            else:
                out[key] = np.mean([copy[key].astype(np.float64) for copy in copies], axis=0).astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, name), **out)
        written += 1
    # which simulator recorded the held-out matches, if the runs agree on it: `cv.py rescore`
    # passes it on, and compare_cv.py warns when two runs were not scored on the same recordings
    said = {cvlib.recorded_with(tag) for tag in args.tags}
    with open(os.path.join(out_dir, "averaged.json"), "w") as handle:
        json.dump({"averaged": args.tags, "generation": said.pop() if len(said) == 1 else None}, handle)
    print("averaged {} runs into {} files under runs/cv/{}".format(len(args.tags), written, args.out_tag))


if __name__ == "__main__":
    main()
