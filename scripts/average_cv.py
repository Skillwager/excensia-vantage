"""Several copies of the models, trained alike but for their random seed, answering together.

Each cross-validation run saves what its models said about every held-out match
(runs/cv/<tag>/prob_*.npz). Runs that differ only in their seed are the same method drawn
twice: averaging what they said is what a set of such copies would say if it were shipped. This
writes the average as a new set of prediction files, which `cv.py rescore` and `cv.py summary`
score like any other.

    python scripts/average_cv.py --tags v10w v12s1 v12s2 --out-tag v12avg
    python scripts/cv.py rescore --tag v12avg --out-tag v12avgb
    python scripts/cv.py summary --tag v12avgb --sources metrica

No competition data is read.
"""

import argparse
import glob
import os

import numpy as np

from vantage.external import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True, help="runs that differ only in their seed")
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
    print("averaged {} runs into {} files under runs/cv/{}".format(len(args.tags), written, args.out_tag))


if __name__ == "__main__":
    main()
