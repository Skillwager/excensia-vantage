"""Where do the misses come from? An error breakdown on held-out *external* matches.

Reads the probabilities a cross-validation fold saved, decodes them with one setting, and breaks
the possession-level errors down by length and by cause. External matches only.

    python scripts/error_report.py --tag v6 --setting typical \
        --decode '{"cost_team": 2.0, "cost_to_dead": 14.0, "cost_from_dead": 14.0, "boundary_weight": 2.0, "min_possession": 1.5, "min_dead": 0.5}'
"""

import argparse
import glob
import json
import os

import numpy as np

from vantage import A, D, H, evaluate
from vantage import _official_metrics as official
from vantage import decode as decoding
from vantage.external import ROOT
from vantage.labels import segments

BUCKETS = ((0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9))


def bucket_of(seconds):
    for index, (low, high) in enumerate(BUCKETS):
        if low <= seconds < high:
            return index
    return len(BUCKETS) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--setting", default="typical")
    parser.add_argument("--decode", required=True)
    args = parser.parse_args()
    config = decoding.DecodeConfig(**json.loads(args.decode))

    recall = np.zeros((len(BUCKETS), 2))
    precision = np.zeros((len(BUCKETS), 2))
    causes = {}
    start_errors, end_errors = [], []
    frame_confusion = np.zeros((3, 3), dtype=np.int64)
    dead_recall = np.zeros((len(BUCKETS), 2))
    dead_precision = np.zeros((len(BUCKETS), 2))

    paths = sorted(glob.glob(os.path.join(ROOT, "runs", "cv", args.tag,
                                          "prob_*_{}.npz".format(args.setting))))
    for path in paths:
        with np.load(path) as data:
            halves = []
            index = 0
            while "y{}".format(index) in data:
                boundary = data["b{}".format(index)] if "b{}".format(index) in data else None
                halves.append((data["y{}".format(index)], data["p{}".format(index)], boundary))
                index += 1
        for truth, prob, boundary in halves:
            predicted = decoding.decode(prob.astype(np.float64), config, boundary)
            graded = truth <= D
            np.add.at(frame_confusion, (truth[graded], predicted[graded]), 1)

            true_spells = [(a, b, k) for a, b, k in segments(truth) if k in (H, A)]
            pred_spells = [(a, b, k) for a, b, k in segments(predicted) if k in (H, A)]
            overlaps = official.spell_overlaps(true_spells, pred_spells)
            matched_true, matched_pred = {}, set()
            for iou, i, j in overlaps:
                if iou < 0.5:
                    break
                if i in matched_true or j in matched_pred:
                    continue
                matched_true[i] = j
                matched_pred.add(j)

            for i, (a, b, k) in enumerate(true_spells):
                bucket = bucket_of((b - a) / 25.0)
                recall[bucket, 1] += 1
                if i in matched_true:
                    recall[bucket, 0] += 1
                    pa, pb, _ = pred_spells[matched_true[i]]
                    start_errors.append(pa - a)
                    end_errors.append(pb - b)
                    continue
                window = predicted[a:b]
                same = np.mean(window == k)
                dead = np.mean(window == D)
                other = 1.0 - same - dead
                partners = [(pa, pb) for pa, pb, pk in pred_spells if pk == k and pb > a and pa < b]
                if same < 0.3:
                    cause = "called dead ball" if dead >= other else "called the other team"
                elif len(partners) >= 2:
                    cause = "split into pieces"
                elif partners and (partners[0][1] - partners[0][0]) > 1.6 * (b - a):
                    cause = "merged into a longer possession"
                else:
                    cause = "boundaries too far off"
                causes[cause] = causes.get(cause, 0) + 1
            for j, (a, b, k) in enumerate(pred_spells):
                bucket = bucket_of((b - a) / 25.0)
                precision[bucket, 1] += 1
                precision[bucket, 0] += j in matched_pred

            true_dead = [(a, b) for a, b, k in segments(truth) if k == D]
            pred_dead = [(a, b) for a, b, k in segments(predicted) if k == D]
            for a, b in true_dead:
                bucket = bucket_of((b - a) / 25.0)
                dead_recall[bucket, 1] += 1
                dead_recall[bucket, 0] += np.mean(predicted[a:b] == D) > 0.5
            for a, b in pred_dead:
                bucket = bucket_of((b - a) / 25.0)
                dead_precision[bucket, 1] += 1
                dead_precision[bucket, 0] += np.mean(truth[a:b] == D) > 0.5

    names = ["<3s", "3-6s", "6-12s", "12-25s", ">25s"]
    print("{} halves from {} matches, setting {!r}".format(
        "all", len(paths), args.setting))
    print("\npossessions matched at IoU 0.5, by length")
    print("{:<8} {:>8} {:>8} {:>10} {:>8} {:>8}".format("", "true", "recall", "", "pred", "precis"))
    for index, name in enumerate(names):
        print("{:<8} {:>8.0f} {:>8.2f} {:>10} {:>8.0f} {:>8.2f}".format(
            name, recall[index, 1], recall[index, 0] / max(recall[index, 1], 1), "",
            precision[index, 1], precision[index, 0] / max(precision[index, 1], 1)))
    print("{:<8} {:>8.0f} {:>8.2f} {:>10} {:>8.0f} {:>8.2f}".format(
        "all", recall[:, 1].sum(), recall[:, 0].sum() / recall[:, 1].sum(), "",
        precision[:, 1].sum(), precision[:, 0].sum() / precision[:, 1].sum()))
    total = sum(causes.values())
    print("\nwhy a true possession was not matched ({} of them)".format(total))
    for cause, count in sorted(causes.items(), key=lambda item: -item[1]):
        print("  {:<34} {:>5}  {:.2f}".format(cause, count, count / max(total, 1)))
    start_errors, end_errors = np.array(start_errors), np.array(end_errors)
    print("\nmatched possessions, boundary error in frames (predicted minus true)")
    for name, values in (("start", start_errors), ("end", end_errors)):
        print("  {:<6} median {:+.0f}  p10 {:+.0f}  p90 {:+.0f}  mean abs {:.1f}  within 12: {:.2f}  within 25: {:.2f}".format(
            name, np.median(values), np.percentile(values, 10), np.percentile(values, 90),
            np.mean(np.abs(values)), np.mean(np.abs(values) <= 12), np.mean(np.abs(values) <= 25)))
    print("\nstoppages found (mostly labelled dead), by length")
    for index, name in enumerate(names):
        print("  {:<8} true {:>5.0f} recall {:.2f}   predicted {:>5.0f} precision {:.2f}".format(
            name, dead_recall[index, 1], dead_recall[index, 0] / max(dead_recall[index, 1], 1),
            dead_precision[index, 1], dead_precision[index, 0] / max(dead_precision[index, 1], 1)))
    print("\nframes, rows = truth H A D, columns = predicted H A D (share of row)")
    for row, name in zip(frame_confusion, "HAD"):
        print("  {}  ".format(name) + "  ".join("{:.3f}".format(v) for v in row / max(row.sum(), 1)))


if __name__ == "__main__":
    main()
