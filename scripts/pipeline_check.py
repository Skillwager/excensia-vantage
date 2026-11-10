"""Check a finished submission file: format, the public five minutes, and its own totals.

Three things, none of which chooses anything:

1. The organisers' validator (the same code the official scorer runs).
2. The public five-minute ground truth, as a pipeline check: are the frames aligned, are home
   and away the right way round, is dead ball where dead ball is. Seventeen possessions cannot
   rank two versions of a method (EVALUATION.md shows it picking the wrong baseline), so this
   number is read once per candidate and never used to choose between candidates.
3. Label-free totals of the file against the ball's visibility in the tracking: how much of the
   untracked time is called dead ball, how many possessions per minute of live play. The same
   totals are printed by scripts/behaviour_report.py for the held-out external matches, where
   the truth is known, so a run that behaves unlike anything seen in validation shows up here
   without any ground truth.

    python scripts/pipeline_check.py runs/match/x/excensia.csv --data-dir /path/to/data \
        --official external/more-than-a-hack-2027
"""

import argparse
import os

import numpy as np

from vantage import competition, epts
from vantage import _official_metrics as official
from vantage.labels import behaviour


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--official", default=None, help="clone of the official repository")
    args = parser.parse_args()

    table = official.read_label_csv(args.csv)
    reason = official.validate_submission(table, competition.N_FRAMES)
    if reason:
        raise SystemExit("INVALID: " + reason)
    labels = official.label_codes(table)
    with open(args.csv, "rb") as handle:
        content = handle.read()
    print("1. format: valid, {} rows, {} bytes, LF only: {}".format(
        len(labels), len(content), b"\r" not in content))

    # From each kick-off to the end of the graded range: the stretch before the second-half
    # kick-off is dead ball by rule, not by anything the method decided, and is left out.
    seen = None
    if args.data_dir:
        paths = competition.tactical_paths(args.data_dir)
        seen = np.isfinite(epts.read(paths["meta"], paths["raw"]).ball[:, 0])
    halves = []
    for period in (1, 2):
        span = slice(competition.KICKOFF[period], competition.GRADED[period][1] + 1)
        halves.append((labels[span], None if seen is None else seen[span]))
    b = behaviour(halves)
    print("\n3. the file's own totals, kick-off to the end of each half")
    print("   {} frames: H {:.3f}  A {:.3f}  D {:.3f}   home share of live time {:.3f}".format(
        b["graded"], b["home"], b["away"], b["dead"], b["home_of_live"]))
    print("   {} possessions, {:.2f} per live minute; seconds mean {:.1f} median {:.1f}; "
          "under 3 s: {:.2f}".format(b["possessions"], b["per_live_minute"], b["possession_mean"],
                                     b["possession_median"], b["possession_under3"]))
    print("   {} stoppages, seconds mean {:.1f} median {:.1f}".format(
        b["stoppages"], b["stoppage_mean"], b["stoppage_median"]))
    if seen is not None:
        print("   untracked ball ({:.3f} of play): called dead {:.3f}".format(
            b["hidden"], b["hidden_called_dead"]))
        print("   tracked ball: called dead {:.3f}".format(b["visible_called_dead"]))
        print("   dead-ball frames with the ball untracked: {:.3f}".format(b["dead_while_hidden"]))

    if args.official:
        gt_path = os.path.join(args.official, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        truth = official.label_codes(official.read_label_csv(gt_path))
        result = official.evaluate(truth, labels)
        print("\n2. public five minutes (pipeline check only, 17 possessions)")
        print("   spellF1 {:.4f}  macroF1 {:.4f}  possessions {} vs {} true  mean {:.1f} s vs {:.1f} s"
              .format(result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
                      result["n_gt_spells"], result["mean_spell_duration_s"],
                      result["gt_mean_spell_duration_s"]))
        for name in ("H", "A", "D"):
            row = result["per_class"][name]
            print("   {}  precision {:.3f}  recall {:.3f}  f1 {:.3f}".format(
                name, row["precision"], row["recall"], row["f1"]))
        for tolerance in (12, 25):
            row = result["transitions"][tolerance]
            print("   changes within {:>2} frames: precision {:.2f} recall {:.2f}  median lag {}".format(
                tolerance, row["precision"], row["recall"], row["median_latency"]))
        print("   alive/dead F1 {:.3f}  team accuracy where both alive {:.3f}".format(
            result["alive_dead_f1"], result["team_acc_both_alive"]))


if __name__ == "__main__":
    main()
