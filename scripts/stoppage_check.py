"""Where is the ball untracked, and does a file call those stretches dead ball? Totals only.

Part 1 needs no labels at all: for each five-minute block of the match, the share of frames with
no tracked ball, and the share a submission file calls dead ball.

Part 2 is a pipeline check on the public five minutes, the one place the truth is known: of the
frames that are truly dead ball, how many have a tracked ball; of the frames truly in play, how
many have none; and how the players and the ball behave in each case. It answers one question,
what the ground truth means by a dead ball in terms the tracking can see. Nothing is fitted.

    python scripts/stoppage_check.py --data-dir /path/to/data --csv runs/match/x/excensia.csv \
        --official external/more-than-a-hack-2027
"""

import argparse
import os

import numpy as np

from vantage import D, FPS, competition
from vantage import _official_metrics as official
from vantage.features import build
from vantage.labels import segments

BLOCK = 5 * 60 * FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--official", default=None)
    args = parser.parse_args()

    halves = competition.load_halves(args.data_dir)
    labels = None
    if args.csv:
        labels = official.label_codes(official.read_label_csv(args.csv))

    print("1. per five-minute block: share of frames with no tracked ball"
          + (", share the file calls dead ball" if labels is not None else ""))
    for half in halves:
        seen = np.isfinite(half.ball_xy[:, 0])
        cells = []
        for start in range(0, half.n_frames, BLOCK):
            stop = min(start + BLOCK, half.n_frames)
            cell = "{:.2f}".format(1.0 - seen[start:stop].mean())
            if labels is not None:
                codes = labels[half.first_frame + start:half.first_frame + stop]
                cell += "/{:.2f}".format(np.mean(codes == D))
            cells.append(cell)
        print("   half {}: {}".format(half.period, "  ".join(cells)))
        gaps = [(b - a) / FPS for a, b, k in segments(seen.astype(np.int8)) if k == 0]
        print("   half {}: {} gaps in the ball, {:.2f} per minute, median {:.1f} s; "
              "under 3 s: {:.2f}, over 20 s: {:.2f}".format(
                  half.period, len(gaps), len(gaps) / (half.n_frames / (60.0 * FPS)),
                  np.median(gaps), np.mean(np.array(gaps) < 3), np.mean(np.array(gaps) > 20)))

    if not args.official:
        return
    gt_path = os.path.join(args.official, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    truth = official.label_codes(official.read_label_csv(gt_path))
    graded = np.flatnonzero(truth != 3)
    first, last = int(graded[0]), int(graded[-1])
    half = halves[0]
    if first < half.first_frame or last >= half.first_frame + half.n_frames:
        raise SystemExit("the public window is not inside the first half")
    window = slice(first - half.first_frame, last + 1 - half.first_frame)
    names, table = build(half)
    column = {name: table[window, index] for index, name in enumerate(names)}
    seen = np.isfinite(half.ball_xy[window, 0])
    true = truth[first:last + 1]
    dead, live = true == D, true != D

    print("\n2. public five minutes ({} frames, {} dead, {} in play)".format(
        len(true), int(dead.sum()), int(live.sum())))
    print("   ball untracked on {:.3f} of the window".format(1.0 - seen.mean()))
    print("   truly dead frames with a tracked ball: {:.3f}".format(np.mean(seen[dead])))
    print("   truly live frames with no tracked ball: {:.3f}".format(np.mean(~seen[live])))
    stoppages = [(a, b) for a, b, k in segments(true) if k == D]
    print("   {} stoppages; share of each with a tracked ball: {}".format(
        len(stoppages), "  ".join("{:.2f}".format(seen[a:b].mean()) for a, b in stoppages)))

    def describe(name, mask, label):
        values = column[name][mask]
        values = values[np.isfinite(values)]
        if not len(values):
            return "   {:<14} {:<22} no values".format(name, label)
        q = np.percentile(values, [10, 50, 90])
        return "   {:<14} {:<22} p10 {:>7.2f}  median {:>7.2f}  p90 {:>7.2f}".format(
            name, label, *q)

    print("   how the tracking looks, by what the truth says (percentiles over frames):")
    for name in ("b_speed", "b_margin", "b_dmin", "t_speed", "t_slow", "h_speed", "a_speed"):
        if name not in column:
            continue
        print(describe(name, dead & seen, "dead, ball tracked"))
        print(describe(name, live & seen, "live, ball tracked"))
        if name.startswith(("t_", "h_", "a_")):
            print(describe(name, dead & ~seen, "dead, ball untracked"))
    if labels is not None:
        mine = labels[first:last + 1]
        print("   the file calls dead: {:.3f} of truly dead frames with a tracked ball, "
              "{:.3f} of those without".format(
                  np.mean(mine[dead & seen] == D) if (dead & seen).any() else float("nan"),
                  np.mean(mine[dead & ~seen] == D) if (dead & ~seen).any() else float("nan")))


if __name__ == "__main__":
    main()
