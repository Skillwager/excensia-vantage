"""Are the tagged set pieces anchored to the restart? A label-free profile around each tag.

Every set-piece tag is a clip of fixed length. If the clip is cut at a fixed offset from the
moment play restarts, that moment shows as a step in what the tracking sees, averaged over all
the tags of one kind: before it the players are slow and the ball is often untracked, after it
they are not. This prints, for each second from 20 s before a tag starts to 40 s after, the
average over the tags of: the share of frames with a tracked ball, the players' mean speed, the
ball's speed where tracked, and, if a submission file is given, the share it calls dead ball.

Totals over tags only; no ground truth is involved.

    python scripts/set_piece_profile.py --data-dir /path/to/data [--csv file.csv]
"""

import argparse
import collections

import numpy as np

from vantage import D, FPS, competition, tagging
from vantage import _official_metrics as official
from vantage.features import build

BEFORE, AFTER = 20, 40


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    n = competition.N_FRAMES
    seen = np.zeros(n, dtype=bool)
    players = np.full(n, np.nan, dtype=np.float32)
    ball = np.full(n, np.nan, dtype=np.float32)
    covered = np.zeros(n, dtype=bool)
    for half in competition.load_halves(args.data_dir):
        names, table = build(half)
        span = slice(half.first_frame, half.first_frame + half.n_frames)
        seen[span] = np.isfinite(half.ball_xy[:, 0])
        players[span] = table[:, names.index("t_speed")]
        ball[span] = table[:, names.index("b_speed")]
        covered[span] = True
    dead = None
    if args.csv:
        dead = official.label_codes(official.read_label_csv(args.csv)) == D

    instances = tagging.load(competition.tactical_paths(args.data_dir)["tagging"])
    by_type = collections.defaultdict(list)
    for item in instances:
        if item["code"] == tagging.SET_PIECE:
            by_type[item["tags"].get("Type", "?")].append(item)
            by_type["all set pieces"].append(item)

    for kind, items in sorted(by_type.items(), key=lambda pair: -len(pair[1])):
        if len(items) < 5:
            continue
        lengths = sorted({round((item["end"] - item["start"]) / FPS, 1) for item in items})
        print("\n{} ({} tags, clip lengths {} s)".format(kind, len(items), lengths))
        print("  second   ball tracked   players m/s   ball m/s" + ("   file: dead" if dead is not None else ""))
        for second in range(-BEFORE, AFTER):
            track, speed, pace, called = [], [], [], []
            for item in items:
                a = item["start"] + second * FPS
                b = a + FPS
                if a < 0 or b > n or not covered[a:b].all():
                    continue
                track.append(seen[a:b].mean())
                speed.append(np.nanmean(players[a:b]) if np.isfinite(players[a:b]).any() else np.nan)
                pace.append(np.nanmean(ball[a:b]) if np.isfinite(ball[a:b]).any() else np.nan)
                if dead is not None:
                    called.append(dead[a:b].mean())
            if not track:
                continue
            row = "  {:>+5}     {:>8.2f}     {:>8.2f}     {:>6.2f}".format(
                second, np.mean(track), np.nanmean(speed), np.nanmean(pace))
            if dead is not None:
                row += "     {:>6.2f}".format(np.mean(called))
            print(row + ("   <- tag starts" if second == 0 else ""))
        # The restart is tagging.RESTART_LEAD into the clip. The five seconds before it are a
        # stoppage whatever the tracker saw of the ball, so they show without any label how
        # often a stoppage keeps its ball in view.
        lead = int(round(tagging.RESTART_LEAD * FPS))
        spans = [slice(item["start"] + lead - 5 * FPS, item["start"] + lead) for item in items
                 if item["start"] + lead - 5 * FPS >= 0
                 and covered[item["start"] + lead - 5 * FPS:item["start"] + lead].all()]
        if spans:
            tracked = np.array([seen[span].mean() for span in spans])
            print("  in the five seconds before the restart the ball is tracked on {:.2f} of frames; "
                  "on at least half of them at {} of {} tags, on every one at {}".format(
                      tracked.mean(), int(np.sum(tracked >= 0.5)), len(spans),
                      int(np.sum(tracked >= 1.0))))
        if dead is not None and spans:
            before = np.array([dead[span].mean() for span in spans])
            print("  in the five seconds before the restart the file calls dead ball on "
                  "{:.2f} of frames; at least half of them at {} of {} tags".format(
                      before.mean(), int(np.sum(before >= 0.5)), len(spans)))


if __name__ == "__main__":
    main()
