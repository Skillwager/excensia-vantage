"""Where a finished file puts the restarts, against the set-piece tags and against the frame the
tracker picks the ball up; and what it calls the gaps in the ball's track.

A label-free check of a finished file against the tracking and the tags, never against the
video or the ground truth. It prints totals only: shares and percentiles over all the tagged
set pieces and over all the gaps of a length, never a frame or a time.

Why it matters. At the set pieces the tracker does not follow, the ball is picked up as the
restart is taken (after a tagged throw-in the track begins at the touchline with the ball
moving in), and that is 68 frames after the tag's own restart in the middle case
(scripts/track_report.py). A file that restarts play at the tag gives those seconds of dead
ball to a team.

    python scripts/restart_check.py submission/Excensia.csv --data-dir /path/to/data
"""

import argparse

import numpy as np

from vantage import D, FPS, competition, epts, tagging
from vantage import _official_metrics as official


def q(values, form="{:.0f}"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return "/".join(form.format(v) for v in np.percentile(values, (10, 25, 50, 75, 90))) if len(values) else "n/a"


def zone(point, pitch):
    length, width = pitch
    x, y = abs(point[0]), abs(point[1])
    if width / 2 - y < 3.0 or length / 2 - x < 3.0:
        return "line"
    return "field"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    table = official.read_label_csv(args.csv)
    reason = official.validate_submission(table, competition.N_FRAMES)
    if reason:
        raise SystemExit("INVALID: " + reason)
    labels = official.label_codes(table)
    paths = competition.tactical_paths(args.data_dir)
    home_name, away_name = epts.read_team_names(paths["meta"])
    tagged = tagging.restarts(tagging.load(paths["tagging"]), home_name, away_name)

    at_tag, at_start, from_tag, from_start, before_start = [], [], [], [], []
    gaps = []                                   # seconds, share called dead, lost zone
    for half in competition.load_halves(args.data_dir):
        own = labels[half.first_frame:half.first_frame + half.n_frames]
        dead = own == D
        seen = np.isfinite(half.ball_xy[:, 0])
        t = half.n_frames
        starts = np.flatnonzero(seen[1:] & ~seen[:-1]) + 1
        edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == -1)[:-1], np.flatnonzero(edges == 1)[1:]):
            if b - a >= 5 * FPS:
                gaps.append(((b - a) / FPS, float(np.mean(dead[a:b])), zone(half.ball_xy[a - 1], half.pitch)))
        for frame, _ in tagged:
            frame -= half.first_frame
            if frame < 12 * FPS or frame >= t - 12 * FPS or seen[frame - 2 * FPS:frame + 2 * FPS].all():
                continue
            near = starts[(starts >= frame - 10 * FPS) & (starts <= frame + 10 * FPS)]
            if not len(near):
                continue
            start = int(near[np.argmin(np.abs(near - frame))])
            at_tag.append(dead[frame])
            at_start.append(dead[max(start - 1, 0)])
            before_start.append(float(np.mean(dead[max(start - 2 * FPS, 0):start])))
            # where the file itself restarts play: the first frame it calls live from five
            # seconds before the tag on, if it calls that moment dead
            if dead[frame - 5 * FPS]:
                live = np.flatnonzero(~dead[frame - 5 * FPS:min(t, start + 10 * FPS)])
                if len(live):
                    first = frame - 5 * FPS + int(live[0])
                    from_tag.append(first - frame)
                    from_start.append(first - start)
    print("{} tagged set pieces the tracker does not follow, with a track beginning within ten seconds".format(len(at_tag)))
    print("   the file calls dead ball: the tag's own restart frame {:.0%}; the frame before the ball is picked up {:.0%}; "
          "the two seconds before the ball is picked up {:.0%} of their frames".format(
              np.mean(at_tag), np.mean(at_start), np.mean(before_start)))
    print("   the file's own restart ({} of them, dead five seconds before the tag): frames after the tag p10/25/50/75/90 {}; "
          "frames after the ball is picked up {}; within half a second of the pick-up {:.0%}, more than a second "
          "before it {:.0%}".format(len(from_tag), q(from_tag), q(from_start),
                                    np.mean(np.abs(np.array(from_start)) <= 12), np.mean(np.array(from_start) < -FPS)))
    gaps = np.array([(row[0], row[1], 1.0 if row[2] == "line" else 0.0) for row in gaps])
    print("{} gaps of five seconds or more in the ball's track".format(len(gaps)))
    for low, high in ((5, 10), (10, 30), (30, 1e9)):
        for name, flag in (("lost at a line", 1.0), ("lost in the field", 0.0)):
            pick = gaps[(gaps[:, 0] >= low) & (gaps[:, 0] < high) & (gaps[:, 2] == flag)]
            if len(pick):
                print("   {:>2.0f} to {:<3} s, {}: {:3d}; share of their frames called dead ball {:.0%}; called dead ball "
                      "throughout {:.0%}, live throughout {:.0%}".format(
                          low, "{:.0f}".format(high) if high < 1e8 else "...", name, len(pick), np.mean(pick[:, 1]),
                          np.mean(pick[:, 1] > 0.98), np.mean(pick[:, 1] < 0.02)))


if __name__ == "__main__":
    main()
