"""The two teams' shape in a recording, and when many players are missing together. Label-free:
whole-match shares and percentiles of how many players each team has on a frame, where its
deepest players stand, and how often five or more of the twenty-two are missing and whether
that happens inside the long gaps of the ball's track. Never a frame, a place or a time. For
the open matches as published and as simulated the same.

    python scripts/shape_report.py --data-dir /path/to/data [--published] [--generations g7 g1]
"""
import argparse
import os
import sys

import numpy as np

from vantage import D, competition, external

FPS = 25


def q(values, form="{:.1f}"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return "/".join(form.format(v) for v in np.percentile(values, (10, 25, 50, 75, 90))) if len(values) else "n/a"


def shape(title, halves):
    counts = {"home": [], "away": []}
    deepest, second, gap12, width, depth, high = [], [], [], [], [], []
    for half in halves:
        length, _ = half.pitch
        for name, xy, sign in (("home", half.home_xy, 1.0), ("away", half.away_xy, -1.0)):
            ok = np.isfinite(xy[:, :, 0])
            counts[name].append(ok.sum(axis=1))
            # distance of each player from his own goal line (home defends -x, away +x)
            from_goal = np.where(ok, length / 2 + sign * xy[:, :, 0], np.inf)
            ordered = np.sort(from_goal, axis=1)
            has_two = np.isfinite(ordered[:, 1])
            deepest.append(ordered[has_two, 0])
            second.append(ordered[has_two, 1])
            gap12.append(ordered[has_two, 1] - ordered[has_two, 0])
            with np.errstate(invalid="ignore", all="ignore"):
                width.append(np.nanstd(np.where(ok, xy[:, :, 1], np.nan), axis=1))
                depth.append(np.nanstd(np.where(ok, xy[:, :, 0], np.nan), axis=1))
            far = np.where(ok, from_goal, -np.inf).max(axis=1)
            high.append(far[np.isfinite(far)])
    both = np.concatenate(counts["home"] + counts["away"])
    print("== " + title)
    print("   players of a team on a frame: 11 at {:.0%}, 10 at {:.0%}, 9 at {:.0%}, 8 or fewer at {:.0%}, 12 or more at {:.0%}; mean {:.2f}".format(
        np.mean(both == 11), np.mean(both == 10), np.mean(both == 9), np.mean(both <= 8), np.mean(both >= 12), both.mean()))
    deepest, second, gap12 = np.concatenate(deepest), np.concatenate(second), np.concatenate(gap12)
    print("   the deepest player, metres from his own goal line p10/25/50/75/90: {}; the second deepest {}; between the two {}".format(
        q(deepest), q(second), q(gap12)))
    print("   the deepest player within 16.5 m of his goal line: {:.0%}; more than 25 m from it: {:.0%}; the two deepest "
          "less than 5 m apart: {:.0%}".format(np.mean(deepest < 16.5), np.mean(deepest > 25.0), np.mean(gap12 < 5.0)))
    print("   a team's spread across the pitch (standard deviation, m) {}; along it {}; its furthest player from his own goal line {}".format(
        q(np.concatenate(width)), q(np.concatenate(depth)), q(np.concatenate(high))))
    sys.stdout.flush()


def missing(title, halves):
    low_all = low_in = gap_frames = frames = 0
    low_runs = []
    dead_low = dead_frames = live_low = live_frames = 0
    for half in halves:
        t = half.n_frames
        frames += t
        home = np.isfinite(half.home_xy[:, :, 0]).sum(axis=1)
        away = np.isfinite(half.away_xy[:, :, 0]).sum(axis=1)
        low = (home + away) <= 17                       # five or more of the twenty-two are missing
        seen = np.isfinite(half.ball_xy[:, 0])
        edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
        inside = np.zeros(t, dtype=bool)
        for a, b in zip(np.flatnonzero(edges == -1)[:-1], np.flatnonzero(edges == 1)[1:]):
            if b - a >= 15 * FPS:
                inside[a:b] = True
        low_all += int(low.sum())
        low_in += int((low & inside).sum())
        gap_frames += int(inside.sum())
        edges = np.diff(np.concatenate(([0], low.astype(np.int8), [0])))
        low_runs += list((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / FPS)
        if half.labels is not None:
            dead = half.labels == D
            dead_low += int((low & dead).sum())
            dead_frames += int(dead.sum())
            live_low += int((low & ~dead).sum())
            live_frames += int((~dead).sum())
    runs = np.array(low_runs)
    print("   {}: five or more players missing on {:.1%} of frames; {:.0%} of those frames lie in gaps of the ball's "
          "track of 15 s or more (which hold {:.0%} of all frames); in such gaps {:.1%} of frames, elsewhere {:.1%}".format(
              title, low_all / frames, low_in / max(low_all, 1), gap_frames / frames,
              low_in / max(gap_frames, 1), (low_all - low_in) / max(frames - gap_frames, 1)))
    if len(runs):
        print("   stretches with five or more missing: {} ({:.2f} a minute), seconds p25/50/75 {}; over 5 s: {}".format(
            len(runs), len(runs) / (frames / 1500.0), "/".join("{:.1f}".format(v) for v in np.percentile(runs, (25, 50, 75))),
            int(np.sum(runs > 5))))
    if dead_frames:
        print("   by the labels: {:.1%} of stoppage frames, {:.1%} of live frames".format(
            dead_low / dead_frames, live_low / max(live_frames, 1)))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--generations", nargs="*", default=["g7"])
    parser.add_argument("--published", action="store_true")
    parser.add_argument("--matches", nargs="*", default=["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9", "metrica1", "metrica2", "metrica3"])
    args = parser.parse_args()
    if args.data_dir:
        halves = competition.load_halves(args.data_dir)
        shape("the recording", halves)
        missing("the recording", halves)
    clean = {match: external.labelled_halves(match) for match in args.matches}
    if args.published:
        for source in ("dfl", "metrica"):
            shape("open matches as published, " + source,
                  [half for match in args.matches if external.source_of(match) == source for half in clean[match]])
    for generation in args.generations:
        os.environ["VANTAGE_SIM"] = generation
        for source in ("dfl", "metrica"):
            halves = [external.condition_half(half, match, "typical")
                      for match in args.matches if external.source_of(match) == source for half in clean[match]]
            shape("{} typical {}".format(generation, source), halves)
            missing("{} typical {}".format(generation, source), halves)


if __name__ == "__main__":
    main()
