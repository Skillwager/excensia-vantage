"""What the ball is doing just before the tracker loses it for five seconds or more, how far
from there it is found again, and how much of the tracked ball is off the pitch. Label-free
for the recording (totals over all the track ends of a kind, never a frame or a time); for
simulated recordings of the open matches the same, split by their labels.

    python scripts/loss_report.py --data-dir /path/to/data [--generations g7] [--conditions typical]
"""
import argparse
import os
import sys

import numpy as np

from vantage import D, competition, external

FPS = 25
BACK = 20          # frames before the end of a track at which it is read (its last frames are held and slowed)


def q(values, form="{:.1f}"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return "/".join(form.format(v) for v in np.percentile(values, (10, 25, 50, 75, 90))) if len(values) else "n/a"


def pc(flags):
    flags = np.asarray(flags, dtype=np.float64)
    return "{:.0%}".format(float(np.mean(flags))) if len(flags) else "n/a"


def read(title, halves):
    rows = []          # speed at read, nearest at read, seconds still&near before read, dead share of the gap, gap s, field
    metres = []        # from where the ball was lost to where it is found, every gap of five seconds or more
    out_frames = out_still = tracked = 0
    out_by_label = {True: [0, 0], False: [0, 0]}
    for half in halves:
        ball = half.ball_xy.astype(np.float64)
        t = len(ball)
        length, width = half.pitch
        seen = np.isfinite(ball[:, 0])
        tracked += int(seen.sum())
        best = np.full(t, np.inf)
        for team in (half.home_xy, half.away_xy):
            delta = team.astype(np.float64) - np.where(seen[:, None], ball, 0.0)[:, None, :]
            distance = np.sqrt(np.sum(delta * delta, axis=-1))
            best = np.minimum(best, np.where(np.isfinite(distance), distance, np.inf).min(axis=1))
        speed = np.full(t, np.nan)
        moved = ball[5:] - ball[:-5]
        speed[2:-3] = np.sqrt(np.sum(moved * moved, axis=1)) * (FPS / 5.0)
        margin = np.minimum(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
        with np.errstate(invalid="ignore"):
            outside = seen & (margin < -1.0)
            still = seen & (best < 1.5) & (speed < 2.5)
        out_frames += int(outside.sum())
        out_still += int((outside & (speed < 2.5)).sum())
        if half.labels is not None:
            dead = half.labels == D
            for flag in (True, False):
                pick = seen & (dead == flag)
                out_by_label[flag][0] += int((outside & pick).sum())
                out_by_label[flag][1] += int(pick.sum())
        edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for index in range(len(stops) - 1):
            a, b = int(stops[index]), int(starts[index + 1])          # [a, b): no ball
            if b - a < 5 * FPS:
                continue
            metres.append(float(np.sqrt(np.sum((ball[b] - ball[a - 1]) ** 2))))
            read_at = a - BACK
            if read_at - int(starts[index]) < 30:
                continue
            rest = 0
            while read_at - rest >= int(starts[index]) and still[read_at - rest]:
                rest += 1
            dead_share = float(np.mean(half.labels[a:b] == D)) if half.labels is not None else np.nan
            field = 1.0 if margin[a - 1] >= 3.0 else 0.0
            rows.append((speed[read_at], best[read_at], rest / FPS, dead_share, (b - a) / FPS, field))
    rows = np.array(rows).reshape(-1, 6)
    print("== {}: {} tracks that end in a gap of five seconds or more, read {} frames before their end".format(
        title, len(rows), BACK))
    groups = [("all", np.ones(len(rows), dtype=bool)), ("lost in the field or a box", rows[:, 5] == 1),
              ("lost at a line", rows[:, 5] == 0), ("gaps of 5 to 10 s", rows[:, 4] < 10)]
    if np.isfinite(rows[:, 3]).any():
        groups += [("a stoppage for most of the gap", rows[:, 3] >= 0.5), ("live for most of it", rows[:, 3] < 0.5)]
    for name, pick in groups:
        part = rows[pick]
        if not len(part):
            continue
        resting = part[:, 2] > 0
        print("   {:<32} {:4d}: ball m/s p10/25/50/75/90 {}; under 2.5 m/s {}; lying still within 1.5 m of a player {}; "
              "of those, seconds it had lain there {} (over 1 s: {}, over 3 s: {})".format(
                  name, len(part), q(part[:, 0]), pc(part[:, 0] < 2.5), pc(resting), q(part[resting, 2]),
                  pc(part[resting, 2] > 1.0), pc(part[resting, 2] > 3.0)))
    if metres:
        print("   all {} gaps of five seconds or more: the ball is found {} m from where it was lost (p25/50/75)".format(
            len(metres), "/".join("{:.1f}".format(v) for v in np.percentile(metres, (25, 50, 75)))))
    print("   tracked ball more than 1 m outside the pitch: {:.2%} of tracked frames; of those, under 2.5 m/s {}".format(
        out_frames / max(tracked, 1), "{:.0%}".format(out_still / out_frames) if out_frames else "n/a"))
    if out_by_label[True][1]:
        print("   by the labels: {:.2%} of the tracked frames of stoppages, {:.2%} of those of live play".format(
            out_by_label[True][0] / max(out_by_label[True][1], 1), out_by_label[False][0] / max(out_by_label[False][1], 1)))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--generations", nargs="*", default=["g7"])
    parser.add_argument("--conditions", nargs="*", default=["typical"])
    parser.add_argument("--matches", nargs="*", default=["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9", "metrica1", "metrica2", "metrica3"])
    args = parser.parse_args()
    if args.data_dir:
        read("the recording", competition.load_halves(args.data_dir))
    if args.generations:
        clean = {match: external.labelled_halves(match) for match in args.matches}
    for generation in args.generations:
        os.environ["VANTAGE_SIM"] = generation
        for condition in args.conditions:
            for source in ("dfl", "metrica"):
                halves = [external.condition_half(half, match, condition)
                          for match in args.matches if external.source_of(match) == source for half in clean[match]]
                read("{} {} {}".format(generation, condition, source), halves)


if __name__ == "__main__":
    main()
