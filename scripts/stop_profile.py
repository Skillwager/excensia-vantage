"""What the ball does at the moment play stops (the labels' first dead frame): open matches as
published, as simulated, and SkillCorner's recorded tracking. Open data only.

For each stoppage of five seconds or more with a ball position around its start: the ball's
speed frame by frame from 15 frames before to 30 after, as a share of its speed just before
(frames -15 to -5), and the change of velocity across the first dead frame.

    python scripts/stop_profile.py [--generations g7] [--skillcorner 6]
"""
import argparse
import os
import sys

import numpy as np

from vantage import D, external, metrica_sample

FPS = 25
OFFSETS = (-10, -5, -2, 0, 2, 5, 10, 15, 20, 30)


def read(title, halves):
    rows, kinks, kinks_live, seen_share = [], [], [], []
    rng = np.random.default_rng(0)
    for half in halves:
        ball = half.ball_xy.astype(np.float64)
        t = len(ball)
        velocity = np.full(ball.shape, np.nan)
        velocity[1:-1] = (ball[2:] - ball[:-2]) * (FPS / 2.0)
        speed = np.sqrt(np.sum(velocity * velocity, axis=1))
        dead = half.labels == D
        edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
        change = np.full(t, np.nan)
        delta = velocity[6:] - velocity[:-6]
        change[3:-3] = np.sqrt(np.sum(delta * delta, axis=1))        # across three frames either side
        live_frames = np.flatnonzero(~dead[40:t - 40]) + 40
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start < 5 * FPS or start < 40 or start > t - 40:
                continue
            seen_share.append(float(np.mean(np.isfinite(ball[start:start + 25, 0]))))
            before = np.nanmean(speed[start - 15:start - 5]) if np.isfinite(speed[start - 15:start - 5]).any() else np.nan
            if not np.isfinite(before) or before < 2.0:
                continue
            rows.append([speed[start + offset] / before for offset in OFFSETS])
            kinks.append(change[start])
            other = int(rng.choice(live_frames))
            kinks_live.append(change[other])
    rows = np.array(rows, dtype=np.float64).reshape(-1, len(OFFSETS))
    kinks, kinks_live = np.array(kinks), np.array(kinks_live)
    print("== {}: {} stoppages with the ball moving at 2 m/s or more before them; a ball position on {:.0%} of "
          "the first second".format(title, len(rows), float(np.mean(seen_share)) if seen_share else float("nan")))
    print("   speed over the speed before, median, at frames " + "  ".join(
        "{:+d}: {:.2f}".format(offset, np.nanmedian(rows[:, index])) for index, offset in enumerate(OFFSETS)))
    print("   with a ball position at frames " + "  ".join(
        "{:+d}: {:.0%}".format(offset, np.mean(np.isfinite(rows[:, index]))) for index, offset in enumerate(OFFSETS)))
    ok, ok_live = np.isfinite(kinks), np.isfinite(kinks_live)
    print("   change of velocity across the first dead frame, m/s p25/50/75: {}; 3 m/s or more at {:.0%} "
          "(at frames of live play taken at random: {}, {:.0%})".format(
              "/".join("{:.1f}".format(v) for v in np.percentile(kinks[ok], (25, 50, 75))) if ok.any() else "n/a",
              np.mean(kinks[ok] >= 3.0) if ok.any() else float("nan"),
              "/".join("{:.1f}".format(v) for v in np.percentile(kinks_live[ok_live], (25, 50, 75))) if ok_live.any() else "n/a",
              np.mean(kinks_live[ok_live] >= 3.0) if ok_live.any() else float("nan")))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", nargs="*", default=["g7"])
    parser.add_argument("--skillcorner", type=int, default=6)
    parser.add_argument("--matches", nargs="*", default=["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9", "metrica1", "metrica2", "metrica3"])
    args = parser.parse_args()
    clean = {match: external.labelled_halves(match) for match in args.matches}
    read("DFL as published", [h for m in args.matches if external.source_of(m) == "dfl" for h in clean[m]])
    read("Metrica's sample with its dead-ball track drawn",
         [metrica_sample.with_dead_ball_track(h, np.random.default_rng(1))
          for m in args.matches if external.source_of(m) == "metrica" for h in clean[m]])
    for generation in args.generations:
        os.environ["VANTAGE_SIM"] = generation
        for source in ("dfl", "metrica"):
            for condition in ("typical", "watched"):
                read("{} {} {}".format(generation, condition, source),
                     [external.condition_half(h, m, condition)
                      for m in args.matches if external.source_of(m) == source for h in clean[m]])
    if args.skillcorner:
        halves = []
        for match in external.SKILLCORNER_MATCHES[:args.skillcorner]:
            halves += external.labelled_halves(match)
        read("SkillCorner as recorded ({} matches)".format(args.skillcorner), halves)


if __name__ == "__main__":
    main()
