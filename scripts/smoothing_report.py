"""How smooth are the paths of the recording's tracker, and does the simulator match them?

A label-free check. It needs no ground truth and prints totals only: rates, shares, percentiles
and correlations over a whole match, never a frame, a position or a time.

A kick changes the ball's velocity at once. A tracker that smooths its ball path spreads that
change over many frames, and everything a model reads off the ball's motion (touches,
accelerations, the moment a pass leaves a foot) then looks different. This script measures the
spreading on the recording and on simulated recordings of the open matches, side by side:

* at every strong change of velocity (8 m/s or more between the velocity 12 frames before and 12
  frames after, taken where that difference peaks), the share of the change that is already
  there over 2, 4 and 8 frames either side;
* the same changes lined up on their middles and averaged (the curve of Figure 1 in the
  technical document): the largest share in one frame and the width at half its height;
* percentiles of the path's second difference (its frame-to-frame curvature);
* the correlation of the path's third difference with itself: an exact average of n frames puts
  a sharp negative mark at a lag of n, any other filter a broad one;
* touches a minute as vantage/features.py finds them, and the time from one to the next;
* whether the ball is in step with the players: for every frame of a carry (ball within 2 m of
  its nearest player, both moving the same way at 3 m/s or more) the distance from the player
  now to the ball s frames later; the s with the smallest median is how far the ball trails;
* the second difference of the players' own paths;
* the measure vantage/smoothing.py uses to decide whether a recording needs smoothing.

    python scripts/smoothing_report.py --data-dir /path/to/data     # the recording and the simulator
    python scripts/smoothing_report.py                               # the simulator only
"""

import argparse
import os

import numpy as np

from vantage import FPS, competition, external, smoothing
from vantage.degrade import DegradeConfig, sample_config
from vantage.features import (BALL_MAX_SPEED, BALL_VEL_HALFSPAN, PLAYER_MAX_SPEED,
                              PLAYER_VEL_HALFSPAN, _central_velocity, build)

SHIFTS = list(range(-15, 16))


def quantiles(values, points, form="{:.2f}"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n/a"
    return " ".join(form.format(v) for v in np.percentile(values, points))


def describe(halves):
    shares = {2: [], 4: [], 8: []}
    second, player_second, between = [], [], []
    strong = touches = visible = 0
    carry = {s: [] for s in SHIFTS}
    measured = []
    profile, changes = np.zeros(2 * smoothing.PROFILE_REACH + 1), 0
    for half in halves:
        ball = half.ball_xy.astype(np.float64)
        t = len(ball)
        seen = np.isfinite(ball[:, 0])
        visible += int(seen.sum())
        d2 = ball[2:] - 2 * ball[1:-1] + ball[:-2]
        curvature = np.sqrt(np.sum(d2 * d2, axis=1))
        second.append(curvature[np.isfinite(curvature)])
        velocity = _central_velocity(half.ball_xy.astype(np.float32), BALL_VEL_HALFSPAN,
                                     BALL_MAX_SPEED).astype(np.float64)
        frames, strength = smoothing._strong_changes(velocity)
        strong += len(frames)
        for halfspan in shares:
            part = smoothing._change(velocity, halfspan)
            best = np.full(len(frames), np.nan)
            for i, frame in enumerate(frames):
                window = part[max(0, frame - smoothing.REACH):frame + smoothing.REACH + 1]
                if np.isfinite(window).any():
                    best[i] = np.nanmax(window)
            shares[halfspan].append(best / strength[frames])
        part, found = smoothing.profile(half.ball_xy)
        profile += part
        changes += found
        measured.append("{:.2f}".format(smoothing.spread(half.ball_xy)[1]))

        names, table = build(half)
        column = {name: index for index, name in enumerate(names)}
        hits = np.flatnonzero((table[:, column["tch_since_h"]] == 0) | (table[:, column["tch_since_a"]] == 0))
        touches += len(hits)
        between.append(np.diff(hits) / float(FPS))

        players = np.concatenate([half.home_xy, half.away_xy], axis=1)
        p2 = players[2:].astype(np.float64) - 2 * players[1:-1] + players[:-2]
        p2 = np.sqrt(np.sum(p2 * p2, axis=-1))
        player_second.append(p2[np.isfinite(p2)][::11])

        ball32 = half.ball_xy.astype(np.float32)
        ball_velocity = _central_velocity(ball32, BALL_VEL_HALFSPAN, BALL_MAX_SPEED)
        ball_speed = np.sqrt(np.sum(ball_velocity * ball_velocity, axis=1))
        delta = players - ball32[:, None, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        distance = np.where(np.isfinite(distance), distance, np.inf)
        slot = np.argmin(distance, axis=1)
        rows = np.arange(t)
        nearest = distance[rows, slot]
        player_velocity = _central_velocity(players, PLAYER_VEL_HALFSPAN, PLAYER_MAX_SPEED)[rows, slot]
        player_speed = np.sqrt(np.sum(player_velocity * player_velocity, axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.sum(player_velocity * ball_velocity, axis=1) / (player_speed * ball_speed)
            carried = (nearest < 2.0) & (player_speed > 3.0) & (ball_speed > 3.0) & (cosine > 0.9)
        at = np.flatnonzero(carried)
        at = at[(at >= 16) & (at < t - 16)]
        spot = players[at, slot[at]]
        for s in SHIFTS:
            gap = ball32[at + s] - spot
            carry[s].append(np.sqrt(np.sum(gap * gap, axis=1)))

    minutes = visible / FPS / 60.0
    medians = {s: float(np.nanmedian(np.concatenate(v))) if len(np.concatenate(v)) else np.nan
               for s, v in carry.items()}
    trail = min(medians, key=medians.get)
    average = profile / max(changes, 1)
    third = smoothing.third_difference([half.ball_xy for half in halves])
    deepest = sorted(third, key=third.get)[:3]
    return {
        "strong changes of the ball's velocity, a minute of tracked ball": "{:.1f}".format(strong / max(minutes, 1e-9)),
        "share of one already there over 2 frames either side, p25/50/75": quantiles(np.concatenate(shares[2]), (25, 50, 75)),
        "share already there over 4 frames either side, p25/50/75": quantiles(np.concatenate(shares[4]), (25, 50, 75)),
        "share already there over 8 frames either side, p25/50/75": quantiles(np.concatenate(shares[8]), (25, 50, 75)),
        "the averaged change: changes, most in one frame, frames wide at half height": "{} {:.3f} {:.1f}".format(
            changes, float(average.max()), smoothing.half_width(average)),
        "ball: second difference, cm, p10/50/90/99": quantiles(np.concatenate(second) * 100, (10, 50, 90, 99), "{:.3g}"),
        "third difference with itself, deepest at lags (correlation)": ", ".join(
            "{} ({:+.2f})".format(lag, third[lag]) for lag in deepest),
        "third difference with itself at lags 20, 21, 22": " ".join("{:+.2f}".format(third[lag]) for lag in (20, 21, 22)),
        "touches found, a minute of tracked ball": "{:.1f}".format(touches / max(minutes, 1e-9)),
        "seconds between touches, p10/50/90": quantiles(np.concatenate(between), (10, 50, 90)),
        "frames the ball trails its players": "{:+d}".format(trail),
        "players: second difference, cm, p10/50/90/99": quantiles(np.concatenate(player_second) * 100, (10, 50, 90, 99), "{:.3g}"),
        "smoothness measure of each half (reference: at most {})".format(smoothing.REFERENCE): " ".join(measured[:8]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--matches", nargs="*", default=list(external.ALL_MATCHES))
    parser.add_argument("--first-generation", action="store_true",
                        help="add the simulator as it was before the paths were smoothed to match")
    args = parser.parse_args()
    columns = []
    if args.data_dir:
        columns.append(("the recording", describe(competition.load_halves(args.data_dir))))
    before = os.environ.get("VANTAGE_SIM")
    generations = ["g2"] + (["g1"] if args.first_generation else [])
    published = []
    for generation in generations:
        os.environ["VANTAGE_SIM"] = generation
        typical, drawn = [], []
        for match in args.matches:
            for half in external.labelled_halves(match):
                rng = np.random.default_rng(external.seed_for("report", match, half.period))
                typical.append(external.observed(half, rng, DegradeConfig()))
                drawn.append(external.observed(half, rng, sample_config(rng)))
                if generation == generations[0]:
                    published.append(external.observed(half, rng, None))
        label = "simulated" if generation == "g2" else "simulated, as first built"
        columns.append((label + ", typical setting", describe(typical)))
        columns.append((label + ", settings drawn at random", describe(drawn)))
    if before is None:
        del os.environ["VANTAGE_SIM"]
    else:
        os.environ["VANTAGE_SIM"] = before
    columns.append(("open matches as published", describe(published)))
    keys = list(columns[0][1])
    width = max(len(key) for key in keys)
    for name, column in columns:
        print("== " + name)
        for key in keys:
            print("   {:<{w}}  {}".format(key, column[key], w=width))


if __name__ == "__main__":
    main()
