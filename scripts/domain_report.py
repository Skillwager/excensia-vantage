"""Compare what automatic tracking looks like with what the noise simulator produces.

A label-free check: it needs no ground truth and prints totals only (percentiles and shares over
a whole half, never a frame, a position or a time). It answers one question: is the real
recording inside the range of conditions the model was trained on? Where it is not, the
simulator's ranges are widened; nothing is fitted to a label.

    python scripts/domain_report.py --data-dir /path/to/data     # the match and the simulator
    python scripts/domain_report.py                               # the simulator only
"""

import argparse

import numpy as np

from vantage import FPS, competition, external
from vantage.degrade import DegradeConfig, sample_config
from vantage.features import (BALL_MAX_SPEED, BALL_VEL_HALFSPAN, PLAYER_MAX_SPEED,
                              PLAYER_VEL_HALFSPAN, _central_velocity)


def run_lengths(flag):
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def percentiles(values, points=(10, 50, 90, 99)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n/a"
    return " ".join("{:.2f}".format(v) for v in np.percentile(values, points))


def describe(half):
    length, width = half.pitch
    ball = half.ball_xy
    seen = np.isfinite(ball[:, 0])
    out = {}
    out["ball hidden share"] = "{:.3f}".format(1 - seen.mean())
    gaps = run_lengths(~seen) / FPS
    runs = run_lengths(seen) / FPS
    out["hidden gaps: count/min"] = "{:.1f}".format(len(gaps) / (len(seen) / FPS / 60.0))
    out["hidden gap seconds p10/50/90/99"] = percentiles(gaps)
    out["share of hidden time in gaps >5s / >15s"] = "{:.2f} / {:.2f}".format(
        gaps[gaps > 5].sum() / max(gaps.sum(), 1e-9), gaps[gaps > 15].sum() / max(gaps.sum(), 1e-9))
    out["visible run seconds p10/50/90/99"] = percentiles(runs)

    margin = np.minimum(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
    out["visible ball outside pitch share"] = "{:.3f}".format(float(np.mean(margin[seen] < 0)))
    out["visible ball >1m outside share"] = "{:.3f}".format(float(np.mean(margin[seen] < -1)))
    velocity = _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED)
    speed = np.sqrt(np.sum(velocity * velocity, axis=1))
    out["ball speed m/s p10/50/90/99"] = percentiles(speed)
    out["ball at rest (<0.8 m/s) share of visible"] = "{:.3f}".format(
        float(np.mean(speed[np.isfinite(speed)] < 0.8)))
    still = run_lengths(np.nan_to_num(speed, nan=99.0) < 0.8) / FPS
    out["ball-at-rest run seconds p50/90/99"] = percentiles(still, (50, 90, 99))
    jump = np.sqrt(np.sum(np.square(ball[1:] - ball[:-1]), axis=1)) * FPS
    out["frame-to-frame ball speed >45 m/s share"] = "{:.4f}".format(
        float(np.mean(jump[np.isfinite(jump)] > BALL_MAX_SPEED)))

    both = np.concatenate([half.home_xy, half.away_xy], axis=1)
    present = np.isfinite(both[..., 0])
    counts = present.sum(axis=1)
    out["players seen mean / p5 / p95"] = "{:.2f} / {:.0f} / {:.0f}".format(
        counts.mean(), np.percentile(counts, 5), np.percentile(counts, 95))
    home_count = np.isfinite(half.home_xy[..., 0]).sum(axis=1)
    away_count = np.isfinite(half.away_xy[..., 0]).sum(axis=1)
    out["frames with a team over 11 share"] = "{:.3f}".format(
        float(np.mean((home_count > 11) | (away_count > 11))))
    track_runs = np.concatenate([run_lengths(present[:, slot]) for slot in range(present.shape[1])]
                                ) / FPS
    out["track run seconds p10/50/90"] = percentiles(track_runs, (10, 50, 90))
    second = both[2:] - 2 * both[1:-1] + both[:-2]
    jitter = np.sqrt(np.sum(second * second, axis=-1))
    out["player jitter (2nd difference, m) p50/90"] = percentiles(jitter, (50, 90))
    player_velocity = _central_velocity(both, PLAYER_VEL_HALFSPAN, PLAYER_MAX_SPEED)
    player_speed = np.sqrt(np.sum(player_velocity * player_velocity, axis=-1))
    out["player speed m/s p50/90/99"] = percentiles(player_speed, (50, 90, 99))
    outside = (np.abs(both[..., 0]) > length / 2 + 1) | (np.abs(both[..., 1]) > width / 2 + 1)
    out["player positions >1m outside pitch share"] = "{:.4f}".format(
        float(np.sum(outside & present) / max(present.sum(), 1)))

    delta = both - ball[:, None, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    nearest_all = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    nearest = nearest_all[np.isfinite(nearest_all)]
    out["nearest player to visible ball (m) p10/50/90/99"] = percentiles(nearest)
    out["visible ball with nobody within 10m share"] = "{:.3f}".format(float(np.mean(nearest > 10)))

    few = counts <= 14
    out["frames with <=14 players share"] = "{:.3f}".format(float(few.mean()))
    out["  of those, ball hidden share"] = "{:.2f}".format(float((~seen)[few].mean()) if few.any() else float("nan"))

    # How a visible run of the ball begins and ends (first / last half second of each run).
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (stops - starts) >= 25
    starts, stops = starts[keep], stops[keep]
    def window_mean(values, index, width, forward):
        rows = [values[i:i + width] if forward else values[max(0, i - width):i] for i in index]
        return np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in rows])
    if len(starts):
        s_speed = window_mean(speed, starts, 12, True); e_speed = window_mean(speed, stops, 12, False)
        s_margin = window_mean(margin, starts, 12, True); e_margin = window_mean(margin, stops, 12, False)
        s_near = window_mean(np.where(np.isfinite(nearest_all), nearest_all, np.nan), starts, 12, True)
        e_near = window_mean(np.where(np.isfinite(nearest_all), nearest_all, np.nan), stops, 12, False)
        out["runs begin: at rest / within 2m of edge / nobody within 5m"] = "{:.2f} / {:.2f} / {:.2f}".format(
            np.nanmean(s_speed < 0.8), np.nanmean(s_margin < 2.0), np.nanmean(s_near > 5.0))
        out["runs end:   at rest / within 2m of edge / nobody within 5m"] = "{:.2f} / {:.2f} / {:.2f}".format(
            np.nanmean(e_speed < 0.8), np.nanmean(e_margin < 2.0), np.nanmean(e_near > 5.0))
        out["runs end:   speed m/s p50/90"] = percentiles(e_speed, (50, 90))
        out["runs begin: speed m/s p50/90"] = percentiles(s_speed, (50, 90))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--matches", nargs="*", default=["J03WOH", "metrica1"])
    args = parser.parse_args()

    columns = []
    if args.data_dir:
        for half in competition.load_halves(args.data_dir):
            columns.append(("match h{}".format(half.period), describe(half)))
    for match in args.matches:
        half = external.labelled_halves(match)[0]
        for name, config in (("typical", DegradeConfig()), ("random", "random")):
            rng = np.random.default_rng(external.seed_for("report", match, name))
            setting = sample_config(rng) if config == "random" else config
            columns.append(("{} {}".format(match[-4:], name),
                            describe(external.observed(half, rng, setting))))

    keys = list(columns[0][1])
    width = max(len(key) for key in keys)
    print(" " * width + " | " + " | ".join("{:>22}".format(name) for name, _ in columns))
    for key in keys:
        print("{:<{w}} | ".format(key, w=width)
              + " | ".join("{:>22}".format(column[key]) for _, column in columns))


if __name__ == "__main__":
    main()
