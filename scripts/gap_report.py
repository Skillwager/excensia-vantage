"""Label-free look at the stretches where the ball is not tracked: how long, and what the players do.

Prints totals only (shares and percentiles over a half). A stoppage and a ball lost in live play
look different in the players: during a stoppage they slow to a walk. Comparing the spread of
"how fast the players moved while the ball was gone" between the recording and the simulator
shows whether the simulator's mix of the two is in a believable range.
"""
import argparse

import numpy as np

from vantage import D, FPS, competition, external
from vantage.degrade import DegradeConfig, sample_config
from vantage.features import PLAYER_MAX_SPEED, PLAYER_VEL_HALFSPAN, _central_velocity


def runs(flag):
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def describe(half):
    seen = np.isfinite(half.ball_xy[:, 0])
    both = np.concatenate([half.home_xy, half.away_xy], axis=1)
    velocity = _central_velocity(both, PLAYER_VEL_HALFSPAN, PLAYER_MAX_SPEED)
    speed = np.sqrt(np.sum(velocity * velocity, axis=-1))
    with np.errstate(invalid="ignore"):
        team_speed = np.nanmean(speed, axis=1)
    gaps = runs(~seen)
    lengths = np.array([b - a for a, b in gaps]) / FPS
    out = {}
    out["gaps"] = str(len(gaps))
    edges = [0, 2, 5, 10, 20, 40, 1e9]
    hist = np.histogram(lengths, bins=edges)[0] / max(len(lengths), 1)
    out["gap length share <2 / 2-5 / 5-10 / 10-20 / 20-40 / >40 s"] = " ".join("{:.2f}".format(v) for v in hist)
    gap_speed = np.array([np.nanmean(team_speed[a:b]) if np.isfinite(team_speed[a:b]).any() else np.nan
                          for a, b in gaps])
    long = lengths >= 3
    out["player speed in gaps >=3s: p10/25/50/75/90"] = " ".join(
        "{:.2f}".format(v) for v in np.nanpercentile(gap_speed[long], (10, 25, 50, 75, 90))) if long.any() else "n/a"
    out["share of gaps >=3s with players >1.8 / >2.2 m/s"] = "{:.2f} / {:.2f}".format(
        np.nanmean(gap_speed[long] > 1.8), np.nanmean(gap_speed[long] > 2.2)) if long.any() else "n/a"
    out["share of hidden time with players >1.8 / >2.2 m/s"] = "{:.2f} / {:.2f}".format(
        np.nansum(lengths[long] * (gap_speed[long] > 1.8)) / lengths.sum(),
        np.nansum(lengths[long] * (gap_speed[long] > 2.2)) / lengths.sum()) if long.any() else "n/a"
    vis_speed = team_speed[seen]
    out["player speed while ball visible: p10/50/90"] = " ".join(
        "{:.2f}".format(v) for v in np.nanpercentile(vis_speed, (10, 50, 90)))
    out["visible time with players <1.2 m/s share"] = "{:.3f}".format(float(np.nanmean(vis_speed < 1.2)))
    if half.labels is not None:
        dead = half.labels == D
        live_gap = np.array([1.0 - dead[a:b].mean() for a, b in gaps])
        out["TRUTH (simulator only): gaps mostly live share"] = "{:.2f}".format(float(np.mean(live_gap > 0.5)))
        out["TRUTH: hidden time that is live share"] = "{:.2f}".format(float((~dead)[~seen].mean()))
        out["TRUTH: dead time that is hidden share"] = "{:.2f}".format(float((~seen)[dead].mean()))
        out["TRUTH: players >2.2 m/s in mostly-live vs mostly-dead gaps"] = "{:.2f} vs {:.2f}".format(
            np.nanmean(gap_speed[long & (live_gap > 0.5)] > 2.2) if (long & (live_gap > 0.5)).any() else float("nan"),
            np.nanmean(gap_speed[long & (live_gap <= 0.5)] > 2.2) if (long & (live_gap <= 0.5)).any() else float("nan"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--matches", nargs="*", default=["J03WOH", "metrica1", "metrica3"])
    args = parser.parse_args()
    columns = []
    if args.data_dir:
        for half in competition.load_halves(args.data_dir):
            columns.append(("match h{}".format(half.period), describe(half)))
    for match in args.matches:
        half = external.labelled_halves(match)[0]
        for name, config in (("typical", DegradeConfig()), ("random", "random")):
            rng = np.random.default_rng(external.seed_for("gap", match, name))
            setting = sample_config(rng) if config == "random" else config
            columns.append(("{} {}".format(match[-4:], name), describe(external.observed(half, rng, setting))))
    keys = []
    for _, column in columns:
        for key in column:
            if key not in keys:
                keys.append(key)
    width = max(len(key) for key in keys)
    print(" " * width + " | " + " | ".join("{:>29}".format(name) for name, _ in columns))
    for key in keys:
        print("{:<{w}} | ".format(key, w=width) + " | ".join("{:>29}".format(column.get(key, "")) for _, column in columns))


if __name__ == "__main__":
    main()
