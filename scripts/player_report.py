"""Label-free totals on how many players are tracked, for the recording and the simulator."""
import argparse
import numpy as np
from vantage import FPS, competition, external
from vantage.degrade import DegradeConfig, sample_config


def describe(half):
    length = half.pitch[0]
    home = np.isfinite(half.home_xy[..., 0]); away = np.isfinite(half.away_xy[..., 0])
    total = home.sum(1) + away.sum(1)
    out = {}
    buckets = [(0, 11), (12, 17), (18, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 99)]
    out["frames by players seen <=11/12-17/18-19/20/21/22/23/24+"] = " ".join(
        "{:.2f}".format(float(np.mean((total >= a) & (total <= b)))) for a, b in buckets)
    for name, xy, sign in (("home", half.home_xy, 1.0), ("away", half.away_xy, -1.0)):
        x = xy[..., 0] * sign
        deepest = np.nanmin(np.where(np.isfinite(x), x, np.inf), axis=1)
        deepest = deepest[np.isfinite(deepest)]
        out[name + " keeper-like player within 25m of own line"] = "{:.3f}".format(float(np.mean(deepest < -length / 2 + 25)))
        count = np.isfinite(x).sum(1)
        out[name + " count share <=9/10/11/12/13+"] = " ".join("{:.2f}".format(v) for v in (
            np.mean(count <= 9), np.mean(count == 10), np.mean(count == 11), np.mean(count == 12), np.mean(count >= 13)))
    # how concentrated are the losses: share of missing player-frames (vs 22) that fall in frames with <=17 seen
    missing = np.clip(22 - total, 0, None)
    out["missing player-frames in frames with <=17 seen (share)"] = "{:.2f}".format(
        float(missing[total <= 17].sum() / max(missing.sum(), 1)))
    out["mean missing per frame when >=18 seen"] = "{:.2f}".format(float(missing[total >= 18].mean()))
    ball = half.ball_xy
    step = ball[2:] - 2 * ball[1:-1] + ball[:-2]
    acc = np.sqrt(np.sum(step * step, axis=1)) * FPS * FPS
    acc = acc[np.isfinite(acc)]
    out["ball frame-to-frame acceleration m/s2 p50/90/99"] = " ".join("{:.1f}".format(v) for v in np.percentile(acc, (50, 90, 99)))
    return out


def main():
    parser = argparse.ArgumentParser()
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
            rng = np.random.default_rng(external.seed_for("players", match, name))
            setting = sample_config(rng) if config == "random" else config
            columns.append(("{} {}".format(match[-4:], name), describe(external.observed(half, rng, setting))))
    keys = list(columns[0][1])
    width = max(len(k) for k in keys)
    print(" " * width + " | " + " | ".join("{:>39}".format(n) for n, _ in columns))
    for key in keys:
        print("{:<{w}} | ".format(key, w=width) + " | ".join("{:>39}".format(c[key]) for _, c in columns))


if __name__ == "__main__":
    main()
