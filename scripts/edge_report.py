"""How the recording's tracks begin and end, and whether the simulator's tracks do the same.

A label-free check. It prints totals only: medians and shares over all the tracks of a match at
fixed distances from a track's first and last frame, never a frame, a position or a time.

A tracker that averages its paths has to do something where a track runs out of frames, and what
it does shows in the speed there. For every unbroken track of 60 frames or more this takes the
speed k frames into the track over the speed 15 to 25 frames in, and the same from the end, and
gives the median over all tracks. Three ways of ending a track are easy to tell apart:

  * the whole true path is averaged and frames are hidden afterwards: nothing happens at the
    ends, but the first and last frames of a track hold something of frames the tracker never
    had (the simulator's second generation did this, and an independent review caught it);
  * the average runs out of frames: both ends move at half speed for ten frames;
  * a lost ball or player is still reported where it was last seen for a few frames, and the
    average takes that place for everything after: a track begins at full speed and slows to a
    stop over its last 16 frames. With 5 frames held and an average of 21, the speed k frames
    before the end is (k + 5) / 21 of what it was.

The recording does the third, for the ball and for the players alike, and the simulator's third
generation does it too. The same reading gives the width of the players' own average: a strong
change of a player's velocity, lined up and averaged as Figure 1 of the technical document does
for the ball, is as wide as an average of 21 frames makes it.

    python scripts/edge_report.py --data-dir /path/to/data            # the recording and the simulator
    python scripts/edge_report.py --data-dir /path/to/data --earlier  # and the two earlier generations
"""

import argparse
import os

import numpy as np

from vantage import FPS, competition, external, smoothing

OFFSETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20)
LONG_ENOUGH = 60        # frames: tracks shorter than this are left out
FLOOR = 3.0             # m/s: and tracks slower than this 15 to 25 frames from the end in question


def ends(tracks):
    """({k: median speed k frames in / inner speed}, the same from the end, share of tracks whose
    first step is under half the inner speed, the same for the last step, tracks read at the
    start, at the end)."""
    first = {k: [] for k in OFFSETS}
    last = {k: [] for k in OFFSETS}
    slow_first, slow_last = [], []
    for track in tracks:
        track = np.asarray(track, dtype=np.float64)
        seen = np.isfinite(track[:, 0])
        step = np.full(len(track), np.nan)
        moved = track[1:] - track[:-1]
        step[1:] = np.sqrt(np.sum(moved * moved, axis=1)) * FPS
        edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start < LONG_ENOUGH:
                continue
            inner = np.mean(step[start + 15:start + 25])
            if inner > FLOOR:
                for k in OFFSETS:
                    first[k].append(step[start + k] / inner)
                slow_first.append(step[start + 1] < 0.5 * inner)
            inner = np.mean(step[stop - 25:stop - 15])
            if inner > FLOOR:
                for k in OFFSETS:
                    last[k].append(step[stop - k] / inner)
                slow_last.append(step[stop - 1] < 0.5 * inner)

    def medians(values):
        return {k: float(np.median(v)) for k, v in values.items() if v}

    def share(flags):
        return float(np.mean(flags)) if flags else float("nan")

    return medians(first), medians(last), share(slow_first), share(slow_last), len(slow_first), len(slow_last)


def players_of(halves):
    return [team[:, slot] for half in halves for team in (half.home_xy, half.away_xy)
            for slot in range(team.shape[1])]


def width(tracks, strong):
    total, count = np.zeros(2 * smoothing.PROFILE_REACH + 1), 0
    for track in tracks:
        part, found = smoothing.profile(track, strong)
        total += part
        count += found
    if not count:
        return "no such change"
    return "{} changes, {:.1f} frames wide at half height".format(count, smoothing.half_width(total / count))


def line(values):
    return " ".join("{}:{:.2f}".format(k, values[k]) for k in OFFSETS if k in values)


def apart(values, reference):
    shared = [k for k in OFFSETS if k in values and k in reference]
    return float(np.mean([abs(values[k] - reference[k]) for k in shared])) if shared else float("nan")


def report(name, halves, reference=None):
    print("== " + name)
    got = {"ball": ends([half.ball_xy for half in halves]), "players": ends(players_of(halves))}
    for what in ("ball", "players"):
        first, last, slow_first, slow_last, n_first, n_last = got[what]
        text = "   {:<8} begins ({:4d} tracks, {:3.0%} with a slow first step): {}".format(
            what, n_first, slow_first, line(first))
        if reference:
            text += "   | {:.3f} from the recording".format(apart(first, reference[what][0]))
        print(text)
        text = "   {:<8} ends   ({:4d} tracks, {:3.0%} with a slow last step):  {}".format(
            what, n_last, slow_last, line(last))
        if reference:
            text += "   | {:.3f} from the recording".format(apart(last, reference[what][1]))
        print(text)
    print("   a strong change of a player's velocity (3 m/s or more): " + width(players_of(halves), 3.0))
    print("   a strong change of the ball's velocity (8 m/s or more): " + width([h.ball_xy for h in halves], smoothing.STRONG))
    return got


def simulated(generation, condition, source):
    before = os.environ.get("VANTAGE_SIM")
    os.environ["VANTAGE_SIM"] = generation
    try:
        return [external.condition_half(half, match, condition)
                for match in external.ALL_MATCHES if external.source_of(match) == source
                for half in external.labelled_halves(match)]
    finally:
        if before is None:
            del os.environ["VANTAGE_SIM"]
        else:
            os.environ["VANTAGE_SIM"] = before


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="the competition data; without it only the simulator is read")
    parser.add_argument("--earlier", action="store_true", help="add the simulator's first two generations")
    args = parser.parse_args()
    print("speed k frames from an end of a track, over the speed 15 to 25 frames from that end; medians")
    reference = None
    if args.data_dir:
        reference = report("the recording", competition.load_halves(args.data_dir))
    names = {"g7": "simulated", "g5": "simulated, fifth generation", "g4": "simulated, fourth generation",
             "g3": "simulated, third generation", "g2": "simulated, second generation",
             "g1": "simulated, as first built"}
    for generation in ("g7", "g3", "g2", "g1") if args.earlier else ("g7",):
        for source, said in (("dfl", "the seven DFL matches"), ("metrica", "Metrica's three sample games")):
            for condition in ("typical", "random1"):
                report("{}, {}, {} setting".format(names[generation], said, condition),
                       simulated(generation, condition, source), reference)


if __name__ == "__main__":
    main()
