"""How much the broadcast ball would change the labels, before deciding whether to use it.

    python scripts/broadcast_align.py --data-dir <data> --step rough
    python scripts/broadcast_align.py --data-dir <data> --step exact
    python scripts/broadcast_try.py --data-dir <data>

The match is labelled twice with the shipped models: once as it is, once with the tactical
tracking's ball filled in from the broadcast tracking wherever it has none and a placed broadcast
record has one. Prints how many frames and possessions differ. No label and no video is read;
the output is counts only, and nothing is written into the submission.
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np

from vantage import A, D, H, X, broadcast, cli, competition
from vantage.external import ROOT
from vantage.labels import segments

CACHE = os.path.join(ROOT, "data", "cache")
NAMES = {H: "home", A: "away", D: "dead"}


def filled_ball(margin=2.0):
    """(tactical ball with the broadcast ball where it had none, which frames were filled)."""
    with np.load(os.path.join(CACHE, "competition_two_trackings_v1.npz")) as data:
        t_ball, b_ball = data["t_ball"], data["b_ball"]
    with np.load(os.path.join(CACHE, "competition_placed_records_v1.npz")) as data:
        frame, displacement, flip = data["frame"], data["displacement"], tuple(bool(v) for v in data["flip"])
    b_ball = broadcast.flips(b_ball, flip)
    out = t_ball.copy()
    filled = np.zeros(len(t_ball), dtype=bool)
    half_l, half_w = competition.PITCH[0] / 2, competition.PITCH[1] / 2
    for record in np.flatnonzero((frame >= 0) & np.isfinite(b_ball[:, 0])):
        target = frame[record]
        if np.isfinite(out[target, 0]):
            continue
        place = b_ball[record] + displacement[record]
        if abs(place[0]) > half_l + margin or abs(place[1]) > half_w + margin:
            continue
        out[target] = place
        filled[target] = True
    return out, filled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default=cli.DEFAULT_MODEL)
    args = parser.parse_args()
    job = SimpleNamespace(model=args.model, data_dir=args.data_dir, no_tagging=False)
    before = cli.analyse(job)[0]

    ball, filled = filled_ball()
    load_halves = competition.load_halves

    def with_broadcast_ball(data_dir):
        halves = load_halves(data_dir)
        for half in halves:
            rows = slice(half.first_frame, half.first_frame + half.n_frames)
            add = filled[rows]
            patch = ball[rows][add]
            half.ball_xy[add] = -patch if half.meta.get("flipped") else patch
        return halves

    competition.load_halves = with_broadcast_ball
    try:
        after = cli.analyse(job)[0]
    finally:
        competition.load_halves = load_halves

    graded = before != X
    print("graded frames {}; given a broadcast ball {} ({:.1%})".format(
        int(graded.sum()), int((filled & graded).sum()), float(filled[graded].mean())))
    changed = graded & (before != after)
    print("frames whose label changes: {} ({:.2%} of graded); of them on a filled frame {:.0%}".format(
        int(changed.sum()), float(changed[graded].mean()),
        float(filled[changed].mean()) if changed.any() else 0.0))
    for a in (H, A, D):
        for b in (H, A, D):
            if a != b:
                count = int((graded & (before == a) & (after == b)).sum())
                if count:
                    print("  {} -> {}: {} frames".format(NAMES[a], NAMES[b], count))
    for name, labels in (("as it is", before), ("with the broadcast ball", after)):
        runs = [(stop - start, code) for start, stop, code in segments(labels) if code in (H, A, D)]
        possessions = [length for length, code in runs if code in (H, A)]
        stoppages = [length for length, code in runs if code == D]
        print("{:<24} possessions {}  stoppages {}  dead {:.1%} of graded frames".format(
            name, len(possessions), len(stoppages), float((labels[graded] == D).mean())))
    # stretches of changed frames: how many, how long
    edges = np.diff(np.concatenate(([0], changed.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    if len(lengths):
        print("the changes come in {} stretches: middle {:.1f} s, longest {:.1f} s".format(
            len(lengths), float(np.median(lengths)) / 25.0, float(lengths.max()) / 25.0))


if __name__ == "__main__":
    main()
