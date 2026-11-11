"""Where the broadcast tracking sits on the tactical video's frames. Prints totals only.

    python scripts/broadcast_align.py --data-dir <the organisers' data> --step rough
    python scripts/broadcast_align.py --data-dir <...> --step exact

`rough` places each minute of broadcast records by the path of the ball (vantage/broadcast.py)
and says how many minutes found a place and at which shifts. `exact` checks every record
against the players and says how many records can be used and how many frames without a
tactical ball they would give a ball to.

No label and no video is read. The numbers printed are counts, shares and shifts.
"""

import argparse
import collections
import os
import time

import numpy as np

from vantage import broadcast, competition, epts
from vantage.external import ROOT

CACHE = os.path.join(ROOT, "data", "cache", "competition_two_trackings_v1.npz")
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def both(data_dir):
    """The two trackings in metres. Kept in data/cache (not in the repository) after the first read."""
    if os.path.isfile(CACHE):
        with np.load(CACHE) as data:
            return {key: data[key] for key in data.files}
    started = time.time()
    files = competition.tactical_paths(data_dir)
    tactical = epts.read(files["meta"], files["raw"])
    second = broadcast.load(data_dir, competition.PITCH)
    out = {"t_xy": broadcast.in_metres(tactical.xy, competition.PITCH), "t_team": tactical.team,
           "t_ball": broadcast.in_metres(tactical.ball, competition.PITCH),
           "b_xy": second["xy"], "b_team": second["team"], "b_ball": second["ball"]}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print("read both trackings in {:.0f}s".format(time.time() - started), flush=True)
    return out


def rough(data, args):
    t_ball, b_ball = data["t_ball"], data["b_ball"]
    print("tactical frames {}, with a ball {:.1%}; broadcast records {}, with a ball {:.1%}".format(
        len(t_ball), float(np.isfinite(t_ball[:, 0]).mean()), len(b_ball), float(np.isfinite(b_ball[:, 0]).mean())))
    # which way round the broadcast pitch is: tried on every eighth minute
    trial = {}
    for flip in FLIPS:
        placed = broadcast.rough_shifts(t_ball, broadcast.flips(b_ball, flip), step=broadcast.STEP * 8)
        good = [row for row in placed if row[2] < 3.0 and row[3] > 1.5 * row[2]]
        trial[flip] = (len(good), len(placed), float(np.median([row[2] for row in placed if np.isfinite(row[2])])))
        print("  pitch turned (x, y) = {}: {} of {} minutes placed; middle distance at the best shift {:.1f} m".format(
            flip, *trial[flip]), flush=True)
    flip = max(trial, key=lambda key: trial[key][0])
    placed = broadcast.rough_shifts(t_ball, broadcast.flips(b_ball, flip))
    good = [row for row in placed if row[2] < 3.0 and row[3] > 1.5 * row[2]]
    print("with the pitch turned {}: {} of {} minutes placed (ball paths within 3 m, and no other place half as good)".format(
        flip, len(good), len(placed)))
    offsets = collections.Counter(int(round((row[1] - row[0]) / 5.0)) * 5 for row in good)
    print("  the shift (tactical frame minus broadcast record) they found, and how many minutes found it:")
    for offset, count in sorted(offsets.items()):
        print("    {:>8}  {}".format(offset, count))
    distance = np.array([row[2] for row in good])
    print("  ball paths at those places: middle {:.2f} m apart, nine in ten under {:.2f} m".format(
        float(np.median(distance)), float(np.percentile(distance, 90))))
    lost = [row for row in placed if not (row[2] < 3.0 and row[3] > 1.5 * row[2])]
    print("  minutes not placed: {} (of them {} had too few frames with both balls)".format(
        len(lost), sum(1 for row in lost if not np.isfinite(row[2]))))
    np.savez(os.path.join(os.path.dirname(CACHE), "competition_rough_shifts_v1.npz"),
             placed=np.array([[row[0], row[1], row[2], row[3], row[4]] for row in placed], dtype=np.float64),
             flip=np.array(flip))


def runs(flag):
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def exact(data, args):
    saved = np.load(os.path.join(os.path.dirname(CACHE), "competition_rough_shifts_v1.npz"))
    placed, flip = saved["placed"], tuple(bool(v) for v in saved["flip"])
    b_xy, b_ball = broadcast.flips(data["b_xy"], flip), broadcast.flips(data["b_ball"], flip)
    t_xy, t_ball, t_team, b_team = data["t_xy"], data["t_ball"], data["t_team"], data["b_team"]
    good = placed[(placed[:, 2] < 3.0) & (placed[:, 3] > 1.5 * placed[:, 2])]
    offsets = collections.Counter(int(round((row[1] - row[0]) / 25.0)) * 25 for row in good)
    # the shifts that several minutes agree on, each made exact on the players
    shifts = []
    for offset, count in sorted(offsets.items()):
        if count < 3:
            continue
        shift, shares = broadcast.best_shift(t_xy, t_team, b_xy, b_team, offset)
        shifts.append(shift)
        print("near {}: most players have a partner at shift {} ({:.1%}); one frame either side {:.1%} and {:.1%}, "
              "ten frames either side {:.1%} and {:.1%}".format(
                  offset, shift, shares[shift], shares.get(shift - 1, 0), shares.get(shift + 1, 0),
                  shares.get(shift - 10, 0), shares.get(shift + 10, 0)), flush=True)
    frame, displacement = broadcast.place_records(t_xy, t_team, b_xy, b_team, shifts)
    kept = frame >= 0
    print("broadcast records kept: {} of {} ({:.1%})".format(int(kept.sum()), len(kept), float(kept.mean())))
    for shift in shifts:
        own = kept & (frame - np.arange(len(frame)) == shift)
        stretches = runs(own)
        print("  at shift {}: {} records in {} stretches, the middle stretch {:.0f} s long".format(
            shift, int(own.sum()), len(stretches),
            float(np.median([b - a for a, b in stretches])) / 25.0 if stretches else 0.0))
    size = np.linalg.norm(displacement[kept], axis=1)
    print("  the players of a kept record are displaced by {:.2f} m in the middle, nine in ten under {:.2f} m".format(
        float(np.median(size)), float(np.percentile(size, 90))))

    # what the broadcast ball would add
    seen_t = np.isfinite(t_ball[:, 0])
    seen_b = np.isfinite(b_ball[:, 0])
    graded = np.zeros(len(t_ball), dtype=bool)
    for first, last in competition.GRADED.values():
        graded[first:last + 1] = True
    rows = np.flatnonzero(kept)
    target = frame[rows]
    covered = np.zeros(len(t_ball), dtype=bool)
    covered[target] = True
    print("tactical frames a kept record shows: {:.1%} of all, {:.1%} of the graded ones".format(
        float(covered.mean()), float(covered[graded].mean())))
    has_b = np.zeros(len(t_ball), dtype=bool)
    has_b[target[seen_b[rows]]] = True
    both_seen = graded & seen_t & has_b
    only_b = graded & ~seen_t & has_b
    print("graded frames: {}; tactical ball on {:.1%}; a kept broadcast record with a ball on {:.1%}".format(
        int(graded.sum()), float(seen_t[graded].mean()), float(has_b[graded].mean())))
    print("  both: {:.1%}; only the broadcast: {:.1%}; so a ball on {:.1%} of graded frames with both, against {:.1%}".format(
        float(both_seen[graded].mean()), float(only_b[graded].mean()),
        float((seen_t | has_b)[graded].mean()), float(seen_t[graded].mean())))
    # how well the two balls agree where both are seen, before and after moving the broadcast
    # ball by its record's displacement
    pairs = rows[seen_b[rows] & seen_t[target] & graded[target]]
    apart = np.linalg.norm(b_ball[pairs] - t_ball[frame[pairs]], axis=1)
    moved = np.linalg.norm(b_ball[pairs] + displacement[pairs] - t_ball[frame[pairs]], axis=1)
    for name, values in (("as recorded", apart), ("moved with the players", moved)):
        print("  the two balls, {}: middle {:.2f} m apart, three in four under {:.2f}, nine in ten under {:.2f}, "
              "over 5 m {:.1%}".format(name, float(np.median(values)), float(np.percentile(values, 75)),
                                       float(np.percentile(values, 90)), float((values > 5).mean())))
    # the gaps in the tactical ball: how many are filled, wholly or partly
    filled_whole = filled_part = untouched = 0
    lengths_filled = []
    for a, b in runs(graded & ~seen_t):
        share = float(has_b[a:b].mean())
        if share > 0.9:
            filled_whole += 1
        elif share > 0.0:
            filled_part += 1
        else:
            untouched += 1
        lengths_filled.append((b - a, share))
    lengths_filled = np.array(lengths_filled)
    print("  stretches without a tactical ball: {}; given a ball throughout {}, in part {}, not at all {}".format(
        len(lengths_filled), filled_whole, filled_part, untouched))
    for low, high in ((0, 25), (25, 75), (75, 250), (250, 10 ** 9)):
        band = (lengths_filled[:, 0] >= low) & (lengths_filled[:, 0] < high)
        if band.any():
            print("    {:>4}-{:<5} frames long: {:>5} stretches, {:.0%} of their frames given a ball".format(
                low, high if high < 10 ** 9 else "", int(band.sum()),
                float((lengths_filled[band, 0] * lengths_filled[band, 1]).sum() / lengths_filled[band, 0].sum())))
    np.savez(os.path.join(os.path.dirname(CACHE), "competition_placed_records_v1.npz"),
             frame=frame, displacement=displacement, shifts=np.array(shifts), flip=np.array(flip))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--step", choices=("rough", "exact"), default="rough")
    args = parser.parse_args()
    data = both(args.data_dir)
    if args.step == "rough":
        rough(data, args)
    else:
        exact(data, args)


if __name__ == "__main__":
    main()
