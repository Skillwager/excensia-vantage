"""Where do the lost points go? Held-out external matches, one kind of error repaired at a time.

Reads the probabilities a cross-validation run saved, decodes them with one setting, and then
repairs one kind of error with the truth and scores again. The gain of each repair is the size
of that kind of error, in the ranking metric itself:

    stoppages     every frame that is dead ball in the truth is made dead ball, and every frame
                  that is not is given to a team (the nearest predicted one)
    team          on frames both call live, the team is made the true one
    timing        every predicted change that has a true change of the same kind within a second
                  is moved onto it
    short         every true possession under three seconds is written in, and every predicted
                  one under three seconds that matches nothing is taken out

    python scripts/error_budget.py --tag v14six \
        --decode '{"cost_team": 2.0, "cost_to_dead": 14.0, "cost_from_dead": 14.0, "boundary_weight": 12.0, "min_possession": 0.2, "min_dead": 0.5}'

External matches only. Nothing here can be run on the competition match: it needs the truth.
"""

import argparse
import json
import os

import numpy as np

from vantage import A, D, H, X, evaluate
from vantage import decode as decoding
from vantage.external import ALL_MATCHES, CONDITIONS, ROOT, source_of
from vantage.labels import segments

SHORT = 75          # frames: three seconds
REACH = 25          # frames: a predicted change is moved onto a true one this near


def load(tag, match, condition):
    path = os.path.join(ROOT, "runs", "cv", tag, "prob_{}_{}.npz".format(match, condition))
    if not os.path.isfile(path):
        return None
    with np.load(path) as data:
        halves, index = [], 0
        while "y{}".format(index) in data:
            boundary = (data["b{}".format(index)].astype(np.float64)
                        if "b{}".format(index) in data else None)
            halves.append((data["y{}".format(index)], data["p{}".format(index)].astype(np.float64),
                           boundary))
            index += 1
    return halves


def repair_stoppages(predicted, truth, prob):
    out = predicted.copy()
    live = (truth == H) | (truth == A)
    out[truth == D] = D
    edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        window = out[start:stop].copy()
        known = np.flatnonzero(window != D)
        if not len(known):
            out[start:stop] = H if prob[start:stop, H].mean() >= prob[start:stop, A].mean() else A
            continue
        carry = np.maximum.accumulate(np.where(window != D, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = window[carry]
    return out


def repair_team(predicted, truth):
    out = predicted.copy()
    both = ((truth == H) | (truth == A)) & ((predicted == H) | (predicted == A))
    out[both] = truth[both]
    return out


def changes(labels):
    index = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return [(int(i), int(labels[i - 1]), int(labels[i])) for i in index
            if labels[i - 1] != X and labels[i] != X]


def pair_changes(predicted, truth, reach):
    """Predicted and true changes of the same kind, paired nearest first, one to one."""
    ours, theirs = changes(predicted), changes(truth)
    frames = np.array([change[0] for change in theirs]) if theirs else np.zeros(0, dtype=int)
    candidates = []
    for i, (frame, before, after) in enumerate(ours):
        low = np.searchsorted(frames, frame - reach)
        high = np.searchsorted(frames, frame + reach, side="right")
        for j in range(low, high):
            if theirs[j][1] == before and theirs[j][2] == after:
                candidates.append((abs(frame - theirs[j][0]), i, j))
    candidates.sort()
    used_ours, used_theirs, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_ours or j in used_theirs:
            continue
        used_ours.add(i)
        used_theirs.add(j)
        pairs.append((i, j))
    return ours, theirs, pairs


def repair_timing(predicted, truth, reach=REACH):
    ours, theirs, pairs = pair_changes(predicted, truth, reach)
    at = [int(frame) for frame in np.flatnonzero(predicted[1:] != predicted[:-1]) + 1]
    place = {frame: index for index, frame in enumerate(at)}
    moved = list(at)
    for i, j in pairs:
        moved[place[ours[i][0]]] = theirs[j][0]
    bounds, original = [0] + moved + [len(predicted)], [0] + at + [len(predicted)]
    for _ in range(10):                 # a move that would cross a neighbour is undone
        undone = False
        for k in range(1, len(bounds) - 1):
            if not bounds[k - 1] < bounds[k] < bounds[k + 1] and bounds[k] != original[k]:
                bounds[k] = original[k]
                undone = True
        if not undone:
            break
    codes = [int(predicted[0])] + [int(predicted[frame]) for frame in at]
    out = predicted.copy()
    for k, code in enumerate(codes):
        if bounds[k + 1] > bounds[k]:
            out[bounds[k]:bounds[k + 1]] = code
    return out


def best_overlap(start, stop, code, spells):
    best = 0.0
    for other_start, other_stop, other_code in spells:
        if other_code != code or other_stop <= start or stop <= other_start:
            continue
        best = max(best, (min(stop, other_stop) - max(start, other_start))
                   / float(max(stop, other_stop) - min(start, other_start)))
    return best


def repair_short(predicted, truth, limit=SHORT):
    out = predicted.copy()
    true_spells = [(a, b, k) for a, b, k in segments(truth) if k in (H, A)]
    for start, stop, code in [(a, b, k) for a, b, k in segments(predicted) if k in (H, A)]:
        if stop - start < limit and best_overlap(start, stop, code, true_spells) < 0.3:
            out[start:stop] = out[start - 1] if start > 0 else (out[stop] if stop < len(out) else code)
    for start, stop, code in true_spells:
        if stop - start < limit:
            out[start:stop] = code
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--decode", required=True, help="JSON of decoding settings")
    args = parser.parse_args()
    config = decoding.DecodeConfig(**json.loads(args.decode))
    repairs = ("as decoded", "stoppages", "team", "timing", "short", "short and timing")
    table = {}
    for match in ALL_MATCHES:
        for condition in CONDITIONS:
            halves = load(args.tag, match, condition)
            if halves is None:
                continue
            truths = [half[0] for half in halves]
            decoded = [decoding.decode(prob, config, boundary) for _, prob, boundary in halves]
            made = {
                "as decoded": decoded,
                "stoppages": [repair_stoppages(p, t, half[1]) for p, t, half in zip(decoded, truths, halves)],
                "team": [repair_team(p, t) for p, t in zip(decoded, truths)],
                "timing": [repair_timing(p, t) for p, t in zip(decoded, truths)],
                "short": [repair_short(p, t) for p, t in zip(decoded, truths)],
            }
            made["short and timing"] = [repair_timing(p, t) for p, t in zip(made["short"], truths)]
            for name in repairs:
                table.setdefault((source_of(match), condition, name), []).append(
                    evaluate.quick(truths, made[name])["spellF1"])
    conditions = [c for c in CONDITIONS if any(key[1] == c for key in table)]
    print("tag {}, decoding {}".format(args.tag, json.dumps(json.loads(args.decode))))
    print("spellF1, mean over held-out matches, after repairing one kind of error with the truth")
    print("{:<26}".format("") + "".join("{:>18}".format(name) for name in repairs))
    for source in ("metrica", "dfl"):
        rows = [c for c in conditions if (source, c, repairs[0]) in table]
        for condition in rows:
            print("{:<26}".format("{} {}".format(source, condition)) + "".join(
                "{:>18.3f}".format(float(np.mean(table[(source, condition, name)]))) for name in repairs))
        simulated = [c for c in rows if c != "clean"]
        if simulated:
            print("{:<26}".format("{} simulated, mean".format(source)) + "".join(
                "{:>18.3f}".format(float(np.mean([np.mean(table[(source, c, name)]) for c in simulated])))
                for name in repairs))


if __name__ == "__main__":
    main()
