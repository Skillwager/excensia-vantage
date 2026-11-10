"""Bring another provider's possession flags to the challenge's definition of possession.

The open DFL data marks a change of possession on every touch by the other team: a blocked pass,
a header won in a duel and a tackle that pokes the ball away each start a new "possession", many
of them under two seconds long. The challenge is explicit that these do not count
(CHALLENGE.md, "What does not count as a change of possession"): possession changes only when the
other team gains *controlled* possession, for which it offers "a player keeps the ball for a few
seconds" as a proxy, and "in phases where the ball is contested or loose, possession stays with
the last team that had it".

`harmonise` applies exactly that to a label sequence: inside each stretch of live play, a spell
shorter than the control threshold is given back to the team that had the ball before it. The
first spell after a restart is always kept, because a restart has a clear possessor by rule.
Dead-ball frames are never changed.
"""

import numpy as np

from . import A, D, FPS, H

# The control threshold, per source of labels.
#
# "metrica": two seconds, the challenge's own proxy ("keeps the ball for a few seconds") applied
#   to the provider whose corrected data the ground truth was derived from.
# "dfl": three seconds. DFL's flag flips on more touches than Metrica's events do, so the same
#   two seconds would leave DFL with more possessions per minute of live play than Metrica has.
#   Three seconds brings the two external sources level (about 4.5 possessions per live minute
#   each; scripts/label_stats.py and scripts/metrica_stats.py print the numbers). This is one
#   external dataset calibrated against another; the competition match is not involved.
CONTROL_SECONDS = {"metrica": 2.0, "dfl": 3.0}


def segments(labels):
    """[(start, stop, code)] constant runs, half-open."""
    if len(labels) == 0:
        return []
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def harmonise(labels, control_seconds, fps=FPS):
    """Copy of `labels` with uncontrolled spells merged into the possession before them."""
    minimum = int(round(control_seconds * fps))
    out = labels.copy()
    owner = None            # team holding the ball in the current stretch of live play
    for start, stop, code in segments(labels):
        if code not in (H, A):
            owner = None    # a dead ball (or an ungraded gap) ends the stretch
            continue
        if owner is None:
            owner = code    # restart: clear possessor, kept whatever its length
        elif code != owner:
            if stop - start >= minimum:
                owner = code
            else:
                out[start:stop] = owner
    return out


def spell_stats(labels, fps=FPS):
    """(count, mean seconds, median seconds, share under 2 s) of the possessions in `labels`."""
    lengths = np.array([stop - start for start, stop, code in segments(labels)
                        if code in (H, A)], dtype=np.float64) / fps
    if not len(lengths):
        return 0, 0.0, 0.0, 0.0
    return len(lengths), float(lengths.mean()), float(np.median(lengths)), float(
        np.mean(lengths < 2.0))


# Kinds of change between consecutive frames, as used by the boundary model.
NO_CHANGE, TURNOVER, STOP, RESTART = 0, 1, 2, 3
DISTANCE_CAP = 250


def change_distances(labels):
    """(3, T) int16: frames from each frame to the nearest turnover, stoppage start and restart.

    A change sits between two frames; both of them are at distance 0. Distances are capped at
    DISTANCE_CAP. Changes into or out of an ungraded stretch are not changes of play and are
    ignored.
    """
    t = len(labels)
    out = np.full((3, t), DISTANCE_CAP, dtype=np.int32)
    if t < 2:
        return out.astype(np.int16)
    before, after = labels[:-1], labels[1:]
    alive_before = (before == H) | (before == A)
    alive_after = (after == H) | (after == A)
    kinds = {
        TURNOVER: alive_before & alive_after & (before != after),
        STOP: alive_before & (after == D),
        RESTART: (before == D) & alive_after,
    }
    index = np.arange(t)
    for kind, flag in kinds.items():
        positions = np.flatnonzero(flag)          # change between frame p and p + 1
        if not len(positions):
            continue
        marks = np.zeros(t, dtype=bool)
        marks[positions] = True
        marks[positions + 1] = True
        last = np.maximum.accumulate(np.where(marks, index, -DISTANCE_CAP * 2))
        nxt = np.minimum.accumulate(np.where(marks, index, t + DISTANCE_CAP * 2)[::-1])[::-1]
        out[kind - 1] = np.minimum(np.minimum(index - last, nxt - index), DISTANCE_CAP)
    return out.astype(np.int16)


def change_classes(distances, width):
    """Per-frame class for the boundary model: the kind of change within `width` frames, if any.

    Where two kinds are both within reach the nearer one wins.
    """
    nearest = np.argmin(distances, axis=0)
    reach = np.min(distances, axis=0) <= width
    return np.where(reach, nearest + 1, NO_CHANGE).astype(np.int8)
