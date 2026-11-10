"""The external possession labels, and how they relate to the challenge's definition.

The challenge says possession changes when the other team gains *controlled* possession, and
offers "a player keeps the ball for a few seconds" as a proxy (CHALLENGE.md). It also says that
the text is not the specification: the ground truth is taken mechanically from corrected
tracking, frame by frame, with nothing smoothed, and "if your reading of the definition disagrees
with it, the ground truth is right". The public five minutes settle one question of reading: they
contain possessions of 0.8 and 1.3 seconds in the middle of live play. So the ground truth has no
minimum length, and neither do the labels used here. Both external sources are used as given:

* Metrica's sample games: the team of the latest on-ball event (vantage/metrica_sample.py).
  Challenges and deflections are not on-ball events, which is the challenge's own list of touches
  that do not count.
* The open DFL data: the provider's per-frame possession flag.

`harmonise` merges spells shorter than a threshold into the possession before them. With the
threshold at zero it changes nothing. It is kept because scripts/label_stats.py uses it to show
what a minimum length would do to either source, and because an earlier version of this method
trained on merged labels (two seconds for Metrica, three for DFL) until the reading above
corrected it. No number here was fitted to the public five minutes.
"""

import numpy as np

from . import A, D, FPS, H

# Minimum length of a possession, per source of labels: none. See the note above.
CONTROL_SECONDS = {"metrica": 0.0, "dfl": 0.0}

# The thresholds of the earlier, merged labels, kept so that the two can be compared.
MERGED_SECONDS = {"metrica": 2.0, "dfl": 3.0}


def segments(labels):
    """[(start, stop, code)] constant runs, half-open."""
    if len(labels) == 0:
        return []
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def harmonise(labels, control_seconds, fps=FPS):
    """Copy of `labels` with spells shorter than `control_seconds` merged into the possession
    before them. The first spell after a restart is always kept; dead-ball frames never change."""
    minimum = int(round(control_seconds * fps))
    if minimum <= 0:
        return labels.copy()
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


def run_lengths(labels):
    """Per frame, the length in frames of the constant run it belongs to."""
    out = np.zeros(len(labels), dtype=np.int32)
    for start, stop, _ in segments(labels):
        out[start:stop] = stop - start
    return out


def spell_stats(labels, fps=FPS):
    """(count, mean seconds, median seconds, share under 2 s) of the possessions in `labels`."""
    lengths = np.array([stop - start for start, stop, code in segments(labels)
                        if code in (H, A)], dtype=np.float64) / fps
    if not len(lengths):
        return 0, 0.0, 0.0, 0.0
    return len(lengths), float(lengths.mean()), float(np.median(lengths)), float(
        np.mean(lengths < 2.0))


def behaviour(halves, fps=FPS):
    """Totals that describe a labelling without any reference labels.

    `halves` is a list of (labels, seen): the labels of one half and, optionally, a boolean mask
    of the frames on which the ball was tracked. Ungraded frames (any code other than H, A, D) are
    left out of every total. The same function describes a decoded external match, its true
    labels and the competition file, so the three can be laid side by side.
    """
    spells, stops = [], []
    frames = {H: 0, A: 0, D: 0}
    hidden = hidden_dead = visible = visible_dead = 0
    with_ball = False
    for labels, seen in halves:
        for code in frames:
            frames[code] += int(np.sum(labels == code))
        for start, stop, code in segments(labels):
            if code in (H, A):
                spells.append((stop - start) / fps)
            elif code == D:
                stops.append((stop - start) / fps)
        if seen is not None:
            with_ball = True
            graded = (labels == H) | (labels == A) | (labels == D)
            dead = labels == D
            hidden += int(np.sum(graded & ~seen))
            hidden_dead += int(np.sum(dead & ~seen))
            visible += int(np.sum(graded & seen))
            visible_dead += int(np.sum(dead & seen))
    live = frames[H] + frames[A]
    graded = live + frames[D]
    spells, stops = np.array(spells), np.array(stops)
    out = {
        "graded": graded,
        "home": frames[H] / max(graded, 1),
        "away": frames[A] / max(graded, 1),
        "dead": frames[D] / max(graded, 1),
        "home_of_live": frames[H] / max(live, 1),
        "possessions": len(spells),
        "per_live_minute": len(spells) / max(live / (60.0 * fps), 1e-9),
        "possession_mean": float(spells.mean()) if len(spells) else 0.0,
        "possession_median": float(np.median(spells)) if len(spells) else 0.0,
        "possession_under3": float(np.mean(spells < 3.0)) if len(spells) else 0.0,
        "stoppages": len(stops),
        "stoppage_mean": float(stops.mean()) if len(stops) else 0.0,
        "stoppage_median": float(np.median(stops)) if len(stops) else 0.0,
    }
    if with_ball:
        out.update({
            "hidden": hidden / max(graded, 1),
            "hidden_called_dead": hidden_dead / max(hidden, 1),
            "visible_called_dead": visible_dead / max(visible, 1),
            "dead_while_hidden": hidden_dead / max(frames[D], 1),
        })
    return out


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
