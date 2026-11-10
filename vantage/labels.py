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
