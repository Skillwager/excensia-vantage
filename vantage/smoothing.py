"""How smooth a recording's ball path is, and bringing a sharper path to the reference.

A kick changes the ball's velocity at once. A tracker that smooths its ball path spreads that
change over many frames, and how many differs from one tracking system to the next: the
recording this method was built for spreads it over 21 frames, another system's over a handful.
Everything the model reads off the ball's motion (touches, accelerations, the moment a pass
leaves a foot) looks different at different amounts of smoothing, so the models are trained at
one amount, the reference, and every recording is brought to it first. Smoothing can be added
and cannot be taken away, so the reference is the smoothest path met so far.

`spread` measures a path without any label: at every strong change of velocity, the share of the
change that already shows over eight frames. `to_reference` smooths a path that is sharper than
the reference with a growing moving average until that share reaches the reference, and leaves
one that is already as smooth alone.

Only exact arithmetic is used (sums, differences, divisions, square roots, sorting), so the
measurement and the decision are the same on every machine.
"""

from dataclasses import replace

import numpy as np

from .features import BALL_MAX_SPEED, BALL_VEL_HALFSPAN, _central_velocity

STRONG = 8.0            # m/s: a change of velocity over 24 frames that counts as strong
LONG, SHORT = 12, 4     # frames either side over which the whole change and its share are read
REACH = 6               # the share is the largest short change within this many frames of the peak
REFERENCE = 0.46        # a path whose median share is at or below this is as smooth as the reference
TARGET = 0.435          # a sharper path is smoothed until its share is at or below this, the
                        # reference recording's own value
WINDOWS = (5, 9, 13, 17, 19, 21, 23, 25)   # moving averages tried, in frames, smallest first
ENOUGH = 40             # strong changes needed before a path is judged at all
PROFILE_REACH = 20      # frames either side of a strong change's middle that `profile` covers


def moving_average(xy, window):
    """Centred moving average along axis 0 that ignores NaN and keeps NaN where the centre is."""
    if window <= 1:
        return xy
    half = window // 2
    finite = np.isfinite(xy[:, 0])
    filled = np.where(finite[:, None], xy, 0.0).astype(np.float64)
    total = np.concatenate((np.zeros((1, xy.shape[1])), np.cumsum(filled, axis=0)))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(xy))
    low = np.maximum(0, index - half)
    high = np.minimum(len(xy), index + half + 1)
    n = (count[high] - count[low]).astype(np.float64)
    out = (total[high] - total[low]) / np.maximum(n, 1.0)[:, None]
    out[~finite] = np.nan
    return out.astype(np.float32)


def _change(velocity, halfspan):
    t = len(velocity)
    out = np.full(t, np.nan, dtype=np.float64)
    if t > 2 * halfspan:
        delta = velocity[2 * halfspan:] - velocity[:-2 * halfspan]
        out[halfspan:-halfspan] = np.sqrt(np.sum(delta * delta, axis=1))
    return out


def _strong_changes(velocity):
    """(frames, strength): the frames where the change of velocity over LONG frames either side
    peaks at STRONG or more, and that change at every frame (0 where it cannot be read)."""
    whole = _change(velocity, LONG)
    strength = np.where(np.isfinite(whole), whole, 0.0)
    peak = strength >= STRONG
    for offset in range(1, LONG + 1):
        peak[offset:] &= strength[offset:] >= strength[:-offset]
        peak[:-offset] &= strength[:-offset] > strength[offset:]
    return np.flatnonzero(peak), strength


def spread(ball_xy):
    """(strong changes found, median share of each that shows over SHORT frames either side).

    The share is NaN when no strong change was found.
    """
    ball = np.asarray(ball_xy, dtype=np.float32)
    velocity = _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED).astype(np.float64)
    frames, strength = _strong_changes(velocity)
    if not len(frames):
        return 0, float("nan")
    part = _change(velocity, SHORT)
    filled = np.where(np.isfinite(part), part, -np.inf)
    best = filled.copy()
    for offset in range(1, REACH + 1):
        best[offset:] = np.maximum(best[offset:], filled[:-offset])
        best[:-offset] = np.maximum(best[:-offset], filled[offset:])
    share = best[frames] / strength[frames]
    share = share[np.isfinite(share)]
    if not len(share):
        return int(len(frames)), float("nan")
    return int(len(frames)), float(np.median(share))


def profile(ball_xy):
    """(total, count): how the strong changes of a path's velocity are spread over the frames
    around them. total / count is the average: the share of a change that falls in each of the
    PROFILE_REACH frames before its middle, the middle frame, and as many after.

    The whole change is the one `spread` reads: from the velocity LONG frames before the peak to
    the velocity LONG frames after it. Each frame's own change of velocity is projected on its
    direction and divided by its size, which makes it a share of the change; between those two
    frames the shares add up to one. The middle of a change is the first frame by which half of
    it has happened. The shares are lined up on that frame and added up over all strong
    changes. A change with a gap in the path nearby is left out.
    """
    ball = np.asarray(ball_xy, dtype=np.float32)
    velocity = _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED).astype(np.float64)
    frames, _ = _strong_changes(velocity)
    step = np.full(velocity.shape, np.nan)
    step[1:] = velocity[1:] - velocity[:-1]
    total = np.zeros(2 * PROFILE_REACH + 1)
    count = 0
    for frame in frames:
        if frame - LONG < 0 or frame + LONG >= len(ball):
            continue
        whole = velocity[frame + LONG] - velocity[frame - LONG]
        size = np.sqrt(np.sum(whole * whole))
        towards = whole / (size * size)
        inner = step[frame - LONG + 1:frame + LONG + 1] @ towards
        if not np.isfinite(inner).all():
            continue
        middle = frame - LONG + 1 + int(np.argmax(np.cumsum(inner) >= 0.5))
        low, high = middle - PROFILE_REACH, middle + PROFILE_REACH + 1
        if low < 1 or high > len(ball):
            continue
        share = step[low:high] @ towards
        if not np.isfinite(share).all():
            continue
        total += share
        count += 1
    return total, count


def for_model(half, card):
    """(half, window): the half as the card's models expect it. Models trained on paths at the
    reference smoothness say so in their card ("smoothing"); for any other card the half is
    returned as it is."""
    if not card.get("smoothing"):
        return half, 1
    return to_reference(half)


def to_reference(half):
    """(half, window): the half with its ball path as smooth as the reference, and the moving
    average that was applied (1 when the path was left as it is)."""
    count, share = spread(half.ball_xy)
    if count < ENOUGH or not share > REFERENCE:
        return half, 1
    ball = half.ball_xy
    window = 1
    for window in WINDOWS:
        ball = moving_average(half.ball_xy, window)
        _, share = spread(ball)
        if not share > TARGET:
            break
    return replace(half, ball_xy=ball.astype(np.float32),
                   meta=dict(half.meta, smoothed=window)), window
