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
REACH = LONG            # the share is the largest short change within this many frames of the peak:
                        # on a sharp path the whole change is a plateau about that wide, and the
                        # peak found on it may be either end
REFERENCE = 0.46        # a path whose median share is at or below this is as smooth as the reference
TARGET = 0.435          # a sharper path is smoothed until its share is at or below this, just
                        # above the reference recording's own 0.42
WINDOWS = (5, 9, 13, 17, 19, 21, 23, 25)   # moving averages tried, in frames, smallest first
ENOUGH = 40             # strong changes needed before a path is judged at all
HOLD = 5                # frames the reference tracker still reports a lost ball or player where it
                        # last saw it (scripts/edge_report.py)
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


def track_average(xy, window, hold=0):
    """Centred moving average inside each unbroken track, the way the reference tracker treats
    the two ends of a track (read off its recording without labels, scripts/edge_report.py).

    Where a track begins it keeps its speed: the track is carried on backwards by reflecting it
    through its first point, so a straight path stays where it is. Where a track ends the
    tracker goes on reporting the last place for `hold` frames and the average takes that place
    for everything after, so every track slows to a stop over its last frames. Nothing outside
    the track is used. The held frames are part of the track that is returned; a track that is
    followed within `hold` frames by the next one is held only up to it.
    """
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    there = np.isfinite(xy[:, 0])
    frames = len(xy)
    edges = np.diff(np.concatenate(([0], there.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()):
        until = stop
        while until < min(frames, stop + hold) and not there[until]:
            until += 1
        part = np.concatenate((xy[start:stop].astype(np.float64),
                               np.repeat(xy[stop - 1:stop].astype(np.float64), until - stop, axis=0)))
        n = len(part)
        if window <= 1 or n < 2:
            out[start:until] = part.astype(np.float32)
            continue
        reach = min(window // 2, n - 1)
        head = 2.0 * part[0] - part[reach:0:-1]                       # reflected through the first point
        tail = np.repeat(part[-1:], reach, axis=0)                    # the last place, held
        longer = np.concatenate((head, part, tail))
        total = np.concatenate((np.zeros((1, xy.shape[1])), np.cumsum(longer, axis=0)))
        out[start:until] = ((total[2 * reach + 1:] - total[:-2 * reach - 1]) / (2 * reach + 1)).astype(np.float32)
    return out


def _change(velocity, halfspan):
    t = len(velocity)
    out = np.full(t, np.nan, dtype=np.float64)
    if t > 2 * halfspan:
        delta = velocity[2 * halfspan:] - velocity[:-2 * halfspan]
        out[halfspan:-halfspan] = np.sqrt(np.sum(delta * delta, axis=1))
    return out


def _strong_changes(velocity, strong=STRONG):
    """(frames, strength): the frames where the change of velocity over LONG frames either side
    peaks at `strong` or more, and that change at every frame (0 where it cannot be read)."""
    whole = _change(velocity, LONG)
    strength = np.where(np.isfinite(whole), whole, 0.0)
    peak = strength >= strong
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


def _along(steps, direction):
    """Each step's part along a direction, written out so that no linear-algebra library is used."""
    return steps[:, 0] * direction[0] + steps[:, 1] * direction[1]


def profile(ball_xy, strong=STRONG):
    """(total, count): how the strong changes of a path's velocity are spread over the frames
    around them. total / count is the average: the share of a change that falls in each of the
    PROFILE_REACH frames before its middle, the middle frame, and as many after. `strong` is
    the change that counts, in m/s: 8 for a ball, and 3 reads a player's path the same way.

    The whole change is the one `spread` reads: from the velocity LONG frames before the peak to
    the velocity LONG frames after it. Each frame's own change of velocity is projected on its
    direction and divided by its size, which makes it a share of the change; between those two
    frames the shares add up to one. The middle of a change is the first frame by which half of
    it has happened. The shares are lined up on that frame and added up over all strong
    changes. A change with a gap in the path nearby is left out.
    """
    ball = np.asarray(ball_xy, dtype=np.float32)
    velocity = _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED).astype(np.float64)
    frames, _ = _strong_changes(velocity, strong)
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
        inner = _along(step[frame - LONG + 1:frame + LONG + 1], towards)
        if not np.isfinite(inner).all():
            continue
        middle = frame - LONG + 1 + int(np.argmax(np.cumsum(inner) >= 0.5))
        low, high = middle - PROFILE_REACH, middle + PROFILE_REACH + 1
        if low < 1 or high > len(ball):
            continue
        share = _along(step[low:high], towards)
        if not np.isfinite(share).all():
            continue
        total += share
        count += 1
    return total, count


def half_width(share):
    """For how many frames an averaged change (`profile`) stays at half its height or more, read
    between frames. A centred moving average of n frames gives a width of about n."""
    share = np.asarray(share, dtype=np.float64)
    top = int(np.argmax(share))
    half = 0.5 * share[top]
    left = top
    while left > 0 and share[left - 1] >= half:
        left -= 1
    right = top
    while right < len(share) - 1 and share[right + 1] >= half:
        right += 1
    width = float(right - left)
    if left > 0:
        width += (share[left] - half) / (share[left] - share[left - 1])
    if right < len(share) - 1:
        width += (share[right] - half) / (share[right] - share[right + 1])
    return width


def third_difference(balls, lags=range(1, 46)):
    """{lag: correlation of the path's third difference with itself that many frames apart}.

    An exact, centred moving average of n frames turns a kick into a constant acceleration that
    lasts n frames, so the third difference of the path is two opposite spikes n frames apart and
    correlates strongly and negatively with itself at a lag of exactly n and at no other. A
    filter that is not an exact average spreads that mark over several lags.
    """
    lags = list(lags)
    top = np.zeros(len(lags))
    bottom = 0.0
    for ball in balls:
        ball = np.asarray(ball, dtype=np.float64)
        third = ball[3:] - 3.0 * ball[2:-1] + 3.0 * ball[1:-2] - ball[:-3]
        there = np.isfinite(third[:, 0])
        third = np.where(there[:, None], third, 0.0)
        bottom += float(np.sum(third * third))
        for index, lag in enumerate(lags):
            both = there[lag:] & there[:-lag]
            top[index] += float(np.sum((third[lag:] * third[:-lag])[both]))
    return {lag: float(value / bottom) if bottom else float("nan") for lag, value in zip(lags, top)}


def for_model(half, card):
    """(half, window): the half as the card's models expect it. Models trained on paths at the
    reference smoothness say so in their card ("smoothing"); for any other card the half is
    returned as it is."""
    wanted = card.get("smoothing")
    if not wanted:
        return half, 1
    # the card's own numbers: the models and the step that feeds them cannot drift apart
    return to_reference(half, wanted.get("reference", REFERENCE), wanted.get("target", TARGET),
                        tuple(wanted.get("windows", WINDOWS)), wanted.get("hold", HOLD))


def to_reference(half, reference=REFERENCE, target=TARGET, windows=WINDOWS, hold=HOLD):
    """(half, window): the half with its ball path as smooth as the reference, and the moving
    average that was applied (1 when the path was left as it is).

    The average is the reference tracker's own (`track_average`): a track keeps its speed where
    it begins and is held `hold` frames where it ends. A half with fewer than ENOUGH strong
    changes cannot be judged and is left as it is.
    """
    count, share = spread(half.ball_xy)
    if count < ENOUGH or not share > reference:
        return half, 1
    ball = half.ball_xy
    window = 1
    for window in windows:
        ball = track_average(half.ball_xy, window, hold)
        _, share = spread(ball)
        if not share > target:
            break
    return replace(half, ball_xy=ball.astype(np.float32),
                   meta=dict(half.meta, smoothed=window)), window
