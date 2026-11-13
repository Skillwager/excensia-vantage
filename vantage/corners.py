"""Putting back the corners a tracker's smoothing cuts off the ball's path.

A tracker that averages the ball's path over about 21 frames cuts every corner: at the frame of
a touch the reported ball is not at the player's feet but inside the bend, by a metre or two
when the ball is struck hard. Outside the reach of the average the path is still the true one,
a straight line coming in and a straight line going out. Extended to the frame of the touch,
each of the two lines says where the ball was, and the corner is put half way between the two
answers. Between a touch and the ends of its bend, and between two touches whose bends overlap,
the path is drawn straight.

Nothing is fitted and no label is used. Whether the reading holds for a recording can be checked
without labels (scripts/corner_check.py): a touch happens at somebody's feet, so the corner
should be nearer a player than the reported ball is, and the two lines should meet.

Only exact arithmetic is used, in single precision throughout, so the result is the same on
every machine. This module imports nothing from the rest of the package.
"""

import numpy as np

SPAN = 10           # frames either side over which a touch's change of velocity is read
THRESHOLD = 3.0     # m/s: a change of velocity at least this large is a touch
REACH = 12          # frames either side of a touch at which the path is straight again: the
                    # half-width of an average of 21 frames followed by one of 5


def touches(ball_velocity, span=SPAN, threshold=THRESHOLD):
    """(frames, strength): where the change of the ball's velocity over `span` frames either side
    peaks at `threshold` or more, and that change at every frame (0 where it cannot be read)."""
    t = len(ball_velocity)
    change = np.full(t, np.nan, dtype=np.float32)
    if t > 2 * span:
        delta = ball_velocity[2 * span:] - ball_velocity[:-2 * span]
        change[span:-span] = np.sqrt(np.sum(delta * delta, axis=1))
    strength = np.nan_to_num(change, nan=0.0)
    peak = strength >= threshold
    for offset in range(1, span + 1):
        peak[offset:] &= strength[offset:] >= strength[:-offset]
        peak[:-offset] &= strength[:-offset] > strength[offset:]
    return np.flatnonzero(peak), strength


def lines(ball, ball_velocity, frames, fps, reach=REACH):
    """(line in, line out): where the path `reach` frames before each touch, carried on at the
    velocity it has there, would be at the frame of the touch, and the same from `reach` frames
    after it, carried back. NaN where the ball or its velocity is missing there."""
    step = np.float32(reach / float(fps))
    line_in = ball[frames - reach] + ball_velocity[frames - reach] * step
    line_out = ball[frames + reach] - ball_velocity[frames + reach] * step
    return line_in, line_out


def rebuild(ball, ball_velocity, fps, span=SPAN, threshold=THRESHOLD, reach=REACH):
    """(rebuilt path, frames of the touches it was rebuilt at).

    `ball` is (T, 2) float32 with NaN where there is no ball, `ball_velocity` its velocity. A
    touch is a peak of the change of velocity over `span` frames either side of at least
    `threshold`; `reach` is how far from a touch the path is taken to be straight again.
    """
    t = len(ball)
    rebuilt = ball.astype(np.float32).copy()
    frames, _ = touches(ball_velocity, span, threshold)
    frames = frames[(frames >= reach) & (frames < t - reach)]
    if not len(frames):
        return rebuilt, frames
    line_in, line_out = lines(ball, ball_velocity, frames, fps, reach)
    has_in = np.isfinite(line_in).all(axis=1)
    has_out = np.isfinite(line_out).all(axis=1)
    corner = np.where((has_in & has_out)[:, None], np.float32(0.5) * (line_in + line_out),
                      np.where(has_in[:, None], line_in, line_out))
    good = has_in | has_out
    frames, corner = frames[good], corner[good].astype(np.float32)
    for k in range(len(frames)):
        frame = int(frames[k])
        rebuilt[frame] = corner[k]
        if k == 0 or frame - int(frames[k - 1]) > 2 * reach:        # the way in
            start = frame - reach
            if np.isfinite(ball[start, 0]):
                weight = (np.arange(start, frame, dtype=np.float32) - start) / np.float32(reach)
                rebuilt[start:frame] = ball[start] + (corner[k] - ball[start]) * weight[:, None]
        if k + 1 < len(frames) and int(frames[k + 1]) - frame <= 2 * reach:   # on to the next touch
            following = int(frames[k + 1])
            weight = (np.arange(frame + 1, following, dtype=np.float32) - frame) / np.float32(following - frame)
            rebuilt[frame + 1:following] = corner[k] + (corner[k + 1] - corner[k]) * weight[:, None]
        else:                                                        # the way out
            stop = frame + reach
            if np.isfinite(ball[stop, 0]):
                weight = (np.arange(frame + 1, stop + 1, dtype=np.float32) - frame) / np.float32(reach)
                rebuilt[frame + 1:stop + 1] = corner[k] + (ball[stop] - corner[k]) * weight[:, None]
    rebuilt[~np.isfinite(ball[:, 0])] = np.nan
    return rebuilt, frames
