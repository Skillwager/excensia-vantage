"""Putting the corners back on a smoothed ball path, on generated data only."""

import numpy as np

from vantage import corners, smoothing
from vantage.features import BALL_MAX_SPEED, BALL_VEL_HALFSPAN, _central_velocity

FPS = 25


def kicked_ball(frames=6000, every=100, speed=12.0, seed=0):
    """A ball that is struck in a new direction every four seconds and flies straight in between."""
    rng = np.random.default_rng(seed)
    position = np.zeros((frames, 2), dtype=np.float64)
    velocity = np.zeros(2)
    for frame in range(1, frames):
        if frame % every == 1:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        position[frame] = position[frame - 1] + velocity / FPS
    return position.astype(np.float32)


def as_tracked(ball):
    """The path as a tracker that averages over 21 frames and then over 5 reports it."""
    return smoothing.moving_average(smoothing.moving_average(ball, 21), 5)


def velocity_of(ball):
    return _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED)


def test_the_corners_are_put_back_where_the_ball_really_turned():
    true = kicked_ball()
    reported = as_tracked(true)
    rebuilt, frames = corners.rebuild(reported, velocity_of(reported), FPS)
    assert len(frames) > 40
    off = np.sqrt(np.sum((reported[frames] - true[frames]) ** 2, axis=1))
    back = np.sqrt(np.sum((rebuilt[frames] - true[frames]) ** 2, axis=1))
    assert np.median(off) > 0.6                      # the average has cut the corner
    assert np.median(back) < 0.15 and back.max() < 0.5
    # and over the whole of every bend the rebuilt path is the nearer one
    whole = np.sqrt(np.sum((rebuilt - true) ** 2, axis=1))
    assert whole[40:-40].mean() < 0.25 * np.sqrt(np.sum((reported - true) ** 2, axis=1))[40:-40].mean()


def test_the_two_lines_meet_at_a_touch():
    reported = as_tracked(kicked_ball())
    velocity = velocity_of(reported)
    frames, strength = corners.touches(velocity)
    frames = frames[(frames >= corners.REACH) & (frames < len(reported) - corners.REACH)]
    line_in, line_out = corners.lines(reported, velocity, frames, FPS)
    apart = np.sqrt(np.sum((line_in - line_out) ** 2, axis=1))
    apart = apart[np.isfinite(apart)]
    assert len(apart) > 40 and np.median(apart) < 0.05


def test_a_path_without_a_touch_comes_back_as_it_is():
    frames = np.arange(2000, dtype=np.float32)
    straight = np.stack([frames * 0.3, frames * 0.1], axis=1).astype(np.float32)
    rebuilt, found = corners.rebuild(straight, velocity_of(straight), FPS)
    assert len(found) == 0 and np.array_equal(rebuilt, straight)


def test_a_gap_stays_a_gap_and_a_touch_beside_it_uses_the_one_line_it_has():
    true = kicked_ball(frames=1200)
    reported = as_tracked(true).copy()
    reported[410:520] = np.nan                        # the ball is lost just after the touch at 400
    rebuilt, frames = corners.rebuild(reported, velocity_of(reported), FPS)
    assert np.isnan(rebuilt[410:520]).all()
    assert np.isfinite(rebuilt[~np.isnan(reported[:, 0])]).all()


def test_the_result_is_single_precision_and_the_same_twice():
    reported = as_tracked(kicked_ball(frames=3000))
    first, _ = corners.rebuild(reported, velocity_of(reported), FPS)
    second, _ = corners.rebuild(reported.copy(), velocity_of(reported), FPS)
    assert first.dtype == np.float32 and np.array_equal(first, second, equal_nan=True)
