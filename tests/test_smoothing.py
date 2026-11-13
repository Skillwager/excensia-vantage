"""The smoothness measure and the simulator's smoothing, on generated data only."""

import numpy as np

from vantage import smoothing
from vantage.degrade import DegradeConfig, degrade
from vantage.half import Half

FPS = 25


def kicked_ball(frames=6000, every=50, speed=12.0, seed=0):
    """A ball that is struck in a new direction every two seconds and flies straight in between."""
    rng = np.random.default_rng(seed)
    position = np.zeros((frames, 2), dtype=np.float64)
    velocity = np.zeros(2)
    for frame in range(1, frames):
        if frame % every == 1:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        position[frame] = position[frame - 1] + velocity / FPS
    return position.astype(np.float32)


def half_of(ball):
    t = len(ball)
    rng = np.random.default_rng(1)
    players = rng.uniform(-30.0, 30.0, size=(1, 6, 2)).astype(np.float32) + np.zeros((t, 6, 2), dtype=np.float32)
    return Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=players.copy(),
                away_xy=-players.copy(), labels=np.zeros(t, dtype=np.int8)).check()


def test_a_sharp_path_shows_most_of_a_kick_at_once_and_a_smoothed_one_does_not():
    ball = kicked_ball()
    count, sharp = smoothing.spread(ball)
    assert count >= smoothing.ENOUGH and sharp > 0.9
    _, smooth = smoothing.spread(smoothing.moving_average(ball, 21))
    assert 0.34 < smooth < smoothing.REFERENCE          # eight frames of twenty-one, and the velocity's own span


def test_a_sharp_path_is_brought_to_the_reference_and_a_smooth_one_is_left_alone():
    sharp = half_of(kicked_ball())
    out, window = smoothing.to_reference(sharp)
    assert window > 1 and out.meta["smoothed"] == window
    assert smoothing.spread(out.ball_xy)[1] <= smoothing.REFERENCE
    smooth = half_of(smoothing.moving_average(kicked_ball(), 21))
    same, window = smoothing.to_reference(smooth)
    assert window == 1 and same is smooth


def test_a_path_with_too_few_kicks_to_judge_is_left_alone():
    ball = kicked_ball(frames=600)
    assert smoothing.spread(ball)[0] < smoothing.ENOUGH
    half = half_of(ball)
    same, window = smoothing.to_reference(half)
    assert window == 1 and same is half


def test_the_moving_average_keeps_gaps_and_does_not_reach_across_them_for_a_centre():
    ball = kicked_ball(frames=400)
    ball[100:140] = np.nan
    out = smoothing.moving_average(ball, 9)
    assert np.isnan(out[100:140]).all() and np.isfinite(out[:100]).all() and np.isfinite(out[140:]).all()
    # next to the gap the average is over the frames that exist
    assert np.allclose(out[99], np.nanmean(ball[95:104], axis=0), atol=1e-4)


def test_the_simulator_smooths_the_ball_and_the_players_when_asked_and_draws_the_rest_the_same():
    half = half_of(kicked_ball(frames=3000))
    quiet = dict(live_loss_rate=0.0, false_rate=0.0, player_missing=0.0, player_blips=0.0,
                 blackout=0.0, extra_share=0.0, wrong_team_rate=0.0, ball_drift=0.0, parallax=0.0)
    sharp = degrade(half, np.random.default_rng(3), DegradeConfig(ball_smooth=1, **quiet))
    smooth = degrade(half, np.random.default_rng(3), DegradeConfig(
        ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=7, **quiet))
    assert smoothing.spread(sharp.ball_xy)[1] > 0.9
    assert smoothing.spread(smooth.ball_xy)[1] <= smoothing.REFERENCE
    # the same frames are seen and the same players are there: only the paths differ
    assert np.array_equal(np.isfinite(sharp.ball_xy[:, 0]), np.isfinite(smooth.ball_xy[:, 0]))
    assert sharp.home_xy.shape == smooth.home_xy.shape
