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


def slowing_ball(frames=6000, every=50, speed=12.0, keep=0.985, seed=0):
    """Struck every two seconds and slowing down in between, as a real ball does."""
    rng = np.random.default_rng(seed)
    position = np.zeros((frames, 2), dtype=np.float64)
    velocity = np.zeros(2)
    for frame in range(1, frames):
        if frame % every == 1:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        position[frame] = position[frame - 1] + velocity / FPS
        velocity = velocity * keep
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


def test_more_smoothing_never_reads_as_sharper_also_when_the_ball_slows_between_kicks():
    for ball in (kicked_ball(), slowing_ball()):
        readings = [smoothing.spread(smoothing.moving_average(ball, window))[1]
                    for window in (1, 3, 5, 7, 9, 13, 17, 21, 25)]
        assert readings[0] > 0.9
        assert all(later <= earlier + 0.01 for earlier, later in zip(readings, readings[1:]))
        assert readings[-2] < smoothing.REFERENCE < readings[-4]     # 21 frames is smooth, 13 is not


def test_a_sharp_path_is_brought_to_the_reference_and_a_smooth_one_is_left_alone():
    sharp = half_of(kicked_ball())
    out, window = smoothing.to_reference(sharp)
    assert window > 1 and out.meta["smoothed"] == window
    assert smoothing.spread(out.ball_xy)[1] <= smoothing.REFERENCE
    smooth = half_of(smoothing.moving_average(kicked_ball(), 21))
    same, window = smoothing.to_reference(smooth)
    assert window == 1 and same is smooth


def test_the_profile_of_a_kick_is_narrow_on_a_sharp_path_and_wide_on_a_smoothed_one():
    ball = kicked_ball(frames=12000, every=120)
    middle = smoothing.PROFILE_REACH
    total, count = smoothing.profile(ball)
    assert count > 20 and (total / count)[middle - 2:middle + 3].sum() > 0.9
    total, count = smoothing.profile(smoothing.moving_average(ball, 21))
    smooth = total / count
    assert count > 20 and smooth.max() < 0.06               # one frame in twenty-one, about
    assert 0.9 < smooth[middle - 12:middle + 13].sum() < 1.01


def test_the_half_width_of_an_averaged_kick_is_the_length_of_the_average():
    ball = kicked_ball(frames=12000, every=120)
    total, count = smoothing.profile(smoothing.moving_average(ball, 21))
    assert 19.5 < smoothing.half_width(total / count) < 22.5
    total, count = smoothing.profile(ball)
    assert smoothing.half_width(total / count) < 6.0


def test_an_exact_average_leaves_its_mark_in_the_third_difference_at_its_own_length():
    ball = kicked_ball().astype(np.float64)
    marks = smoothing.third_difference([smoothing.moving_average(ball, 21).astype(np.float64)])
    assert marks[21] < -0.2 and min(range(5, 46), key=marks.get) == 21
    assert abs(marks[17]) < 0.1 and abs(marks[25]) < 0.1
    # two averages one after the other leave no such mark at either length
    twice = smoothing.moving_average(smoothing.moving_average(ball, 13).astype(np.float64), 9).astype(np.float64)
    marks = smoothing.third_difference([twice])
    assert marks[21] > -0.1


def test_a_track_keeps_its_speed_where_it_begins_and_slows_to_a_stop_where_it_ends():
    path = np.full((400, 2), np.nan, dtype=np.float32)
    path[50:350, 0] = np.arange(300, dtype=np.float32) * 0.4       # 10 m/s along x
    path[50:350, 1] = 3.0
    out = smoothing.track_average(path, 21, hold=5)
    assert np.isnan(out[:50]).all() and np.isnan(out[355:]).all()  # five frames longer, no more
    assert np.isfinite(out[50:355]).all()
    step = np.diff(out[50:355, 0].astype(np.float64))
    assert np.allclose(step[:250], 0.4, atol=1e-3)                 # at full speed from the first frame
    assert abs(step[-1] / 0.4 - 6 / 21) < 0.01                     # the last step: 6 of 21 frames still move
    assert abs(step[-6] / 0.4 - 11 / 21) < 0.01
    assert np.all(np.diff(step[-17:]) <= 1e-4)                     # and slowing all the way there
    assert np.allclose(out[50:355, 1], 3.0, atol=1e-4)


def test_a_lost_track_is_held_only_until_the_next_one_begins():
    path = np.full((200, 2), np.nan, dtype=np.float32)
    path[10:80] = 1.0
    path[82:150] = 5.0                                             # found again two frames later
    out = smoothing.track_average(path, 21, hold=5)
    assert np.isfinite(out[10:150]).all() and np.isnan(out[150 + 5:]).all()
    assert np.allclose(out[80:82], 1.0) and np.allclose(out[82:150], 5.0)
    same = smoothing.track_average(path, 1)
    assert np.array_equal(same, path, equal_nan=True)              # no window and no hold: nothing changes


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
