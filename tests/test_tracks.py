"""How the simulator treats the two ends of a track of the ball, on generated data only."""

from dataclasses import replace

import numpy as np

from vantage import A, D, H
from vantage.degrade import DegradeConfig, degrade
from vantage.half import Half

# Nothing random but the moments the ball is lost and found again around a stoppage.
QUIET = dict(live_loss_rate=0.0, false_rate=0.0, player_missing=0.0, player_blips=0.0, blackout=0.0,
             extra_share=0.0, wrong_team_rate=0.0, ball_drift=0.0, player_drift=0.0, parallax=0.0,
             dead_hidden=1.0, dead_watched=0.0, dead_watched_in=0.0, early_find=0.0, long_miss=0.0,
             linger=0.5)
THIRD = DegradeConfig(ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=21,
                      edges="keep", track_hold=5, **QUIET)


def stoppage_half(roam):
    """Live play, a stoppage of sixteen seconds, live play. With `roam` the ball is moved about in
    the middle of the stoppage, where the tracker has lost it; every other frame is the same."""
    t = 2000
    labels = np.full(t, H, dtype=np.int8)
    labels[800:1200] = D
    labels[1200:] = A
    ball = np.zeros((t, 2), dtype=np.float32)
    ball[:700, 0] = -20.0
    ball[700:800, 0] = -20.0 + np.arange(100) * (8.0 / 25)          # 8 m/s, then at rest
    ball[800:, 0] = ball[799, 0]
    ball[1200:1260, 1] = np.arange(60) * (10.0 / 25)                # struck at the restart
    ball[1260:, 1] = ball[1259, 1]
    if roam:
        away = np.minimum(np.arange(950, 1198) - 950, 1198 - np.arange(950, 1198)) * (6.0 / 124)
        ball[950:1198, 1] += away.astype(np.float32)
    rng = np.random.default_rng(1)
    players = rng.uniform(-30.0, 30.0, size=(1, 6, 2)).astype(np.float32) + np.zeros((t, 6, 2), dtype=np.float32)
    return Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=players.copy(),
                away_xy=-players.copy(), labels=labels).check()


def test_what_the_ball_did_while_it_was_hidden_leaves_no_trace_on_the_frames_that_are_seen():
    plain = degrade(stoppage_half(False), np.random.default_rng(7), THIRD)
    roamed = degrade(stoppage_half(True), np.random.default_rng(7), THIRD)
    assert np.isnan(plain.ball_xy[940:1200, 0]).all()               # the stoppage did hide the ball
    assert np.array_equal(plain.ball_xy, roamed.ball_xy, equal_nan=True)
    # averaging the whole true path and hiding frames afterwards does leave a trace: that was
    # the second generation, and it is why it was dropped
    through = replace(THIRD, edges="through", track_hold=0)
    plain = degrade(stoppage_half(False), np.random.default_rng(7), through)
    roamed = degrade(stoppage_half(True), np.random.default_rng(7), through)
    assert not np.array_equal(plain.ball_xy, roamed.ball_xy, equal_nan=True)


def test_a_lost_ball_is_reported_five_frames_longer_and_nothing_else_about_what_is_seen_changes():
    half = stoppage_half(False)
    kept = degrade(half, np.random.default_rng(7), THIRD)
    plain = degrade(half, np.random.default_rng(7), replace(THIRD, track_hold=0))
    seen, without = np.isfinite(kept.ball_xy[:, 0]), np.isfinite(plain.ball_xy[:, 0])
    assert (seen & ~without).sum() == 5 and not (without & ~seen).any()
    lost = int(np.flatnonzero(without[:-1] & ~without[1:])[0]) + 1   # the first frame without a ball
    assert seen[lost:lost + 5].all() and not seen[lost + 5]


def test_a_found_ball_is_at_speed_from_its_first_frame():
    out = degrade(stoppage_half(False), np.random.default_rng(7), THIRD)
    seen = np.isfinite(out.ball_xy[:, 0])
    found = int(np.flatnonzero(~seen[:-1] & seen[1:])[0]) + 1        # the restart: 10 m/s in y
    assert 1200 <= found < 1245
    step = np.diff(out.ball_xy[found:found + 8, 1].astype(np.float64)) * 25
    assert np.all(step > 9.0) and np.all(step < 10.5)
