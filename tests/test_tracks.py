"""How the simulator treats the two ends of a track of the ball, on generated data only."""

import os
from dataclasses import replace

import numpy as np

from vantage import A, D, H, external, metrica_sample
from vantage.degrade import AT_PLAYER, KICK_REACH, DegradeConfig, _resting, _wander, degrade
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


# -- the fourth generation: the ball is lost and found as the recording's tracker does it ---------

def test_the_short_average_taken_inside_a_track_holds_nothing_of_the_frames_before_it():
    """The third generation softened every kick over five frames of the true path before it hid
    anything, so the first two frames of a track held a trace of the two hidden frames before
    them (an independent review caught it). The fourth takes that average inside the track."""
    half = stoppage_half(False)
    for config, clean in ((replace(THIRD, soften_first=False), True), (THIRD, False)):
        plain = degrade(half, np.random.default_rng(7), config)
        seen = np.isfinite(plain.ball_xy[:, 0])
        # the true ball is somewhere else on every frame the tracker did not have, right up to
        # the frame before it finds the ball again
        moved = half.ball_xy.copy()
        gap = ~seen
        gap[:800] = False
        assert gap.sum() > 100
        moved[gap, 1] += 3.0
        other = degrade(replace(half, ball_xy=moved), np.random.default_rng(7), config)
        assert np.array_equal(np.isfinite(other.ball_xy[:, 0]), seen)
        assert np.array_equal(plain.ball_xy, other.ball_xy, equal_nan=True) == clean


def passing_half():
    """Two minutes of one pass after another between two players twenty metres apart: a second
    in flight, a second at the receiver's feet."""
    t = 3000
    ball = np.zeros((t, 2), dtype=np.float32)
    flight = np.linspace(-10.0, 10.0, 26, dtype=np.float32)[1:]
    for start in range(0, t, 50):
        there = (start // 50) % 2 == 0
        ball[start:start + 25, 0] = flight if there else -flight
        ball[start + 25:start + 50, 0] = 10.0 if there else -10.0
    home = np.zeros((t, 1, 2), dtype=np.float32)
    home[:, 0, 0] = -10.0
    away = np.zeros((t, 1, 2), dtype=np.float32)
    away[:, 0, 0] = 10.0
    return Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=home, away_xy=away,
                labels=np.full(t, H, dtype=np.int8)).check()


def _ends(half, out):
    """(distance of the true ball from the nearest player on the first frame of every track that
    begins after the kick-off, the same on the first frame after every track that ends)."""
    seen = np.isfinite(out.ball_xy[:, 0])
    near = np.minimum(np.abs(half.ball_xy[:, 0] + 10.0), np.abs(half.ball_xy[:, 0] - 10.0))
    begins = np.flatnonzero(~seen[:-1] & seen[1:]) + 1
    lost = np.flatnonzero(seen[:-1] & ~seen[1:]) + 1
    return near[begins], near[lost]


def test_a_ball_lost_in_play_is_picked_up_at_a_kick_and_lost_as_it_reaches_a_player():
    half = passing_half()
    quiet = dict(QUIET, live_loss_rate=8.0, live_loss_median=2.5, live_loss_spread=0.3, until_restart=0.0,
                 w_air=0.0, w_fast=0.0, w_crowd=0.0)
    base = DegradeConfig(ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=21,
                         edges="keep", track_hold=0, soften_first=False, find_offset=0, **quiet)
    begins, lost = _ends(half, degrade(half, np.random.default_rng(3), base))
    assert len(begins) >= 6 and len(lost) >= 6
    # out in the open, as the first three generations draw it: wherever the ball happens to be
    assert np.mean(begins > KICK_REACH) > 0.2 and np.mean(lost > AT_PLAYER) > 0.2
    kicks = replace(base, find_at_kick=1.0, lose_at_player=1.0)
    begins, lost = _ends(half, degrade(half, np.random.default_rng(3), kicks))
    assert len(begins) >= 6 and len(lost) >= 6
    assert np.all(begins <= KICK_REACH)                 # every track begins as the ball is struck
    assert np.mean(lost <= AT_PLAYER) > 0.7             # and most end as it arrives


def test_blips_fall_in_stoppages_as_in_play():
    """Were they kept to live play, a gap of half a second would say that play had not stopped."""
    half = stoppage_half(False)
    watched = replace(THIRD, soften_first=False, dead_hidden=0.0, dead_watched=1.0, dead_watched_in=1.0,
                      ball_blips=40.0, track_hold=0)
    out = degrade(half, np.random.default_rng(5), watched)
    gone = ~np.isfinite(out.ball_xy[:, 0])
    assert gone[:780].any() and gone[1260:].any() and gone[850:1150].any()
    none = degrade(half, np.random.default_rng(5), replace(watched, ball_blips=0.0))
    assert not (~np.isfinite(none.ball_xy[850:1150, 0])).any()


# -- the fifth generation: no short gap, a still ball lost, tags that run early --------------------

def _gaps(out):
    """Lengths in frames of the gaps in the ball's track that touch neither end of the half."""
    seen = np.isfinite(out.ball_xy[:, 0])
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return np.array([b - a for a, b in zip(stops[:-1], starts[1:])])


def test_no_gap_lasts_between_a_blip_and_the_floor():
    """The recording has no gap between 0.6 and 5.16 seconds: what its tracker loses for less it
    bridges. A lost ball is held five frames, so a loss of 5.36 s shows as a gap of 129 frames."""
    half = passing_half()
    quiet = dict(QUIET, live_loss_rate=10.0, live_loss_median=4.0, live_loss_spread=0.6, until_restart=0.0)
    base = DegradeConfig(ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=21, edges="keep",
                         track_hold=5, soften_first=False, **quiet)
    loose = _gaps(degrade(half, np.random.default_rng(2), base))
    assert np.any((loose > 25) & (loose < 129))                      # short losses, as drawn
    for seed in (2, 3, 4):
        # with false tracks drawn into the gaps and blips falling inside them too
        floored = _gaps(degrade(half, np.random.default_rng(seed), replace(
            base, min_gap=5.36, ball_blips=6.0, blip_frames=(6, 12), false_rate=8.0,
            false_floor_strict=seed > 2)))
        assert len(floored) >= 6
        assert np.all((floored < 25) | (floored >= 129))
        assert np.any(floored < 25) and np.any(floored >= 129)


def test_a_blip_inside_a_long_gap_does_not_give_the_gap_back():
    half = held_half()
    quiet = dict(QUIET, live_loss_rate=0.0)
    lost = DegradeConfig(ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=21, edges="keep",
                         track_hold=5, soften_first=False, min_gap=5.36, hold_loss=1.0, ball_blips=40.0,
                         blip_frames=(6, 12), **quiet)
    for seed in range(6):
        seen = np.isfinite(degrade(half, np.random.default_rng(seed), lost).ball_xy[:, 0])
        assert not seen[1015:1190].any()


def test_blips_fall_where_the_ball_is_fast_when_asked():
    half = passing_half()                                            # a second at 10 m/s, a second at rest
    flying = np.zeros(half.n_frames, dtype=bool)
    for start in range(0, half.n_frames, 50):
        flying[start + 2:start + 24] = True
    quiet = dict(QUIET, live_loss_rate=0.0)
    base = DegradeConfig(ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=21, edges="keep",
                         track_hold=0, soften_first=False, ball_blips=20.0, blip_frames=(2, 3), **quiet)
    anywhere = ~np.isfinite(degrade(half, np.random.default_rng(1), base).ball_xy[:, 0])
    fast = ~np.isfinite(degrade(half, np.random.default_rng(1), replace(base, blip_fast=100.0)).ball_xy[:, 0])
    assert 0.3 < np.mean(flying[anywhere]) < 0.6 and np.mean(flying[fast]) > 0.85


def held_half():
    """The passing half, but once the ball lies at a player's feet for eight seconds and once for
    three. Play is live throughout."""
    half = passing_half()
    ball = half.ball_xy.copy()
    ball[1000:1200] = (-10.0, 0.0)
    ball[2000:2075] = (-10.0, 0.0)
    return replace(half, ball_xy=ball)


def test_a_ball_that_lies_still_at_a_players_feet_is_lost_until_it_is_struck():
    half = held_half()
    quiet = dict(QUIET, live_loss_rate=0.0)
    base = DegradeConfig(ball_smooth=21, ball_smooth2=5, ball_jitter=0.0005, player_smooth=21, edges="keep",
                         track_hold=5, soften_first=False, find_at_kick=1.0, min_gap=5.36, **quiet)
    seen = np.isfinite(degrade(half, np.random.default_rng(4), base).ball_xy[:, 0])
    assert seen.all()                                                # nothing is lost unless asked
    seen = np.isfinite(degrade(half, np.random.default_rng(4), replace(base, hold_loss=1.0)).ball_xy[:, 0])
    assert not seen[1015:1190].any()                                 # at his feet from 975 to 1200
    assert seen[900:970].all() and seen[1212:1300].all()             # found as it is played on
    assert seen[1950:2150].all()                                     # three seconds are bridged


def test_the_rule_for_a_resting_ball_does_not_ask_whether_play_is_live():
    """It is given the ball and the players and nothing else, and a ball that waits with whoever
    takes the restart of a stoppage the tracker follows is lost by the same rule as a ball at a
    defender's feet in live play."""
    half = held_half()
    config = DegradeConfig(min_gap=5.36, hold_loss=1.0, hold_delay=0.0)
    lost = _resting(half.ball_xy, half.home_xy, half.away_xy, config, np.random.default_rng(0))
    assert lost[980:1195].all() and not lost[:960].any() and not lost[1215:].any()
    # in a stoppage: the taker stands at the restart spot, the ball is brought to him and waits
    stopped = stoppage_half(False)                                   # stopped from 800 to 1200
    home = stopped.home_xy.copy()
    home[:, 0] = stopped.ball_xy[1200]                               # a player at the restart spot
    stopped = replace(stopped, home_xy=home)
    followed = replace(THIRD, soften_first=False, dead_watched=1.0, dead_watched_in=1.0, min_gap=5.36,
                       find_at_kick=1.0, find_at_kick_restart=1.0, restart_back=12, restart_wait=1.0)
    for seed in range(4):
        seen = np.isfinite(degrade(stopped, np.random.default_rng(seed), followed).ball_xy[:, 0])
        assert seen[700:1300].all()                                  # followed right through
        held = np.isfinite(degrade(stopped, np.random.default_rng(seed), replace(followed, hold_loss=1.0)).ball_xy[:, 0])
        gone = np.flatnonzero(~held)
        if len(gone):                                                # lost while it waits, when it waits long enough
            assert gone.min() >= 800 and gone.max() < 1215 and len(gone) >= 129
            assert held[1215:1300].all()                             # and found as the restart is struck


def test_a_stoppage_shorter_than_the_floor_is_drawn_and_not_shown_as_the_source_has_it():
    """Corrected data freezes the ball where it left play. Bridging a short stoppage with that
    would put a motionless ball off the pitch, which only a stoppage has."""
    t = 2000
    labels = np.full(t, H, dtype=np.int8)
    labels[800:890] = D                                              # three and a half seconds
    ball = np.zeros((t, 2), dtype=np.float32)
    ball[:800, 1] = np.linspace(20.0, 34.0, 800)                     # runs out over the touchline
    ball[800:890] = (0.0, 38.0)                                      # and is frozen four metres outside
    ball[890:, 1] = np.linspace(34.0, 10.0, t - 890)                 # thrown in from the line
    rng = np.random.default_rng(1)
    players = rng.uniform(-30.0, 30.0, size=(1, 6, 2)).astype(np.float32) + np.zeros((t, 6, 2), dtype=np.float32)
    half = Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=players.copy(),
                away_xy=-players.copy(), labels=labels).check()
    config = replace(THIRD, soften_first=False, min_gap=5.36, dead_hidden=1.0)
    for seed in range(4):
        out = degrade(half, np.random.default_rng(seed), config).ball_xy
        assert np.isfinite(out[780:910, 0]).all()                    # bridged: no gap
        assert np.all(np.abs(out[800:890, 1]) < 36.0)                # and not left out there


def test_the_fifth_generations_tags_mostly_run_early():
    labels = np.full(40 * 750, H, dtype=np.int8)
    stops = []
    for k in range(40):
        labels[k * 750 + 300:k * 750 + 550] = D                      # ten seconds, every thirty
        stops.append(k * 750 + 550)
    stops = np.array(stops)
    before = []
    for seed in range(8):
        for frame, _ in external._tagged_early(labels, np.random.default_rng(seed))[1:]:
            nearest = stops[np.argmin(np.abs(stops - frame))]
            if abs(nearest - frame) < 200:                           # not a tag that points at nothing
                before.append((nearest - frame) / 25.0)
    before = np.array(before)
    assert len(before) > 80
    assert 1.5 < np.median(before) < 3.6                             # early, by two to three seconds
    assert 0.08 < np.mean(np.abs(before) < 0.75) < 0.5               # and some about right


# -- the sixth generation: the ball does not stop because play does --------------------------------

def test_the_path_drawn_for_a_followed_stoppage_begins_with_the_balls_own_motion():
    half = stoppage_half(False)                # 8 m/s along x until play stops at frame 800
    followed = replace(THIRD, soften_first=False, dead_watched=1.0, dead_watched_in=1.0, wander_roll=True)
    for seed in range(6):
        out = degrade(half, np.random.default_rng(seed), followed).ball_xy[:, 0].astype(np.float64)
        assert np.isfinite(out[780:830]).all()
        # it goes on the way it was going: 0.8 m and more over the next ten frames, no turning back
        assert out[810] - out[800] > 0.8 and np.all(np.diff(out[795:812]) > 0)


def test_a_dfl_stoppage_gets_a_ball_that_rolls_on_in_the_sixth_generation_only():
    """The DFL matches publish a ball that stands still from the first dead frame on."""
    half = stoppage_half(False)
    half = replace(half, meta=dict(half.meta, source="dfl"))
    # nothing hidden near the start of the stoppage: the ball is held in view twelve seconds
    seen = DegradeConfig(**dict(QUIET, linger=1000.0))
    before = os.environ.get("VANTAGE_SIM")
    try:
        moved = {}
        for generation in ("g5", "g6"):
            os.environ["VANTAGE_SIM"] = generation
            out = external.observed(half, np.random.default_rng(3), seen).ball_xy[:, 0].astype(np.float64)
            moved[generation] = out[812] - out[800]
    finally:
        if before is None:
            del os.environ["VANTAGE_SIM"]
        else:
            os.environ["VANTAGE_SIM"] = before
    # (the fifth's 0.7 m are the tracker's own average running out, not the ball)
    assert moved["g6"] > 2.0 and abs(moved["g5"]) < 1.0


def test_a_stoppage_whose_own_gap_is_under_the_floor_is_drawn_not_bridged():
    half = stoppage_half(False)                # the ball lies on the pitch from 800 to 1200
    # it is held in view twelve seconds, so the gap that is left is four: too short to report
    late = replace(THIRD, soften_first=False, min_gap=5.36, linger=1000.0, hold_loss=0.0)
    for seed in range(4):
        bridged = degrade(half, np.random.default_rng(seed), late).ball_xy.astype(np.float64)
        drawn = degrade(half, np.random.default_rng(seed), replace(late, follow_short_gap=True)).ball_xy.astype(np.float64)
        assert np.isfinite(bridged[820:1180, 0]).all() and np.isfinite(drawn[820:1180, 0]).all()
        # bridged, it is the source's ball: it does not move. Drawn, the players move it about.
        assert np.ptp(bridged[820:1180], axis=0).max() < 1.5
        assert np.ptp(drawn[820:1180], axis=0).max() > 1.5


# -- the seventh generation: the dead ball rolls on as a free ball does -----------------------------

def test_the_dead_ball_does_not_roll_on_by_one_law():
    """A speed that falls off by 1/e a second from the first dead frame on is the same curve at
    every stoppage: twelve frames on it is always 0.62 of what it was. Grass slows a ball at a
    steady rate, and not the same rate every time."""
    half = stoppage_half(False)                # 8 m/s along x until play stops at frame 800
    after = {False: [], True: []}
    for friction in (False, True):
        for seed in range(8):
            x = metrica_sample.with_dead_ball_track(half, np.random.default_rng(seed), friction).ball_xy[:, 0]
            speed = np.diff(x.astype(np.float64)) * 25
            after[friction].append(speed[811] / speed[799])
            assert speed[800] > 6.0                                  # no stop on the first dead frame
            if friction:
                # a steady rate: the loss of speed over twelve frames is the same early and late
                assert abs((speed[800] - speed[806]) - (speed[806] - speed[812])) < 0.05
    assert np.ptp(after[False]) < 0.01 and 0.60 < after[False][0] < 0.65
    assert np.ptp(after[True]) > 0.05


def test_the_path_drawn_for_a_followed_stoppage_has_no_motion_no_ball_has():
    half = stoppage_half(False)
    for seed in range(40):
        path = _wander(half, np.random.default_rng(seed), 800, 1200, half.ball_xy, roll=True, soft=True)
        whole = np.concatenate((half.ball_xy[790:800], path, half.ball_xy[1200:1202])).astype(np.float64)
        steps = np.sqrt(np.sum(np.diff(whole, axis=0) ** 2, axis=1))
        assert steps.max() < 0.85, seed                              # 20 m/s is 0.8 m a frame
