"""Make clean, corrected tracking look like automatic tracking.

The external training matches are official, corrected data: the ball is known on every frame and
every player is followed all game. The competition match is the raw output of a computer-vision
system. A model trained on the clean version would lean on information it will not have, so every
training half is passed through this simulator first.

What the simulator imitates
---------------------------
Its shape follows what automatic ball tracking actually looks like, measured without any labels on
the provided recording (scripts/domain_report.py prints the same totals for the recording and for
the simulator, side by side):

* The ball is followed smoothly for long unbroken runs and then is gone for a long time, once or
  twice a minute. It does not flicker. Most disappearances go with a stoppage: the track ends
  around the moment play stops and resumes shortly after the restart, once the ball is moving
  again (a run of the ball almost never begins with the ball at rest).
* At a stoppage where the ball never leaves the pitch (a foul), the ball rolls to a stop in view
  and the track holds it for a few seconds before losing it. A ball at rest is almost never
  tracked for longer than ten seconds.
* It is also lost during live play, a few times a half, and can stay lost for many seconds.
* A visible ball is almost never off the pitch.
* Player tracks are smooth and long. Players go missing for long spells rather than single
  frames, a team is sometimes credited with one body too many (a duplicate, an official), and now
  and then most of the players are missing at once.

What is *not* known is how much of each: how often a stoppage hides the ball, how often it is lost
in live play, for how long. Those cannot be measured without labels, and labels on the provided
match may not be used. So every call draws them from wide ranges (`sample_config`): the model sees
recordings where a missing ball nearly always means a stoppage and recordings where it often does
not, and has to rely on evidence that holds across all of them.
"""

from dataclasses import dataclass, replace

import numpy as np

from . import D, FPS


@dataclass
class DegradeConfig:
    # -- ball: stoppages ------------------------------------------------------------------
    dead_hidden: float = 0.85         # chance a short stoppage (under 5 s) hides the ball
    dead_watched: float = 0.03        # chance a longer stoppage is tracked throughout when the
                                      # ball leaves the pitch (throw-in, corner, goal kick)
    dead_watched_in: float = 0.2      # chance a stoppage under 10 s with the ball still on the
                                      # pitch (a foul) is tracked throughout
    linger: float = 3.0               # s, mean time the track holds a ball that has stopped on
                                      # the pitch before losing it
    lose_shift: float = 0.0           # s, mean of when the track ends relative to the stoppage
    lose_spread: float = 0.8          # s
    early_find: float = 0.05          # chance the ball is picked up again before the restart
    early_find_mean: float = 2.0      # s before the restart, when it is
    find_lag: float = 0.3             # s after the restart before the moving ball is picked up
    long_miss: float = 0.05           # chance the restart is missed and the ball found much later
    # -- ball: lost in live play -----------------------------------------------------------
    live_loss_rate: float = 0.8       # losses per minute of live play
    live_loss_median: float = 6.0     # s
    until_restart: float = 0.35       # chance a live loss lasts until the next restart
    w_air: float = 1.0                # how much height, speed and crowding raise the loss rate
    w_fast: float = 0.6
    w_crowd: float = 0.6
    # -- ball: measurement -----------------------------------------------------------------
    out_margin: float = 1.5           # m: a ball further than this off the pitch is not reported
    ball_drift: float = 0.15          # m, slow error
    ball_smooth: int = 5              # frames: the tracker's own smoothing of the ball's path
    parallax: float = 0.1             # 0 = true ground position, 1 = full single-camera parallax
    camera_x: float = 0.0
    camera_back: float = 30.0         # m behind the near touchline
    camera_height: float = 18.0
    false_rate: float = 0.05          # false ball tracks per minute while the ball is hidden
    # -- players ---------------------------------------------------------------------------
    player_missing: float = 0.03      # fraction of player-frames with no detection
    w_player_crowd: float = 1.0       # how much standing close to others raises that
    player_gap_mean: float = 8.0      # s, mean length of a player's absence
    player_blips: float = 0.3         # brief losses (under 2 s) per player-minute
    player_drift: float = 0.2         # m, slow error
    blackout: float = 0.35            # chance a long stoppage loses many players for a while
    extra_share: float = 0.1          # share of the half with an extra body on some team
    wrong_team_rate: float = 0.0005   # per player-second


def sample_config(rng):
    """One random noise setting. Wide on purpose: see the module docstring."""
    return DegradeConfig(
        dead_hidden=float(rng.uniform(0.5, 1.0)),
        dead_watched=float(rng.uniform(0.0, 0.1)),
        dead_watched_in=float(rng.uniform(0.0, 0.4)),
        linger=float(rng.uniform(1.0, 6.0)),
        lose_shift=float(rng.uniform(-2.5, 0.5)),
        lose_spread=float(rng.uniform(0.3, 2.0)),
        early_find=float(rng.uniform(0.0, 0.15)),
        early_find_mean=float(rng.uniform(0.5, 4.0)),
        find_lag=float(rng.uniform(0.05, 1.0)),
        long_miss=float(rng.uniform(0.0, 0.15)),
        live_loss_rate=float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))),
        live_loss_median=float(rng.uniform(2.5, 14.0)),
        until_restart=float(rng.uniform(0.1, 0.6)),
        w_air=float(rng.uniform(0.0, 2.0)),
        w_fast=float(rng.uniform(0.0, 1.2)),
        w_crowd=float(rng.uniform(0.0, 1.2)),
        out_margin=float(rng.uniform(0.5, 4.0)),
        ball_drift=float(rng.uniform(0.0, 0.35)),
        ball_smooth=int(rng.choice((3, 5, 7, 9))),
        parallax=float(rng.uniform(0.0, 0.3)),
        camera_x=float(rng.uniform(-12.0, 12.0)),
        camera_back=float(rng.uniform(15.0, 45.0)),
        camera_height=float(rng.uniform(10.0, 28.0)),
        false_rate=float(rng.uniform(0.0, 0.2)),
        player_missing=float(rng.uniform(0.01, 0.06)),
        w_player_crowd=float(rng.uniform(0.0, 2.0)),
        player_gap_mean=float(rng.uniform(3.0, 20.0)),
        player_blips=float(rng.uniform(0.0, 1.0)),
        player_drift=float(rng.uniform(0.0, 0.4)),
        blackout=float(rng.uniform(0.0, 0.7)),
        extra_share=float(rng.uniform(0.0, 0.25)),
        wrong_team_rate=float(rng.uniform(0.0, 0.001)),
    )


# -- helpers -----------------------------------------------------------------------------------

def _runs(flag):
    """[(start, stop)] of the true runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _smooth_noise(rng, shape, sigma, timescale_frames):
    """Slowly varying noise with standard deviation `sigma` along axis 0."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    steps = max(2, int(np.ceil(shape[0] / float(timescale_frames))) + 2)
    knots = rng.normal(0.0, sigma, size=(steps,) + tuple(shape[1:]))
    positions = np.linspace(0, steps - 1, shape[0])
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, steps - 1)
    weight = (positions - lower).reshape((-1,) + (1,) * (len(shape) - 1))
    return ((1 - weight) * knots[lower] + weight * knots[upper]).astype(np.float32)


def _moving_average(xy, window):
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


def _neighbours_within(point_xy, home_xy, away_xy, radius):
    count = np.zeros(len(point_xy), dtype=np.int16)
    for team in (home_xy, away_xy):
        delta = team - point_xy[:, None, :]
        with np.errstate(invalid="ignore"):
            count += np.sum(np.sum(delta * delta, axis=-1) < radius * radius, axis=1).astype(
                np.int16)
    return count


def _frames(seconds):
    return int(round(seconds * FPS))


# -- ball visibility ---------------------------------------------------------------------------

def _hidden_ball(half, rng, config, ball, height):
    """Boolean array: frames on which the automatic tracker reports no ball."""
    t = half.n_frames
    dead = (half.labels == D) if half.labels is not None else np.zeros(t, dtype=bool)
    hidden = np.zeros(t, dtype=bool)
    stoppages = _runs(dead)

    length, width = half.pitch
    margin = np.minimum(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
    for start, stop in stoppages:
        short = stop - start < 5 * FPS
        # Does the ball leave the pitch as play stops? A ball that stays on it (a foul, an
        # offside) is in plain view and is often tracked straight through the stoppage.
        window = margin[max(0, start - 5):min(t, start + 3 * FPS)]
        leaves = bool(np.any(window[np.isfinite(window)] < 0.3))
        if short:
            watched = 1.0 - config.dead_hidden
        elif leaves:
            watched = config.dead_watched
        else:
            watched = config.dead_watched_in if stop - start < 10 * FPS else 0.0
        if rng.random() < watched:
            continue                                   # this stoppage is tracked throughout
        if leaves:
            lose = start + _frames(rng.normal(config.lose_shift, config.lose_spread))
        else:
            # The ball rolls to a stop in view and is held for a few seconds before it is lost.
            lose = start + _frames(min(rng.exponential(config.linger), 12.0))
        if rng.random() < config.early_find:
            find = stop - _frames(rng.exponential(config.early_find_mean))
        else:
            # The usual case: the ball is picked up once it is moving again after the restart.
            find = stop + _frames(rng.exponential(config.find_lag))
            if rng.random() < config.long_miss:
                find += _frames(rng.exponential(8.0))
        lose, find = max(0, lose), min(t, find)
        if find > lose:
            hidden[lose:find] = True

    # Lost in live play: more likely when the ball is high, fast or among players.
    velocity = np.zeros_like(ball)
    velocity[1:-1] = (ball[2:] - ball[:-2]) * (FPS / 2.0)
    speed = np.nan_to_num(np.sqrt(np.sum(velocity * velocity, axis=1)))
    crowd = _neighbours_within(ball, half.home_xy, half.away_xy, 2.5)
    risk = np.exp(config.w_air * np.clip(height / 2.0, 0.0, 2.0)
                  + config.w_fast * np.clip((speed - 10.0) / 10.0, 0.0, 1.5)
                  + config.w_crowd * np.clip((crowd - 1) / 2.0, 0.0, 1.5))
    live = ~dead
    risk = np.where(live, risk, 0.0)
    if risk.sum() > 0:
        expected = config.live_loss_rate * live.sum() / (FPS * 60.0)
        restarts = np.array([stop for _, stop in stoppages] + [t], dtype=np.int64)
        for _ in range(int(rng.poisson(expected))):
            start = int(rng.choice(t, p=risk / risk.sum()))
            if rng.random() < config.until_restart:
                stop = int(restarts[np.searchsorted(restarts, start, side="right")])
            else:
                length = np.exp(rng.normal(np.log(config.live_loss_median), 0.9))
                stop = start + _frames(float(np.clip(length, 0.3, 90.0)))
            hidden[start:min(t, stop)] = True
    return hidden


# -- the simulator -----------------------------------------------------------------------------

def degrade(half, rng, config):
    """A copy of `half` as automatic tracking might have recorded it. Labels are untouched."""
    if config is None:
        return half
    length, width = half.pitch
    t = half.n_frames
    ball = half.ball_xy.astype(np.float32).copy()
    height = half.ball_z if half.ball_z is not None else np.zeros(t, dtype=np.float32)
    height = np.nan_to_num(height, nan=0.0)
    unavailable = ~np.isfinite(ball[:, 0])              # the source itself has no ball here

    # -- where the ball is reported: single-camera parallax and a slow error ---------------
    camera = np.array([config.camera_x, -(width / 2 + config.camera_back)], dtype=np.float32)
    safe_height = np.clip(height, 0.0, config.camera_height * 0.8)
    stretch = config.parallax * safe_height / (config.camera_height - safe_height)
    reported = ball + stretch[:, None] * (ball - camera)
    reported += _smooth_noise(rng, reported.shape, config.ball_drift, 2 * FPS)
    reported = _moving_average(reported, config.ball_smooth)

    hidden = _hidden_ball(half, rng, config, ball, height) | unavailable

    # A tracker does not report a ball well off the pitch.
    margin = np.minimum(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
    with np.errstate(invalid="ignore"):
        hidden |= margin < -config.out_margin
    reported[hidden] = np.nan

    # -- a track that is not the ball -------------------------------------------------------
    minutes_hidden = hidden.sum() / (FPS * 60.0)
    for _ in range(int(rng.poisson(config.false_rate * minutes_hidden))):
        candidates = np.flatnonzero(hidden)
        start = int(rng.choice(candidates))
        stop = min(t, start + _frames(rng.uniform(0.5, 6.0)))
        spot = np.array([rng.uniform(-length / 2, length / 2), rng.uniform(-width / 2, width / 2)])
        drift = rng.normal(0.0, 1.5, size=2)
        seconds = (np.arange(stop - start) / float(FPS))[:, None]
        rows = slice(start, stop)
        fake = (spot[None, :] + drift[None, :] * seconds).astype(np.float32)
        keep = hidden[rows]
        reported[rows][keep] = fake[keep]

    # -- players ---------------------------------------------------------------------------
    dead = (half.labels == D) if half.labels is not None else np.zeros(t, dtype=bool)
    # During a long stoppage the camera may leave the pitch: many players vanish together.
    blackout_spans = []
    for start, stop in _runs(dead):
        if stop - start > 15 * FPS and rng.random() < config.blackout:
            a = start + int(rng.uniform(0.05, 0.5) * (stop - start))
            b = a + int(rng.uniform(0.2, 0.6) * (stop - start))
            blackout_spans.append((a, b, float(rng.uniform(0.3, 1.0))))

    teams = []
    for own, other in ((half.home_xy, half.away_xy), (half.away_xy, half.home_xy)):
        xy = own.astype(np.float32).copy()
        xy += _smooth_noise(rng, xy.shape, config.player_drift, 4 * FPS)
        for slot in range(xy.shape[1]):
            on_pitch = np.isfinite(xy[:, slot, 0])
            if not on_pitch.any():
                continue
            # A player is lost where others stand close (occlusion), rarely when alone.
            close = _neighbours_within(own[:, slot], own, other, 2.0) - 1
            risk = np.where(on_pitch, np.exp(config.w_player_crowd * np.clip(close, 0, 3)), 0.0)
            rate = config.player_missing / max(config.player_gap_mean, 1e-6)   # losses per second
            for _ in range(int(rng.poisson(rate * on_pitch.sum() / FPS))):
                start = int(rng.choice(t, p=risk / risk.sum()))
                stop = start + _frames(rng.exponential(config.player_gap_mean))
                xy[start:stop, slot] = np.nan
            minutes = on_pitch.sum() / (FPS * 60.0)
            for _ in range(int(rng.poisson(config.player_blips * minutes))):
                start = int(rng.integers(0, t))
                xy[start:start + _frames(rng.uniform(0.2, 2.0)), slot] = np.nan
            for a, b, share in blackout_spans:
                if rng.random() < share:
                    xy[a:b, slot] = np.nan
        teams.append(xy)
    home_xy, away_xy = teams

    # an extra body on a team: a duplicate of a player, or someone who is not a player at all
    centre = np.nanmean(np.concatenate([half.home_xy, half.away_xy], axis=1), axis=1)
    extra = {0: [], 1: []}
    remaining = config.extra_share * t
    while remaining > 0:
        span = min(t, _frames(rng.uniform(2.0, 60.0)))
        start = int(rng.integers(0, max(1, t - span)))
        side = int(rng.integers(0, 2))
        track = np.full((t, 2), np.nan, dtype=np.float32)
        rows = slice(start, start + span)
        if rng.random() < 0.5:
            source = (home_xy, away_xy)[side]
            slot = int(rng.integers(0, source.shape[1]))
            track[rows] = source[rows, slot] + rng.normal(0.0, 0.5, size=2)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            offset = rng.uniform(6.0, 16.0) * np.array([np.cos(angle), np.sin(angle)])
            track[rows] = centre[rows] + offset + _smooth_noise(rng, (span, 2), 2.0, 3 * FPS)
        extra[side].append(track)
        remaining -= span

    seconds = t / float(FPS)
    for side, source in ((0, home_xy), (1, away_xy)):
        for _ in range(int(rng.poisson(config.wrong_team_rate * source.shape[1] * seconds))):
            slot = int(rng.integers(0, source.shape[1]))
            start = int(rng.integers(0, t))
            stop = min(t, start + _frames(rng.uniform(1.0, 10.0)))
            moved = np.full((t, 2), np.nan, dtype=np.float32)
            moved[start:stop] = source[start:stop, slot]
            source[start:stop, slot] = np.nan
            extra[1 - side].append(moved)
    if extra[0]:
        home_xy = np.concatenate([home_xy, np.stack(extra[0], axis=1)], axis=1)
    if extra[1]:
        away_xy = np.concatenate([away_xy, np.stack(extra[1], axis=1)], axis=1)

    return replace(half, ball_xy=reported.astype(np.float32), home_xy=home_xy, away_xy=away_xy,
                   ball_z=None, meta=dict(half.meta, degraded=True))
