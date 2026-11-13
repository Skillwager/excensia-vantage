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
  twice a minute. It does not flicker. Many disappearances go with a stoppage: the track ends
  around the moment play stops and resumes shortly after the restart, once the ball is moving
  again (a run of the ball almost never begins with the ball at rest).
* A stoppage does not have to hide the ball. The tracker can follow it all the way through one,
  while it is fetched, carried to the line and put down. A first version of this simulator
  assumed that was rare. The pipeline check on the public five minutes showed otherwise (two of
  its four stoppages have a ball on every frame), and label-free totals agree: on a quarter of
  the recording the ball is tracked while the players are close to standing still. How often it
  happens is now one more unknown drawn from a wide range.
* A ball that is followed through a stoppage is rarely still. While the players are close to
  standing, the tracked ball is at rest on only about a fifth of the frames; the rest of the time
  it is rolled, carried or passed about at a few metres a second (scripts/slow_report.py, no
  labels). Corrected data does not show this: one source freezes the ball when play stops and the
  other has no ball at all. So the simulator moves the ball itself during a followed stoppage:
  it rests, goes along with the nearest player, or is passed to another, and ends on the spot
  play restarts from.
* At a stoppage where the ball never leaves the pitch (a foul), the track may also hold the ball
  for a few seconds and then lose it.
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
from .smoothing import track_average


@dataclass
class DegradeConfig:
    # -- ball: stoppages ------------------------------------------------------------------
    dead_hidden: float = 0.7          # chance a short stoppage (under 5 s) hides the ball
    dead_watched: float = 0.25        # chance a longer stoppage is tracked throughout
    dead_watched_in: float = 0.4      # the same for a stoppage under 10 s with the ball still on
                                      # the pitch (a foul)
    hold_at_line: float = 0.5         # chance that a ball followed through a stoppage is kept at
                                      # the edge of the pitch while it is off it, not lost there
    linger: float = 3.0               # s, mean time the track holds a ball that has stopped on
                                      # the pitch before losing it
    lose_shift: float = 0.0           # s, mean of when the track ends relative to the stoppage
    lose_spread: float = 0.8          # s
    early_find: float = 0.25          # chance the ball is picked up again before the restart
    early_find_mean: float = 4.0      # s before the restart, when it is
    find_lag: float = 0.3             # s after the restart before the moving ball is picked up
    long_miss: float = 0.05           # chance the restart is missed and the ball found much later
    # -- ball: lost in live play -----------------------------------------------------------
    live_loss_rate: float = 0.5       # losses per minute of live play
    live_loss_median: float = 6.0     # s
    until_restart: float = 0.35       # chance a live loss lasts until the next restart
    find_at_kick: float = 0.0         # chance a hidden ball is picked up again at a kick: the frame
                                      # after a player strikes it, with the ball still at his feet
    lose_at_player: float = 0.0       # chance a ball lost in live play is lost as it reaches a
                                      # player, not out in the open
    find_offset: int = 1              # frames after the kick at which such a track begins
    find_at_kick_restart: float = -1.0  # the same chance for the pick-up after a restart, where it
                                      # differs (below 0: the same)
    lose_at_player_dead: float = 0.0  # chance the ball is lost as it reaches a player when play
                                      # has stopped
    soften_first: bool = True         # the short average that stands for a real kick runs over
                                      # the true path before anything is hidden (generation 3)
                                      # or, when False, inside each track after it
    find_lag_spread: float = 0.0      # 0: the time from a restart to the pick-up is drawn as it
                                      # always was (exponential, mean find_lag). Above 0: a
                                      # restart is picked up at once with chance find_quick
                                      # (exponential, mean find_lag_quick); otherwise the
                                      # logarithm of the time is drawn around find_lag with this
                                      # spread and the pick-up is at the first kick after it: a
                                      # ball restarted through the air is not tracked until it
                                      # is played on the ground
    find_quick: float = 0.3
    find_lag_quick: float = 0.3       # s
    live_loss_spread: float = 0.9     # spread of the length of a live loss (of its logarithm)
    ball_blips: float = 0.0           # losses of under a second, per minute of live play
    min_gap: float = 0.0              # s: a loss shorter than this is not reported as one. The
                                      # recording has no gap between 0.6 and 5 seconds: its
                                      # tracker bridges a ball it loses for less (here the true
                                      # path stands for the bridge)
    blip_frames: tuple = ()           # (shortest, longest) loss in frames that is reported all the
                                      # same; empty: blips last 0.2 to 0.9 s, as first drawn
    blip_fast: float = 0.0            # how much more often a blip falls where the ball is fast
                                      # (0: anywhere)
    hold_loss: float = 0.0            # chance that a ball which lies still within a player's
                                      # reach (a keeper holding it, a defender standing on it,
                                      # whoever is about to take a free kick) is lost until it
                                      # is struck, whether play is live or not. Under min_gap
                                      # such a loss is bridged like any other
    hold_delay: float = 0.0           # s: mean time the ball has lain there when it is lost
    restart_back: int = 3             # frames before the labelled restart from which the strike
                                      # that restarts play is looked for
    restart_wait: float = 4.0         # s from there for which it is looked for (KICK_WAIT)
    until_restart_as_restart: bool = False  # a ball lost in play until the next restart is picked
                                      # up as any restart is, not as a ball lost in play is
    wander_roll: bool = False         # the path drawn for a stoppage the tracker follows begins
                                      # with the ball's own motion: it rolls on and dies away,
                                      # as a ball does when the whistle goes
    follow_short_gap: bool = False    # a stoppage whose own gap would come out shorter than
                                      # min_gap is followed (its ball is drawn) and not bridged
                                      # with what the source has there
    wander_soft: bool = False         # the path drawn for a followed stoppage has no motion that
                                      # no ball has: the roll-on is slowed as grass slows a ball,
                                      # no pass is faster than 20 m/s, and the wait at the
                                      # restart spot begins without a hop
    height_fall: int = 0              # frames over which a ball that is in the air when play
                                      # stops comes down (0: it is on the ground at once, which
                                      # the camera's parallax shows as a step)
    false_floor_strict: bool = False  # two false tracks in one gap may not leave a piece under
                                      # the floor between them either
    wander_smooth: int = 1            # frames: the path drawn for a ball that is followed through
                                      # a stoppage is averaged over this many, so that it has no
                                      # kick that nobody made
    w_air: float = 1.0                # how much height, speed and crowding raise the loss rate
    w_fast: float = 0.6
    w_crowd: float = 0.6
    # -- ball: measurement -----------------------------------------------------------------
    out_margin: float = 1.5           # m: a ball further than this off the pitch is not reported
    ball_drift: float = 0.5           # m, slow error
    ball_smooth: int = 5              # frames: the tracker's own smoothing of the ball's path
    ball_smooth2: int = 1             # frames: a second, shorter moving average after the first
    ball_jitter: float = 0.0          # m: frame-to-frame noise left in the smoothed path
    track_hold: int = 0               # frames a lost ball or player is still reported where it
                                      # was last seen, before the smoothing (only with edges
                                      # "keep"). It is what makes every track end slowly
    edges: str = "through"            # what the smoothing does where a track begins and ends:
                                      # "through" averages the whole true path and hides frames
                                      # afterwards, so a track's first and last frames hold
                                      # something of the frames the tracker never had; "shrink"
                                      # hides first and lets the average run out of frames, which
                                      # halves the speed there; "keep" hides first and carries
                                      # each track on past its ends at the speed it has there
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
    player_smooth: int = 1            # frames: the tracker's own smoothing of a player's path
    player_smooth2: int = 1           # frames: a second, shorter moving average after the first
    blackout: float = 0.35            # chance a long stoppage loses many players for a while
    extra_share: float = 0.1          # share of the half with an extra body on some team
    wrong_team_rate: float = 0.0005   # per player-second


def sample_config(rng):
    """One random noise setting. Wide on purpose: see the module docstring."""
    return DegradeConfig(
        dead_hidden=float(rng.uniform(0.2, 1.0)),
        dead_watched=float(rng.uniform(0.0, 0.7)),
        dead_watched_in=float(rng.uniform(0.0, 0.9)),
        hold_at_line=float(rng.uniform(0.0, 1.0)),
        linger=float(rng.uniform(1.0, 6.0)),
        lose_shift=float(rng.uniform(-2.5, 0.5)),
        lose_spread=float(rng.uniform(0.3, 2.0)),
        early_find=float(rng.uniform(0.0, 0.5)),
        early_find_mean=float(rng.uniform(0.5, 8.0)),
        find_lag=float(rng.uniform(0.05, 1.0)),
        long_miss=float(rng.uniform(0.0, 0.15)),
        live_loss_rate=float(np.exp(rng.uniform(np.log(0.1), np.log(2.0)))),
        live_loss_median=float(rng.uniform(2.5, 14.0)),
        until_restart=float(rng.uniform(0.1, 0.6)),
        w_air=float(rng.uniform(0.0, 2.0)),
        w_fast=float(rng.uniform(0.0, 1.2)),
        w_crowd=float(rng.uniform(0.0, 1.2)),
        out_margin=float(rng.uniform(0.5, 4.0)),
        ball_drift=float(rng.uniform(0.1, 1.0)),
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


def sample_config_found_at_kicks(rng):
    """One random noise setting for a tracker that picks the ball up at a kick and loses it at a
    player (the fourth generation, see vantage/external.py). The ranges that say how often and for
    how long the ball is lost in live play sit around what the recording shows without labels;
    the rest are the ranges of `sample_config`."""
    base = sample_config(rng)
    return replace(
        base,
        dead_watched=float(rng.uniform(0.1, 0.7)),
        early_find=float(rng.uniform(0.0, 0.25)),
        find_quick=float(rng.uniform(0.1, 0.45)),
        find_lag_quick=float(rng.uniform(0.2, 1.0)),
        find_lag=float(rng.uniform(2.0, 3.8)),
        find_lag_spread=float(rng.uniform(0.3, 0.6)),
        live_loss_rate=float(np.exp(rng.uniform(np.log(1.0), np.log(4.0)))),
        live_loss_median=float(rng.uniform(5.0, 11.0)),
        live_loss_spread=float(rng.uniform(0.3, 0.7)),
        until_restart=float(rng.uniform(0.05, 0.35)),
        find_at_kick=float(rng.uniform(0.6, 1.0)),
        find_at_kick_restart=float(rng.uniform(0.0, 0.6)),
        lose_at_player=float(rng.uniform(0.3, 1.0)),
        lose_at_player_dead=float(rng.uniform(0.0, 0.6)),
        soften_first=False,
        find_offset=int(rng.choice((-1, 0, 1))),
        ball_blips=float(rng.uniform(0.0, 0.4)),
    )


def sample_config_fifth(rng):
    """One random noise setting of the fifth generation (see vantage/external.py): the ranges sit
    around what the recording shows without labels, and what cannot be read off it is drawn
    widely. A pick-up at a kick and a loss at a player have one chance each, in play and out of
    it alike, so that how a gap begins or ends does not say whether play had stopped."""
    base = sample_config(rng)
    at_kick = float(rng.uniform(0.6, 1.0))
    at_player = float(rng.uniform(0.5, 1.0))
    return replace(
        base,
        dead_watched=float(rng.uniform(0.05, 0.55)),
        dead_watched_in=float(rng.uniform(0.15, 0.7)),
        early_find=float(rng.uniform(0.0, 0.25)),
        find_quick=float(rng.uniform(0.6, 1.0)),
        find_lag_quick=float(rng.uniform(0.05, 0.3)),
        find_lag=float(rng.uniform(0.6, 1.6)),
        find_lag_spread=float(rng.uniform(0.3, 0.6)),
        restart_back=12,
        restart_wait=1.0,
        until_restart_as_restart=True,
        live_loss_rate=float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))),
        live_loss_median=float(rng.uniform(5.5, 10.0)),
        live_loss_spread=float(rng.uniform(0.5, 0.9)),
        until_restart=float(rng.uniform(0.03, 0.25)),
        find_at_kick=at_kick,
        find_at_kick_restart=at_kick,
        lose_at_player=at_player,
        lose_at_player_dead=at_player,
        hold_loss=float(rng.uniform(0.5, 1.0)),
        hold_delay=float(rng.uniform(0.2, 1.0)),
        min_gap=5.36,
        ball_blips=float(rng.uniform(0.0, 0.3)),
        blip_frames=(6, 12),
        blip_fast=100.0,
        soften_first=False,
        find_offset=int(rng.choice((-1, 0, 1))),
    )


SIXTH_ONLY = dict(wander_roll=True, follow_short_gap=True, false_floor_strict=True, blip_frames=(6, 16))
SEVENTH_ONLY = dict(SIXTH_ONLY, wander_soft=True, height_fall=12)


def sample_config_sixth(rng):
    """One random noise setting of the sixth generation: the fifth's, with the sixth's rules."""
    return replace(sample_config_fifth(rng), **SIXTH_ONLY)


def sample_config_seventh(rng):
    """One random noise setting of the seventh generation: the fifth's, with the seventh's rules."""
    return replace(sample_config_fifth(rng), **SEVENTH_ONLY)


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

def _wander(half, rng, start, stop, ball, roll=False, soft=False):
    """Where the ball is during a stoppage the tracker follows right through, (stop - start, 2).

    It starts where play stopped and ends where play restarts. In between it is moved about in
    short stages: left where it is, taken along by the nearest player, or passed to another one.
    For the last part of the stoppage it is at the restart spot, in the hands or at the feet of
    whoever will take the restart, and moves about a little with him.
    """
    t = half.n_frames
    length, width = half.pitch
    first = ball[start - 1] if start > 0 and np.isfinite(ball[start - 1, 0]) else None
    last = ball[stop] if stop < t and np.isfinite(ball[stop, 0]) else None
    if first is None and last is None:
        return None
    first = last if first is None else first
    last = first if last is None else last
    players = np.concatenate([half.home_xy, half.away_xy], axis=1)
    out = np.empty((stop - start, 2), dtype=np.float32)
    position = np.array(first, dtype=np.float32)
    # the ball reaches the restart spot some way through and waits there with the taker
    settle = max(_frames(0.3), int(rng.uniform(0.1, 0.7) * (stop - start)))
    frame = start
    velocity = None
    if roll and start > 2 and np.isfinite(ball[start - 1, 0]) and np.isfinite(ball[start - 3, 0]):
        # the ball does not stop because play does
        velocity = (ball[start - 1] - ball[start - 3]).astype(np.float32) * np.float32(FPS / 2.0)
        room = stop - start - settle - 2 * FPS
        if room > FPS // 2:
            stage = min(_frames(rng.uniform(0.6, 2.0)), room)
            seconds = (np.arange(1, stage + 1, dtype=np.float32) / np.float32(FPS))[:, None]
            if soft:
                # slowed at a steady rate, as grass slows a ball, and at rest by the stage's end
                speed = float(np.sqrt(np.sum(velocity * velocity)))
                slow = max(float(rng.uniform(0.4, 1.5)), speed / (stage / float(FPS)))
                moving = np.minimum(seconds, np.float32(speed / slow))
                heading = velocity / np.float32(speed) if speed > 0.05 else np.zeros(2, dtype=np.float32)
                out[:stage] = position + heading * (np.float32(speed) * moving - np.float32(0.5 * slow) * moving * moving)
            else:
                out[:stage] = position + velocity * (1.0 - np.exp(-seconds))
            position = out[stage - 1].copy()
            frame = start + stage
            velocity = None                         # spent: what follows starts from near rest
        elif soft and room > 0:
            settle += room                          # no room for a roll: the path bends to the spot
    while frame < stop:
        remaining = stop - frame
        row = frame - start
        if remaining <= settle + 2 * FPS:
            travel = max(1, min(remaining - settle, 3 * FPS))
            if soft:
                # not faster than a hard pass: the way to the restart spot takes the time it needs,
                # out of the time the ball would have waited there
                need = int(np.ceil(float(np.sqrt(np.sum((last - position) ** 2))) / 0.8))
                travel = max(travel, min(remaining, need))
            weight = (np.arange(1, travel + 1) / float(travel))[:, None]
            if velocity is not None and row == 0:
                # a short stoppage: the path leaves at the ball's own velocity and bends to the
                # restart spot (a cubic that has that velocity at its start and none at its end)
                u = weight.astype(np.float32)
                seconds = np.float32(travel / float(FPS))
                out[row:row + travel] = ((2 * u ** 3 - 3 * u ** 2 + 1) * position
                                         + (u ** 3 - 2 * u ** 2 + u) * seconds * velocity
                                         + (-2 * u ** 3 + 3 * u ** 2) * last)
            else:
                out[row:row + travel] = (1 - weight) * position + weight * last
            waiting = remaining - travel
            if waiting > 0:
                sway = _smooth_noise(rng, (waiting, 2), rng.uniform(0.3, 1.6), 1.5 * FPS)
                sway *= np.minimum(1.0, np.arange(waiting, 0, -1) / float(FPS))[:, None]
                if soft:
                    sway *= np.minimum(1.0, np.arange(1, waiting + 1) / float(FPS))[:, None]   # no hop
                out[row + travel:] = last + sway
            break
        stage = max(1, min(remaining - settle - 2 * FPS, _frames(rng.uniform(0.5, 3.5))))
        rows = slice(row, row + stage)
        delta = players[frame] - position
        with np.errstate(invalid="ignore"):
            distance = np.sqrt(np.sum(delta * delta, axis=1))
        seen = np.isfinite(distance)
        choice = rng.random()
        if choice < 0.1 or not seen.any():
            out[rows] = position
        elif choice < 0.5:
            # taken along by the nearest player, a little further from his feet than it started
            slot = int(np.nanargmin(distance))
            track = players[frame:frame + stage, slot]
            ok = np.isfinite(track[:, 0])
            track = np.where(ok[:, None], track, players[frame, slot])
            fade = np.linspace(1.0, 0.4, stage, dtype=np.float32)[:, None]
            out[rows] = track + (position - players[frame, slot]) * fade
        else:
            # passed to another player, or simply kicked away
            candidates = np.flatnonzero(seen & (distance > 4.0) & (distance < 40.0))
            if len(candidates):
                slot = int(rng.choice(candidates))
                target = players[min(t - 1, frame + stage), slot]
                if not np.isfinite(target[0]):
                    target = players[frame, slot]
            else:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                target = position + rng.uniform(5.0, 20.0) * np.array([np.cos(angle), np.sin(angle)])
            gap = float(np.sqrt(np.sum((target - position) ** 2)))
            speed = float(np.clip(np.exp(rng.normal(np.log(4.5), 0.5)), 1.5, 14.0))
            stage = max(1, min(stage, int(np.ceil(gap / speed * FPS)) + _frames(rng.uniform(0.0, 0.6))))
            if soft and gap > 20.0 * stage / float(FPS):
                # there is not the time to get there: it goes as far as a hard pass would
                target = position + (target - position) * np.float32(20.0 * stage / float(FPS) / gap)
                gap = 20.0 * stage / float(FPS)
            rows = slice(row, row + stage)
            travel = max(1, min(stage, int(np.ceil(gap / speed * FPS))))
            weight = np.minimum(np.arange(1, stage + 1) / float(travel), 1.0)[:, None]
            out[rows] = (1 - weight) * position + weight * target
        position = out[row + stage - 1].copy()
        frame += stage
    edge = np.array([length / 2 + 1.0, width / 2 + 1.0], dtype=np.float32)
    return np.clip(out, -edge, edge)


KICK_CHANGE = 4.0       # m/s: a change of the true ball's velocity over four frames that is a kick
KICK_REACH = 1.5        # m: with a player this near
AT_PLAYER = 1.0         # m: the ball has reached a player when one is this near
KICK_WAIT = 4.0         # s: how long a find or a loss is put off to wait for a kick or an arrival
HOLD_REACH = 1.5        # m: a ball this near a player
HOLD_SPEED = 2.5        # m/s: and slower than this is with him, not on its way


def _kicks_and_arrivals(ball, home_xy, away_xy):
    """(kick, near, nearest): frames at which the true ball is struck, frames on which it is at
    a player, and how far the nearest player is. A kick is a change of velocity of KICK_CHANGE or
    more over four frames with a player within KICK_REACH, taken where the change is largest."""
    t = len(ball)
    nearest = np.full(t, np.inf)
    for team in (home_xy, away_xy):
        delta = team - ball[:, None, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        nearest = np.minimum(nearest, np.where(np.isfinite(distance), distance, np.inf).min(axis=1))
    ball = ball.astype(np.float64)
    velocity = np.full(ball.shape, np.nan)
    velocity[1:-1] = (ball[2:] - ball[:-2]) * (FPS / 2.0)
    change = np.zeros(t)
    delta = velocity[4:] - velocity[:-4]
    change[2:-2] = np.nan_to_num(np.sqrt(np.sum(delta * delta, axis=1)))
    kick = (change >= KICK_CHANGE) & (nearest <= KICK_REACH)
    for offset in (1, 2):
        kick[offset:] &= change[offset:] >= change[:-offset]
        kick[:-offset] &= change[:-offset] > change[offset:]
    return kick, nearest <= AT_PLAYER, nearest


def _next(flag, frame, wait):
    """The first frame at or after `frame` where `flag` is true, or None if there is none within
    `wait` frames."""
    stop = min(len(flag), frame + wait + 1)
    if frame >= stop:
        return None
    hits = np.flatnonzero(flag[frame:stop])
    return frame + int(hits[0]) if len(hits) else None


def _hidden_ball(half, rng, config, ball, height, own):
    """(hidden, held, followed): frames on which the tracker reports no ball; frames of stoppages
    it follows right through on which a ball that is off the pitch is still reported, at its
    edge; and the (start, stop) of the stoppages it follows right through.

    `rng` makes the draws the simulator has always made, in the same order. Everything that was
    added later (the pick-up at a kick, the loss at a player, the late pick-up after a restart,
    the blips) draws from `own`, so the earlier generations come out as they did.
    """
    t = half.n_frames
    dead = (half.labels == D) if half.labels is not None else np.zeros(t, dtype=bool)
    hidden = np.zeros(t, dtype=bool)
    held = np.zeros(t, dtype=bool)
    followed = []
    stoppages = _runs(dead)

    length, width = half.pitch
    margin = np.minimum(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
    at_restart = config.find_at_kick if config.find_at_kick_restart < 0 else config.find_at_kick_restart
    snap = max(config.find_at_kick, at_restart, config.lose_at_player, config.lose_at_player_dead) > 0
    timing = config.find_lag_spread > 0
    wait = _frames(KICK_WAIT)
    if snap:
        kick, at_player, nearest = _kicks_and_arrivals(ball, half.home_xy, half.away_xy)
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
            watched = config.dead_watched_in if stop - start < 10 * FPS else config.dead_watched
        if config.min_gap > 0 and stop - start < _frames(config.min_gap):
            # A tracker that reports no gap this short bridges the stoppage. What it shows is
            # drawn as for a stoppage it follows: the source's own ball stands still there, or
            # is missing, and would give the stoppage away.
            watched = 1.0
        if rng.random() < watched:
            # This stoppage is tracked throughout.
            followed.append((start, stop))
            if rng.random() < config.hold_at_line:
                held[start:stop] = True
            continue
        if leaves:
            lose = start + _frames(rng.normal(config.lose_shift, config.lose_spread))
        else:
            # The ball rolls to a stop in view and is held for a few seconds before it is lost.
            lose = start + _frames(min(rng.exponential(config.linger), 12.0))
        if rng.random() < config.early_find:
            find = stop - _frames(rng.exponential(config.early_find_mean))
        else:
            # The ball is picked up once it is moving again after the restart.
            find = stop + _frames(rng.exponential(config.find_lag))
            late = False
            if timing:
                if own.random() < config.find_quick:
                    find = stop + _frames(own.exponential(config.find_lag_quick))
                else:
                    late = True
                    find = stop + _frames(float(np.exp(own.normal(np.log(config.find_lag),
                                                                  config.find_lag_spread))))
            missed = rng.random() < config.long_miss
            if missed:
                find += _frames(rng.exponential(8.0))
            if snap and own.random() < at_restart:
                # picked up as it is struck: at the restart itself, or at the first kick after
                # the time drawn when the pick-up is late
                if late or missed:
                    struck = _next(kick, max(0, find), wait)
                else:
                    struck = _next(kick, max(0, stop - config.restart_back),
                                   _frames(config.restart_wait))
                if struck is not None:
                    find = struck + config.find_offset
        if snap and own.random() < config.lose_at_player_dead:
            # lost as it reaches somebody: the player who fetches it, the keeper who gathers it
            reached = _next(at_player, max(0, lose), wait)
            if reached is not None and reached < find:
                lose = reached
        lose, find = max(0, lose), min(t, find)
        if config.follow_short_gap and config.min_gap > 0 and find - lose < _frames(config.min_gap):
            # too short a gap for this tracker to report: it follows the stoppage, and what it
            # shows is drawn, not taken from the source
            followed.append((start, stop))
            continue
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
            to_restart = rng.random() < config.until_restart
            if to_restart:
                stop = int(restarts[np.searchsorted(restarts, start, side="right")])
            else:
                length = np.exp(rng.normal(np.log(config.live_loss_median), config.live_loss_spread))
                stop = start + _frames(float(np.clip(length, 0.3, 90.0)))
            if snap:
                if own.random() < config.lose_at_player:
                    reached = _next(at_player, start, wait)
                    if reached is not None and reached < stop:
                        start = reached
                if to_restart and config.until_restart_as_restart:
                    if own.random() < at_restart:
                        struck = _next(kick, max(0, min(t - 1, stop) - config.restart_back),
                                       _frames(config.restart_wait))
                        if struck is not None:
                            stop = struck + config.find_offset
                elif own.random() < config.find_at_kick:
                    struck = _next(kick, min(t - 1, stop), wait)
                    if struck is not None:
                        stop = struck + config.find_offset
            hidden[start:min(t, stop)] = True
    return hidden, held, followed


def _speed(ball):
    """The ball's speed on every frame from the positions either side, 0 where it has none."""
    velocity = np.zeros_like(ball)
    velocity[1:-1] = (ball[2:] - ball[:-2]) * (FPS / 2.0)
    return np.nan_to_num(np.sqrt(np.sum(velocity * velocity, axis=1)))


def _blips(config, own, ball):
    """Frames lost for under a second, anywhere in the half: in a stoppage the tracker follows as
    well. With blip_fast they fall where the ball, as the tracker has it, is fast."""
    t = len(ball)
    brief = np.zeros(t, dtype=bool)
    if config.ball_blips > 0:
        chance = None
        if config.blip_fast > 0:
            # a hard pass, a shot, a cross
            chance = 1.0 + config.blip_fast * np.clip((_speed(ball) - 8.0) / 8.0, 0.0, 2.0)
            chance = chance / chance.sum()
        for _ in range(int(own.poisson(config.ball_blips * t / (FPS * 60.0)))):
            start = int(own.integers(0, t)) if chance is None else int(own.choice(t, p=chance))
            if config.blip_frames:
                frames = int(own.integers(config.blip_frames[0], config.blip_frames[1] + 1))
            else:
                frames = _frames(float(own.uniform(0.2, 0.9)))
            brief[start:min(t, start + frames)] = True
    return brief


def _resting(ball, home_xy, away_xy, config, stream):
    """Frames on which a ball that lies still within a player's reach is lost to the tracker.

    A ball at a player's feet or in his hands is not seen apart from him; it is found again as
    it is struck. The rule reads the ball and the players and nothing else: whether play is
    live does not come into it. A keeper holding the ball and a free kick waiting to be taken
    are lost in the same way, which is what the recording shows (most of its gaps of five to
    ten seconds begin and end in open field with the ball a few metres from where it was).
    """
    t = len(ball)
    out = np.zeros(t, dtype=bool)
    nearest = np.full(t, np.inf)
    for team in (home_xy, away_xy):
        delta = team - ball[:, None, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        nearest = np.minimum(nearest, np.where(np.isfinite(distance), distance, np.inf).min(axis=1))
    gone = np.full(t, np.inf)
    moved = ball[5:].astype(np.float64) - ball[:-5]
    gone[2:-3] = np.nan_to_num(np.sqrt(np.sum(moved * moved, axis=1)) * (FPS / 5.0), nan=np.inf)
    still = (nearest <= HOLD_REACH) & (gone < HOLD_SPEED)
    least = _frames(max(config.min_gap, 1.0))
    for start, stop in _runs(still):
        delay = _frames(float(stream.exponential(config.hold_delay))) if config.hold_delay > 0 else 0
        lost = stream.random() < config.hold_loss
        if lost and stop - start - delay >= least:
            out[start + delay:stop] = True
    return out


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
    # two streams of their own for what was added later, so that every earlier draw stays: the
    # first is the jitter's, as it always was, the second is for losing and finding the ball
    jitter_stream, own_stream, late_stream = rng.bit_generator.seed_seq.spawn(3)
    own = np.random.default_rng(own_stream)
    hidden, held, followed = _hidden_ball(half, rng, config, ball, height, own)
    # Through a stoppage the tracker follows, the ball is moved about by the players. Corrected
    # data freezes it or drops it there, so its path is drawn here.
    for start, stop in followed:
        path = _wander(half, rng, start, stop, ball, config.wander_roll, config.wander_soft)
        if path is not None:
            ball[start:stop] = path
            height[start:stop] = 0.0
            if config.height_fall > 0 and start > 0:
                frames = min(config.height_fall, stop - start)
                height[start:start + frames] = height[start - 1] * (
                    1.0 - np.arange(1, frames + 1, dtype=np.float32) / np.float32(config.height_fall))
            if config.wander_smooth > 1:
                # the drawn path and its joins with the true one, without a sudden change
                low = max(0, start - config.wander_smooth)
                high = min(t, stop + config.wander_smooth)
                ball[low:high] = _moving_average(ball[low:high], config.wander_smooth)
    unavailable = ~np.isfinite(ball[:, 0])              # the source itself has no ball here
    # On the path as it now stands, drawn stoppages and all: the losses of under a second, and
    # the ball that lies still with a player.
    brief = _blips(config, own, ball)
    hidden |= brief
    if config.hold_loss > 0:
        hidden |= _resting(ball, half.home_xy, half.away_xy, config, np.random.default_rng(late_stream))

    # -- where the ball is reported: single-camera parallax and a slow error ---------------
    camera = np.array([config.camera_x, -(width / 2 + config.camera_back)], dtype=np.float32)
    safe_height = np.clip(height, 0.0, config.camera_height * 0.8)
    stretch = config.parallax * safe_height / (config.camera_height - safe_height)
    reported = ball + stretch[:, None] * (ball - camera)
    reported += _smooth_noise(rng, reported.shape, config.ball_drift, 2 * FPS)
    average = _moving_average
    jitter = None
    if config.ball_jitter > 0:
        jitter = np.random.default_rng(jitter_stream).normal(
            0.0, config.ball_jitter, size=reported.shape).astype(np.float32)
    if config.edges == "through":
        reported = average(reported, config.ball_smooth)
        if config.ball_smooth2 > 1:
            reported = average(reported, config.ball_smooth2)
        if jitter is not None:
            reported = reported + jitter
    elif config.edges == "keep" and config.ball_smooth2 > 1 and config.soften_first:
        # A real kick is not as sudden as the open matches draw it. That is the ball, not the
        # tracker, so this short average runs over the true path, before anything is hidden.
        reported = average(reported, config.ball_smooth2)

    hidden |= unavailable

    # A tracker does not report a ball well off the pitch: it loses it, or, where it is
    # following the ball through a stoppage, keeps reporting it at the edge.
    margin = np.minimum(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
    with np.errstate(invalid="ignore"):
        hidden |= (margin < -config.out_margin) & ~held
    if config.min_gap > 0:
        # A tracker that joins its tracks up reports no short gap: a ball it loses for less
        # than min_gap it bridges, whatever made it lose the ball. Such a gap is given back, and
        # the true path stands for the bridge. Only the losses of under a second stay.
        least = _frames(config.min_gap)
        for start, stop in _runs(hidden):
            if stop - start < least and start > 0 and stop < t and not unavailable[start:stop].any():
                hidden[start:stop] = brief[start:stop]
    edge = np.array([length / 2 + 0.3, width / 2 + 0.3], dtype=np.float32)
    reported[held] = np.clip(reported[held], -edge, edge)
    reported[hidden] = np.nan
    if config.edges != "through":
        # the tracker smooths what it has: the frames it lost are gone before the average runs
        if config.edges == "keep":
            if config.ball_smooth2 > 1 and not config.soften_first:
                # inside each track, keeping its speed at both ends: nothing of the frames the
                # tracker did not have is in it
                reported = track_average(reported, config.ball_smooth2, end="carry")
            reported = track_average(reported, config.ball_smooth, config.track_hold)
            hidden = ~np.isfinite(reported[:, 0])
        else:
            reported = average(reported, config.ball_smooth)
            if config.ball_smooth2 > 1:
                reported = average(reported, config.ball_smooth2)
        if jitter is not None:
            reported = reported + jitter

    # -- a track that is not the ball -------------------------------------------------------
    minutes_hidden = hidden.sum() / (FPS * 60.0)
    gap_left = hidden.copy()                 # the gaps as they stand once false tracks are drawn in
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
        if config.min_gap > 0:
            # a tracker that reports no short gap does not report one beside a false track
            # either: the pieces of the gap left on both sides are long, or there are none
            shortest = _frames(config.min_gap) - config.track_hold
            around = gap_left if config.false_floor_strict else hidden
            low = start
            while low > 0 and around[low - 1]:
                low -= 1
            high = min(stop, t)
            while high < t and around[high]:
                high += 1
            if 0 < start - low < shortest or (around[min(stop, t) - 1] and 0 < high - min(stop, t) < shortest):
                continue
            if config.false_floor_strict and not around[start]:
                continue                            # this spot already has a false track
        reported[rows][keep] = fake[keep]
        gap_left[rows] = False

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
        if config.player_smooth > 1 and config.edges == "through":
            for slot in range(xy.shape[1]):
                xy[:, slot] = _moving_average(xy[:, slot], config.player_smooth)
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
        if config.player_smooth > 1 and config.edges != "through":
            for slot in range(xy.shape[1]):
                if config.edges == "keep":
                    xy[:, slot] = track_average(xy[:, slot], config.player_smooth, config.track_hold)
                    if config.player_smooth2 > 1:
                        xy[:, slot] = track_average(xy[:, slot], config.player_smooth2)
                else:
                    xy[:, slot] = average(xy[:, slot], config.player_smooth)
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
