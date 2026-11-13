"""Turn one Half into a per-frame feature table.

Design rules, all of which exist because the tracking is automatic:

* **No player identities.** Every player feature is a property of the *set* of players a team has
  on a frame (nearest to the ball, average position, spread, how many are walking), so the order
  and number of track slots never matters.
* **The ball is often missing.** Nothing here interpolates it. A frame without a ball gets NaN for
  the ball's own measurements, plus explicit "gap" features: how long since it was last seen, how
  long until it is seen again, and what the picture looked like at both ends of the gap.
* **Offline.** A frame may look at what happens after it. Possession is decided for a whole
  recording at once, so the seconds after a touch are as usable as the seconds before it.
* **Symmetric.** Home attacks +x (see half.py). The away team's numbers are computed the same way,
  so swapping the teams and rotating the pitch gives the mirrored table.

The gradient-boosted model takes NaN natively, so missing stays missing all the way through.
"""

import warnings

import numpy as np

from . import A, FPS, H, corners

# -- parameters --------------------------------------------------------------------------------
# These describe measurement, not football: how far apart two samples must be to estimate a
# velocity from noisy positions, and what speed no real object reaches.
PLAYER_VEL_HALFSPAN = 5        # frames either side: velocity over 0.4 s
PLAYER_MAX_SPEED = 13.0        # m/s; faster means the slot changed person
BALL_VEL_HALFSPAN = 2          # frames either side: velocity over 0.16 s
BALL_MAX_SPEED = 45.0          # m/s
BALL_ACC_CAP = 30.0            # m/s^2

CONTROL_RADIUS = 2.0           # m: a player this close to the ball may be controlling it
CONTROL_MARGIN = 0.7           # m: and must be this much closer than any opponent
STILL_SPEED = 0.8              # m/s: below this the ball counts as at rest
TOUCH_HALFSPAN = 4             # frames either side: velocity change over about a third of a second
TOUCH_DELTA = 3.0              # m/s: a change of velocity at least this large is a touch
TOUCH_RADIUS = 3.0             # m: with a player this close
TOUCH_MARGIN = 0.3             # m: and that much nearer than any opponent
TOUCH_RUN_CAP = 6.0            # touches in a row are counted up to this
TIME_CAP = 30.0                # s: "how long since / until" features saturate here
TOUCH_WIDE = 10                # frames either side: a touch as a path smoothed over 21 frames shows it
CARRY_RADIUS = 1.5             # m: the ball this close to a player
CARRY_SPEED = 2.0              # m/s: and moving with him to within this
CARRY_MARGIN = 0.5             # m: and that much nearer than any opponent
END_BACK = 20                  # frames before the end of a track at which its motion is read: the
                               # tracker slows every track to a stop over its last 16 frames
START_IN = 3                   # frames into a track at which its motion is read: the first frame
                               # at which the ball has a velocity

CONTEXT_WINDOWS = (12, 37, 125)        # centred rolling means: about 1 s, 3 s, 10 s wide
TREND_WINDOWS = (25, 75, 250)          # mean of the next w frames minus mean of the last w


# -- small array tools -------------------------------------------------------------------------

def _central_velocity(xy, halfspan, max_speed):
    """Velocity by central difference; NaN where either sample is missing or the speed is absurd."""
    velocity = np.full(xy.shape, np.nan, dtype=np.float32)
    if xy.shape[0] > 2 * halfspan:
        delta = xy[2 * halfspan:] - xy[:-2 * halfspan]
        velocity[halfspan:-halfspan] = delta * (FPS / (2.0 * halfspan))
    speed = np.sqrt(np.sum(velocity * velocity, axis=-1))
    velocity[speed > max_speed] = np.nan
    return velocity


def _frames_since(flag):
    """For each frame, frames since `flag` was last true (0 where it is true); inf before the first."""
    index = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, index, -1))
    out = (index - last).astype(np.float32)
    out[last < 0] = np.inf
    return out, last


def _frames_until(flag):
    since, last = _frames_since(flag[::-1])
    n = len(flag)
    nxt = np.where(last[::-1] >= 0, n - 1 - last[::-1], -1)
    return since[::-1], nxt


def _seconds(frames):
    return np.minimum(frames / float(FPS), TIME_CAP).astype(np.float32)


def _rolling_mean(values, before, after):
    """Mean over [t - before, t + after], ignoring NaN. NaN where the window has no value."""
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0), dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - before)
    high = np.minimum(len(values), index + after + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[high] - total[low]) / n
    mean[n == 0] = np.nan
    return mean.astype(np.float32)


def _take(values, index):
    """values[index], with NaN where index is -1."""
    out = np.full(len(index), np.nan, dtype=np.float32)
    good = index >= 0
    out[good] = values[index[good]]
    return out


def _run_length(flag):
    """Length in frames of the run of true values each frame sits in (0 where false)."""
    n = len(flag)
    padded = np.concatenate(([False], flag, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(n, dtype=np.float32)
    for start, stop in zip(starts, stops):
        out[start:stop] = stop - start
    return out


# -- per-team blocks ---------------------------------------------------------------------------

def _team_block(xy, velocity, direction, pitch):
    """Collective description of one team. `direction` is +1 for home, -1 for away, so that
    'forward' and 'own goal' mean the same thing for both sides."""
    length, width = pitch
    x = xy[..., 0] * direction
    y = xy[..., 1]
    present = np.isfinite(x)
    n = present.sum(axis=1).astype(np.float32)

    speed = np.sqrt(np.sum(velocity * velocity, axis=-1))
    forward_speed = velocity[..., 0] * direction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid_x = np.nanmean(x, axis=1)
        centroid_y = np.nanmean(y, axis=1)
        spread_x = np.nanstd(x, axis=1)
        spread_y = np.nanstd(y, axis=1)
        mean_speed = np.nanmean(speed, axis=1)
        median_speed = np.nanmedian(speed, axis=1)
        mean_forward = np.nanmean(forward_speed, axis=1)
        slow = np.nanmean(np.where(np.isfinite(speed), speed < 1.0, np.nan), axis=1)
        fast = np.nanmean(np.where(np.isfinite(speed), speed > 4.0, np.nan), axis=1)

    # Rank players along the pitch. The deepest is normally the goalkeeper, so the second
    # deepest is the defensive line; the most advanced is the front of the team.
    filled_low = np.sort(np.where(present, x, np.inf), axis=1)
    filled_high = -np.sort(np.where(present, -x, np.inf), axis=1)
    deepest = np.where(n >= 1, filled_low[:, 0], np.nan)
    line = np.where(n >= 2, filled_low[:, min(1, x.shape[1] - 1)], np.nan)
    front = np.where(n >= 1, filled_high[:, 0], np.nan)

    third = length / 6.0
    in_own_third = np.sum(present & (x < -third), axis=1).astype(np.float32)
    in_final_third = np.sum(present & (x > third), axis=1).astype(np.float32)
    in_opp_half = np.sum(present & (x > 0), axis=1).astype(np.float32)
    box = present & (x > length / 2 - 16.5) & (np.abs(y) < 20.16)
    in_opp_box = np.sum(box, axis=1).astype(np.float32)

    touch_margin = np.where(present, width / 2 - np.abs(y), np.inf)
    nearest_touch = np.min(touch_margin, axis=1)
    nearest_touch[n == 0] = np.nan

    return {
        "n": n, "cx": centroid_x, "cy": centroid_y, "sx": spread_x, "sy": spread_y,
        "deepest": deepest, "line": line, "front": front,
        "speed": mean_speed, "speed_med": median_speed, "fwd": mean_forward,
        "slow": slow, "fast": fast,
        "own_third": in_own_third, "final_third": in_final_third,
        "opp_half": in_opp_half, "opp_box": in_opp_box, "touch": nearest_touch,
    }


def _ball_to_team(ball_xy, xy, velocity, ball_velocity):
    """Distances from the ball to one team's players, and what the nearest one is doing."""
    delta = xy - ball_xy[:, None, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    dist = np.where(np.isfinite(dist), dist, np.inf)
    order = np.sort(dist, axis=1)
    nearest = order[:, 0]
    second = order[:, 1] if dist.shape[1] > 1 else np.full(len(dist), np.inf, dtype=np.float32)
    within2 = np.sum(dist < 2.0, axis=1).astype(np.float32)
    within5 = np.sum(dist < 5.0, axis=1).astype(np.float32)

    pick = np.argmin(dist, axis=1)
    rows = np.arange(len(dist))
    nearest_velocity = velocity[rows, pick]
    nearest_speed = np.sqrt(np.sum(nearest_velocity * nearest_velocity, axis=-1))
    relative = nearest_velocity - ball_velocity
    relative_speed = np.sqrt(np.sum(relative * relative, axis=-1))

    missing = ~np.isfinite(nearest) | ~np.isfinite(ball_xy[:, 0])
    out = {"d1": nearest, "d2": second, "n2": within2, "n5": within5,
           "near_speed": nearest_speed, "rel_speed": relative_speed}
    for key in out:
        value = out[key].astype(np.float32)
        value[missing] = np.nan
        value[~np.isfinite(value)] = np.nan
        out[key] = value
    return out


def _touch_features(ball_velocity, seen, nearest_home, nearest_away, ball=None,
                    halfspan=None, prefix="tch_", sequence="tq_"):
    """Features built on touches: who struck the ball, how often in a row, and when.

    Distance alone cannot tell a player running beside the ball from one playing it. A touch can:
    the ball's velocity changes abruptly, and someone is next to it when it does. The challenge's
    own idea of controlled possession is phrased in touches ("a team completes more than two
    consecutive passes"), so the run of touches by one team is given to the model directly.

    A touch is a local peak in the change of the ball's velocity over about a third of a second,
    of at least TOUCH_DELTA, with a player within TOUCH_RADIUS. It belongs to the nearer team;
    when both are equally near it belongs to neither and breaks no run.
    """
    t = len(seen)
    span = TOUCH_HALFSPAN if halfspan is None else halfspan
    change = np.full(t, np.nan, dtype=np.float32)
    if t > 2 * span:
        delta = ball_velocity[2 * span:] - ball_velocity[:-2 * span]
        change[span:-span] = np.sqrt(np.sum(delta * delta, axis=1))
    strength = np.nan_to_num(change, nan=0.0)

    # Local peaks: the largest change within the span either side, above the threshold.
    peak = strength >= TOUCH_DELTA
    for offset in range(1, span + 1):
        peak[offset:] &= strength[offset:] >= strength[:-offset]
        peak[:-offset] &= strength[:-offset] > strength[offset:]

    home_near = np.where(np.isfinite(nearest_home), nearest_home, np.inf)
    away_near = np.where(np.isfinite(nearest_away), nearest_away, np.inf)
    by_home = peak & (home_near < TOUCH_RADIUS) & (home_near + TOUCH_MARGIN < away_near)
    by_away = peak & (away_near < TOUCH_RADIUS) & (away_near + TOUCH_MARGIN < home_near)

    frames = np.flatnonzero(by_home | by_away)
    teams = np.where(by_home[frames], 1, -1).astype(np.int8)        # +1 home, -1 away

    out = {}
    since_home, _ = _frames_since(by_home)
    since_away, _ = _frames_since(by_away)
    until_home, _ = _frames_until(by_home)
    until_away, _ = _frames_until(by_away)
    out["tch_since_h"] = _seconds(since_home)
    out["tch_since_a"] = _seconds(since_away)
    out["tch_until_h"] = _seconds(until_home)
    out["tch_until_a"] = _seconds(until_away)
    for window in (75, 250):
        for name, flag in (("h", by_home), ("a", by_away)):
            series = flag.astype(np.float32)
            out["tch_{}_past{}".format(name, window)] = _rolling_mean(series, window, 0) * (window + 1)
            out["tch_{}_next{}".format(name, window)] = _rolling_mean(series, 0, window) * (window + 1)

    last_team = np.full(t, np.nan, dtype=np.float32)
    next_team = np.full(t, np.nan, dtype=np.float32)
    run_last = np.full(t, np.nan, dtype=np.float32)
    run_next = np.full(t, np.nan, dtype=np.float32)
    flight = np.full(t, np.nan, dtype=np.float32)
    if len(frames):
        # How many touches in a row by the same team, counted up to and from each touch.
        back = np.ones(len(frames), dtype=np.float32)
        for i in range(1, len(frames)):
            if teams[i] == teams[i - 1]:
                back[i] = back[i - 1] + 1
        forward = np.ones(len(frames), dtype=np.float32)
        for i in range(len(frames) - 2, -1, -1):
            if teams[i] == teams[i + 1]:
                forward[i] = forward[i + 1] + 1

        index = np.arange(t)
        previous = np.searchsorted(frames, index, side="right") - 1      # last touch at or before
        following = previous + 1
        has_previous = previous >= 0
        has_following = following < len(frames)
        last_team[has_previous] = teams[previous[has_previous]]
        run_last[has_previous] = np.minimum(back[previous[has_previous]], TOUCH_RUN_CAP)
        next_team[has_following] = teams[following[has_following]]
        run_next[has_following] = np.minimum(forward[following[has_following]], TOUCH_RUN_CAP)
        both = has_previous & has_following
        flight[both] = (frames[following[both]] - frames[previous[both]]) / float(FPS)
    out["tch_last"] = last_team
    out["tch_next"] = next_team
    out["tch_run_last"] = run_last
    out["tch_run_next"] = run_next
    out["tch_flight"] = np.minimum(flight, TIME_CAP)
    # Signed: +n when the last / next n touches in a row are home's, -n when away's.
    out["tch_run_last_signed"] = last_team * run_last
    out["tch_run_next_signed"] = next_team * run_next
    if prefix != "tch_":
        out = {prefix + key[4:]: value for key, value in out.items()}
    if ball is not None:
        out.update(_touch_sequence(t, frames, teams, strength, ball_velocity, ball,
                                   home_near, away_near, span, sequence))
    return out


def _touch_sequence(t, frames, teams, strength, ball_velocity, ball, home_near, away_near, span,
                    prefix="tq_"):
    """The touches either side of a frame, one by one, and what kind of touch the nearest two were.

    A possession of a second or two is a touch by one team between touches of the other. The
    features above say who touched last and next and how long ago; these add the touch before
    that and the touch after, so that 'home, away, home within two seconds' is visible in one
    row, and they describe the last and the next touch themselves: how hard the ball was struck,
    how fast it came and went, whether it turned, how near the toucher and the nearest opponent
    were, and whether it left towards the opponents' goal. A deflection and a played ball differ
    in exactly these.

    Team-signed features are +1 home, -1 away. 'fwd' and 'x' are from the toucher's side, so they
    read the same for both teams.
    """
    out = {}
    index = np.arange(t)
    n = len(frames)
    names_seq = []
    for k in (1, 2, 3):
        for side in ("p", "n"):
            names_seq.append(prefix + "{}{}_team".format(side, k))
            names_seq.append(prefix + "{}{}_dt".format(side, k))
    quality = ("str", "sin", "sout", "turn", "own", "opp", "fwd", "x")
    for side in ("p1", "n1"):
        for name in quality:
            names_seq.append(prefix + "{}_{}".format(side, name))
    names_seq += [prefix + "lone_last", prefix + "lone_next"]
    for name in names_seq:
        out[name] = np.full(t, np.nan, dtype=np.float32)
    if not n:
        return out

    before = np.clip(frames - span, 0, t - 1)
    after = np.clip(frames + span, 0, t - 1)
    v_in, v_out = ball_velocity[before], ball_velocity[after]
    s_in = np.sqrt(np.sum(v_in * v_in, axis=1))
    s_out = np.sqrt(np.sum(v_out * v_out, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.sum(v_in * v_out, axis=1) / (s_in * s_out)
    turn[~np.isfinite(turn)] = np.nan
    sign = teams.astype(np.float32)
    home_d = np.where(np.isfinite(home_near[frames]), home_near[frames], np.nan)
    away_d = np.where(np.isfinite(away_near[frames]), away_near[frames], np.nan)
    per_touch = {
        "str": np.minimum(strength[frames], 30.0),
        "sin": s_in, "sout": s_out, "turn": turn,
        "own": np.where(teams > 0, home_d, away_d),
        "opp": np.where(teams > 0, away_d, home_d),
        "fwd": v_out[:, 0] * sign,
        "x": ball[frames, 0] * sign,
    }

    previous = np.searchsorted(frames, index, side="right") - 1      # last touch at or before
    for k in (1, 2, 3):
        pk = previous - (k - 1)
        ok = pk >= 0
        out[prefix + "p{}_team".format(k)][ok] = sign[pk[ok]]
        out[prefix + "p{}_dt".format(k)][ok] = np.minimum((index[ok] - frames[pk[ok]]) / float(FPS), TIME_CAP)
        nk = previous + k
        ok = nk < n
        out[prefix + "n{}_team".format(k)][ok] = sign[nk[ok]]
        out[prefix + "n{}_dt".format(k)][ok] = np.minimum((frames[nk[ok]] - index[ok]) / float(FPS), TIME_CAP)
    ok = previous >= 0
    for name in quality:
        out[prefix + "p1_" + name][ok] = per_touch[name][previous[ok]]
    following = previous + 1
    ok = following < n
    for name in quality:
        out[prefix + "n1_" + name][ok] = per_touch[name][following[ok]]

    # a touch that stands alone between two touches of the other team
    lone = np.zeros(n, dtype=np.float32)
    if n >= 3:
        middle = (teams[1:-1] != teams[:-2]) & (teams[1:-1] != teams[2:])
        lone[1:-1] = np.where(middle, sign[1:-1], 0.0)
    ok = previous >= 0
    out[prefix + "lone_last"][ok] = lone[previous[ok]]
    ok = following < n
    out[prefix + "lone_next"][ok] = lone[following[ok]]
    return out


EXTENDED = ("tcw_", "tq_", "car_", "rb_", "tcr_", "rq_", "ed_", "b_kink", "b_dspeed4", "b_turn4")


def is_extended(name):
    """Whether a column is one build() leaves out with extended=False."""
    if name.startswith(EXTENDED) or name.endswith(("_rel_speed_past12", "_rel_speed_next12")):
        return True
    stem, _, last = name.rpartition("_")
    return bool(stem) and last[:3] in ("atm", "atp") and last[3:].isdigit()


def restart_features(t, restarts):
    """Four features from tagged restarts: how far each frame is from one, and whose it is.

    `restarts` is a list of (row, team): the row at which a tag says play resumed after a
    stoppage, and the team that restarted (H, A or None when the tag does not say). A set-piece
    tag is evidence of two things at once: the seconds before it were a stoppage, and from it
    on the ball is alive. Without tags the features sit at their far value and say nothing.
    """
    until = np.full(t, TIME_CAP, dtype=np.float32)
    since = np.full(t, TIME_CAP, dtype=np.float32)
    next_team = np.zeros(t, dtype=np.float32)
    last_team = np.zeros(t, dtype=np.float32)
    index = np.arange(t)
    for row, team in sorted(restarts, key=lambda item: item[0]):
        row = int(min(max(row, 0), t - 1))
        sign = {H: 1.0, A: -1.0}.get(team, 0.0)
        ahead = np.minimum((row - index) / float(FPS), TIME_CAP)
        closer = (index <= row) & (ahead < until)
        until[closer] = ahead[closer]
        next_team[closer] = sign
        behind = np.minimum((index - row) / float(FPS), TIME_CAP)
        closer = (index >= row) & (behind <= since)
        since[closer] = behind[closer]
        last_team[closer] = sign
    next_team[until >= TIME_CAP] = 0.0
    last_team[since >= TIME_CAP] = 0.0
    return {"sp_until": until, "sp_since": since, "sp_next_team": next_team,
            "sp_last_team": last_team}


# -- the table ---------------------------------------------------------------------------------

def build(half, extended=True):
    """(names, table) for one Half. table is float32, one row per frame, NaN = not available.

    With extended=False the table holds the 255 columns the first models were trained on, and
    those columns are the same numbers either way (`is_extended` says which a column is).
    """
    length, width = half.pitch
    ball = half.ball_xy.astype(np.float32)
    t = half.n_frames
    seen = np.isfinite(ball[:, 0])

    home_velocity = _central_velocity(half.home_xy, PLAYER_VEL_HALFSPAN, PLAYER_MAX_SPEED)
    away_velocity = _central_velocity(half.away_xy, PLAYER_VEL_HALFSPAN, PLAYER_MAX_SPEED)
    ball_velocity = _central_velocity(ball, BALL_VEL_HALFSPAN, BALL_MAX_SPEED)
    ball_speed = np.sqrt(np.sum(ball_velocity * ball_velocity, axis=-1))
    ball_acc_vector = _central_velocity(ball_velocity, BALL_VEL_HALFSPAN, 1e9)
    # Capped: how hard the ball was struck matters less than that it was, and the extreme values
    # depend on how much a tracker smooths its ball path.
    ball_acc = np.minimum(np.sqrt(np.sum(ball_acc_vector * ball_acc_vector, axis=-1)), BALL_ACC_CAP)

    f = {}

    # -- the ball itself ----------------------------------------------------------------------
    f["b_x"] = ball[:, 0]
    f["b_yabs"] = np.abs(ball[:, 1])
    margin_x = length / 2 - np.abs(ball[:, 0])
    margin_y = width / 2 - np.abs(ball[:, 1])
    f["b_margin_x"] = margin_x
    f["b_margin_y"] = margin_y
    f["b_margin"] = np.minimum(margin_x, margin_y)
    f["b_speed"] = ball_speed
    f["b_vx"] = ball_velocity[:, 0]
    f["b_acc"] = ball_acc

    home = _ball_to_team(ball, half.home_xy, home_velocity, ball_velocity)
    away = _ball_to_team(ball, half.away_xy, away_velocity, ball_velocity)
    for key, value in home.items():
        f["bh_" + key] = value
    for key, value in away.items():
        f["ba_" + key] = value
    f["b_dmin"] = np.fmin(home["d1"], away["d1"])
    f["b_ddiff"] = away["d1"] - home["d1"]           # > 0: a home player is nearer

    # -- when the ball is not there -----------------------------------------------------------
    since_seen, last_seen = _frames_since(seen)
    until_seen, next_seen = _frames_until(seen)
    f["seen"] = seen.astype(np.float32)
    f["gap_since"] = _seconds(since_seen)
    f["gap_until"] = _seconds(until_seen)
    f["gap_len"] = _seconds(since_seen + until_seen)
    with np.errstate(invalid="ignore", divide="ignore"):
        position = since_seen / (since_seen + until_seen)
    f["gap_pos"] = np.where(seen, 0.0, np.where(np.isfinite(position), position, np.nan)).astype(
        np.float32)

    # Where on the pitch the ball was: restarts are taken from recognisable places.
    corner = np.sqrt(np.square(length / 2 - np.abs(ball[:, 0]))
                     + np.square(width / 2 - np.abs(ball[:, 1])))
    f["b_corner"] = corner
    f["b_centre"] = np.sqrt(np.sum(ball * ball, axis=1))
    # Speed towards the nearest boundary line: positive when the ball is on its way out.
    toward_x = np.sign(ball[:, 0]) * ball_velocity[:, 0]
    toward_y = np.sign(ball[:, 1]) * ball_velocity[:, 1]
    f["b_outward"] = np.where(margin_x < margin_y, toward_x, toward_y).astype(np.float32)

    for name in ("b_x", "b_yabs", "b_margin", "b_speed", "b_ddiff", "b_dmin", "b_vx", "b_corner",
                 "b_centre", "b_outward"):
        f["last_" + name] = _take(f[name], last_seen)
        f["next_" + name] = _take(f[name], next_seen)

    # The run of visible ball a frame belongs to, mirrored from the gap features above.
    since_hidden, _ = _frames_since(~seen)
    until_hidden, _ = _frames_until(~seen)
    f["vis_since"] = _seconds(since_hidden)
    f["vis_until"] = _seconds(until_hidden)
    f["vis_len"] = _seconds(since_hidden + until_hidden)
    jump = _take(ball[:, 0], next_seen) - _take(ball[:, 0], last_seen)
    jump_y = _take(ball[:, 1], next_seen) - _take(ball[:, 1], last_seen)
    f["gap_dx"] = jump
    f["gap_dist"] = np.sqrt(jump * jump + jump_y * jump_y)

    # -- who last had it, who has it next ------------------------------------------------------
    with np.errstate(invalid="ignore"):
        control_home = seen & (home["d1"] < CONTROL_RADIUS) & (
            home["d1"] + CONTROL_MARGIN < np.where(np.isfinite(away["d1"]), away["d1"], np.inf))
        control_away = seen & (away["d1"] < CONTROL_RADIUS) & (
            away["d1"] + CONTROL_MARGIN < np.where(np.isfinite(home["d1"]), home["d1"], np.inf))
    since_home, _ = _frames_since(control_home)
    since_away, _ = _frames_since(control_away)
    until_home, _ = _frames_until(control_home)
    until_away, _ = _frames_until(control_away)
    f["ctl_since_h"] = _seconds(since_home)
    f["ctl_since_a"] = _seconds(since_away)
    f["ctl_until_h"] = _seconds(until_home)
    f["ctl_until_a"] = _seconds(until_away)
    with np.errstate(invalid="ignore"):                 # inf - inf: neither team yet
        f["ctl_last"] = np.sign(since_away - since_home).astype(np.float32)   # +1 home, -1 away
        f["ctl_next"] = np.sign(until_away - until_home).astype(np.float32)
    f["ctl_run_h"] = _run_length(control_home) / FPS
    f["ctl_run_a"] = _run_length(control_away) / FPS
    for window in TREND_WINDOWS:
        f["ctl_h_past{}".format(window)] = _rolling_mean(control_home.astype(np.float32), window, 0)
        f["ctl_a_past{}".format(window)] = _rolling_mean(control_away.astype(np.float32), window, 0)
        f["ctl_h_next{}".format(window)] = _rolling_mean(control_home.astype(np.float32), 0, window)
        f["ctl_a_next{}".format(window)] = _rolling_mean(control_away.astype(np.float32), 0, window)

    # -- touches: the moments the ball's motion changes next to a player -------------------------
    f.update(_touch_features(ball_velocity, seen, home["d1"], away["d1"]))

    if extended:
        # -- touches as a smoothed path shows them --------------------------------------------
        # The recording's tracker spreads a kick over 21 frames, so over the 8 frames the touch
        # features above look at, a change of velocity shows two fifths of itself. The same
        # features are built again over TOUCH_WIDE frames either side, where it shows whole,
        # and the touch-by-touch features are built on those.
        f.update(_touch_features(ball_velocity, seen, home["d1"], away["d1"], ball,
                                 halfspan=TOUCH_WIDE, prefix="tcw_"))

        # -- the ball carried: next to a player and moving with him ----------------------------
        with np.errstate(invalid="ignore"):
            away_d1 = np.where(np.isfinite(away["d1"]), away["d1"], np.inf)
            home_d1 = np.where(np.isfinite(home["d1"]), home["d1"], np.inf)
            carry_home = seen & (home["d1"] < CARRY_RADIUS) & (home["rel_speed"] < CARRY_SPEED) & (
                home["d1"] + CARRY_MARGIN < away_d1)
            carry_away = seen & (away["d1"] < CARRY_RADIUS) & (away["rel_speed"] < CARRY_SPEED) & (
                away["d1"] + CARRY_MARGIN < home_d1)
        for name, flag in (("h", carry_home), ("a", carry_away)):
            series = flag.astype(np.float32)
            f["car_since_" + name] = _seconds(_frames_since(flag)[0])
            f["car_until_" + name] = _seconds(_frames_until(flag)[0])
            f["car_run_" + name] = _run_length(flag) / FPS
            for window in (12, 37):
                f["car_{}_past{}".format(name, window)] = _rolling_mean(series, window, 0)
                f["car_{}_next{}".format(name, window)] = _rolling_mean(series, 0, window)

        # -- the same instant signals a few frames either side, not averaged -------------------
        def shifted(values, offset):
            out = np.full(t, np.nan, dtype=np.float32)
            if offset > 0:
                out[:-offset] = values[offset:]
            else:
                out[-offset:] = values[:offset]
            return out

        for name in ("b_ddiff", "bh_d1", "ba_d1", "b_speed"):
            series = f[name].astype(np.float32)
            for offset in (-12, -6, -3, 3, 6, 12):
                f["{}_at{}{}".format(name, "p" if offset > 0 else "m", abs(offset))] = shifted(
                    series, offset)

        # -- how sharply the ball's motion is changing, as a number and not a yes or no --------
        # The touch features above need a change of TOUCH_DELTA with one team clearly nearer.
        # Half of the changes of possession in the open matches have no such touch within a
        # tenth of a second, so the raw signal is given too, at three time scales.
        def kink(halfspan):
            out = np.full(t, np.nan, dtype=np.float32)
            if t > 2 * halfspan:
                delta = ball_velocity[2 * halfspan:] - ball_velocity[:-2 * halfspan]
                out[halfspan:-halfspan] = np.minimum(np.sqrt(np.sum(delta * delta, axis=1)), 30.0)
            return out

        def rolling_max(values, before, after):
            best = np.full(t, -np.inf, dtype=np.float32)
            filled = np.where(np.isfinite(values), values, -np.inf).astype(np.float32)
            for offset in range(-before, after + 1):
                if offset == 0:
                    best = np.maximum(best, filled)
                elif offset > 0:
                    best[:-offset] = np.maximum(best[:-offset], filled[offset:])
                else:
                    best[-offset:] = np.maximum(best[-offset:], filled[:offset])
            best[~np.isfinite(best)] = np.nan
            return best

        for halfspan in (2, 4, 8, 12):
            f["b_kink{}".format(halfspan)] = kink(halfspan)
        f["b_kink8_pmax"] = rolling_max(f["b_kink8"], 12, -1)      # the 12 frames before this one
        f["b_kink8_nmax"] = rolling_max(f["b_kink8"], -1, 12)      # the 12 frames after it
        speed_after = shifted(ball_speed.astype(np.float32), 4)
        speed_before = shifted(ball_speed.astype(np.float32), -4)
        f["b_dspeed4"] = speed_after - speed_before
        v_before = np.stack([shifted(ball_velocity[:, 0], -4), shifted(ball_velocity[:, 1], -4)], axis=1)
        v_after = np.stack([shifted(ball_velocity[:, 0], 4), shifted(ball_velocity[:, 1], 4)], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            turn = np.sum(v_before * v_after, axis=1) / (speed_before * speed_after)
        f["b_turn4"] = np.where(np.isfinite(turn), turn, np.nan).astype(np.float32)
        # the ball moving with the nearest player of each team, a little earlier and later
        for name in ("bh_rel_speed", "ba_rel_speed"):
            series = f[name].astype(np.float32)
            f[name + "_past12"] = _rolling_mean(series, 12, 0)
            f[name + "_next12"] = _rolling_mean(series, 0, 12)

        # -- the ball with its corners put back ----------------------------------------------
        # Who is at the ball when it is struck is read off a path whose corners the tracker
        # has cut. With the corners put back the same questions are asked again: how near is
        # each team, who controls it, who touched it.
        rebuilt, _ = corners.rebuild(ball, ball_velocity, FPS, TOUCH_WIDE, TOUCH_DELTA)
        home_r = _ball_to_team(rebuilt, half.home_xy, home_velocity, ball_velocity)
        away_r = _ball_to_team(rebuilt, half.away_xy, away_velocity, ball_velocity)
        f["rb_dh"] = home_r["d1"]
        f["rb_da"] = away_r["d1"]
        f["rb_dmin"] = np.fmin(home_r["d1"], away_r["d1"])
        f["rb_ddiff"] = away_r["d1"] - home_r["d1"]
        moved = rebuilt - ball
        f["rb_moved"] = np.sqrt(np.sum(moved * moved, axis=1))
        for name in ("rb_ddiff", "rb_dh", "rb_da"):
            series = f[name].astype(np.float32)
            for offset in (-12, -6, -3, 3, 6, 12):
                f["{}_at{}{}".format(name, "p" if offset > 0 else "m", abs(offset))] = shifted(
                    series, offset)
        with np.errstate(invalid="ignore"):
            near_home = np.where(np.isfinite(home_r["d1"]), home_r["d1"], np.inf)
            near_away = np.where(np.isfinite(away_r["d1"]), away_r["d1"], np.inf)
            held_home = seen & (near_home < CONTROL_RADIUS) & (near_home + CONTROL_MARGIN < near_away)
            held_away = seen & (near_away < CONTROL_RADIUS) & (near_away + CONTROL_MARGIN < near_home)
        for name, flag in (("h", held_home), ("a", held_away)):
            f["rb_ctl_since_" + name] = _seconds(_frames_since(flag)[0])
            f["rb_ctl_until_" + name] = _seconds(_frames_until(flag)[0])
            f["rb_ctl_run_" + name] = _run_length(flag) / FPS
        with np.errstate(invalid="ignore"):
            f["rb_ctl_last"] = np.sign(_frames_since(held_away)[0]
                                       - _frames_since(held_home)[0]).astype(np.float32)
            f["rb_ctl_next"] = np.sign(_frames_until(held_away)[0]
                                       - _frames_until(held_home)[0]).astype(np.float32)
        corner_touches = _touch_features(ball_velocity, seen, home_r["d1"], away_r["d1"], rebuilt,
                                         halfspan=TOUCH_WIDE, prefix="tcr_", sequence="rq_")
        for key, value in corner_touches.items():
            # how hard and which way the ball went is already there from the touches above
            if key.startswith("rq_") and key.rsplit("_", 1)[-1] in ("str", "sin", "sout", "turn", "fwd"):
                continue
            f[key] = value

    if extended:
        # -- how a track of the ball ended, and how the next one began --------------------------
        # A ball that was heading out of play when it was lost and one that was lost among
        # players are different gaps, and a ball found coming in from the line is a restart. The
        # last frames of a track cannot say which: the velocity has no value at the very end,
        # and the tracker slows every track to a stop over its last 16 frames whatever the ball
        # was doing. So the track is read END_BACK frames before its end and carried on from
        # there, and START_IN frames after its beginning and carried back.
        index = np.arange(t)
        with np.errstate(invalid="ignore"):
            track_start = np.where(seen, index - np.where(np.isfinite(since_hidden), since_hidden, index + 1) + 1,
                                   next_seen).astype(np.int64)
            track_end = np.where(seen, index + np.where(np.isfinite(until_hidden), until_hidden, t - index) - 1,
                                 last_seen).astype(np.int64)
        # the track a frame belongs to, or for a frame without a ball the one before it (its
        # end) and the one after it (its start)
        ends_at = track_end
        starts_at = track_start
        length_before = np.where(ends_at >= 0, _take(since_hidden, np.maximum(ends_at, 0)), 0.0)
        length_after = np.where(starts_at >= 0, _take(until_hidden, np.maximum(starts_at, 0)), 0.0)
        read_end = np.where((ends_at >= 0) & (np.nan_to_num(length_before, nan=0.0, posinf=1e9) > END_BACK + BALL_VEL_HALFSPAN),
                            ends_at - END_BACK, -1)
        read_start = np.where((starts_at >= 0) & (starts_at + START_IN < t)
                              & (np.nan_to_num(length_after, nan=0.0, posinf=1e9) > START_IN + BALL_VEL_HALFSPAN),
                              starts_at + START_IN, -1)

        def carried(read, seconds):
            """The ball's place at `read`, carried on for `seconds` at the velocity it has there."""
            x = _take(ball[:, 0], read) + _take(ball_velocity[:, 0], read) * np.float32(seconds)
            y = _take(ball[:, 1], read) + _take(ball_velocity[:, 1], read) * np.float32(seconds)
            return np.minimum(length / 2 - np.abs(x), width / 2 - np.abs(y)).astype(np.float32)

        f["ed_lost_speed"] = _take(ball_speed, read_end)
        f["ed_lost_vx"] = _take(ball_velocity[:, 0], read_end)
        f["ed_lost_outward"] = _take(f["b_outward"], read_end)
        f["ed_lost_ahead"] = carried(read_end, END_BACK / float(FPS))            # where it would be at the end
        f["ed_lost_ahead2"] = carried(read_end, (END_BACK + 25) / float(FPS))    # and a second later
        f["ed_found_speed"] = _take(ball_speed, read_start)
        f["ed_found_vx"] = _take(ball_velocity[:, 0], read_start)
        f["ed_found_outward"] = _take(f["b_outward"], read_start)
        f["ed_found_back"] = carried(read_start, -(START_IN + 12) / float(FPS))  # where it came from

    # -- a ball at rest ------------------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        still = seen & (ball_speed < STILL_SPEED)
    f["still_run"] = _run_length(still) / FPS
    f["still_since"] = _seconds(_frames_since(still)[0])
    f["still_until"] = _seconds(_frames_until(still)[0])
    with np.errstate(invalid="ignore"):
        outside = seen & (f["b_margin"] < 0)
    f["out_since"] = _seconds(_frames_since(outside)[0])
    f["out_until"] = _seconds(_frames_until(outside)[0])

    # -- the two teams -------------------------------------------------------------------------
    home_team = _team_block(half.home_xy, home_velocity, +1.0, half.pitch)
    away_team = _team_block(half.away_xy, away_velocity, -1.0, half.pitch)
    for key, value in home_team.items():
        f["h_" + key] = value.astype(np.float32)
    for key, value in away_team.items():
        f["a_" + key] = value.astype(np.float32)

    f["t_n"] = home_team["n"] + away_team["n"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f["t_speed"] = np.nanmean(np.stack([home_team["speed"], away_team["speed"]]), axis=0)
    f["t_slow"] = (home_team["slow"] + away_team["slow"]) / 2.0
    # Both centroids on one axis: home's forward is +x, away's forward is -x.
    f["t_gap_x"] = (-away_team["cx"]) - home_team["cx"]
    f["t_push"] = home_team["fwd"] - away_team["fwd"]     # > 0: home going forward, away retreating
    f["t_field"] = (home_team["cx"] + (-away_team["cx"])) / 2.0   # where the game is, along x
    f["t_touch"] = np.fmin(home_team["touch"], away_team["touch"])
    f["t_touch_side"] = np.sign(away_team["touch"] - home_team["touch"]).astype(np.float32)

    # -- one number per gap and per run: what the players and the ball did throughout ----------
    def per_run(flag, series):
        out = np.full(t, np.nan, dtype=np.float32)
        edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            window = series[start:stop]
            if np.isfinite(window).any():
                out[start:stop] = np.nanmean(window)
        return out

    f["gap_speed"] = per_run(~seen, f["t_speed"])       # how fast the players moved while it was gone
    f["gap_slow"] = per_run(~seen, f["t_slow"])
    f["gap_push"] = per_run(~seen, f["t_push"])
    f["run_ctl_h"] = per_run(seen, control_home.astype(np.float32))   # who was on the ball in this run
    f["run_ctl_a"] = per_run(seen, control_away.astype(np.float32))
    f["run_speed"] = per_run(seen, f["b_speed"])

    # -- context: the same signals a little earlier and later ----------------------------------
    context = ("b_speed", "b_margin", "b_dmin", "b_ddiff", "seen", "t_speed", "t_slow", "t_push",
               "h_fwd", "a_fwd", "h_cx", "a_cx", "t_gap_x", "t_field", "h_sx", "a_sx", "t_touch")
    for name in context:
        series = f[name].astype(np.float32)
        for window in CONTEXT_WINDOWS:
            f["{}_m{}".format(name, window)] = _rolling_mean(series, window, window)
        for window in TREND_WINDOWS:
            past = _rolling_mean(series, window, 0)
            future = _rolling_mean(series, 0, window)
            f["{}_d{}".format(name, window)] = future - past

    # -- tagged restarts, where the source has them ------------------------------------------------
    f.update(restart_features(t, half.meta.get("restarts", ())))

    names = sorted(f)
    table = np.empty((t, len(names)), dtype=np.float32)
    for column, name in enumerate(names):
        values = np.asarray(f[name], dtype=np.float32)
        values = np.where(np.isfinite(values), values, np.nan)
        table[:, column] = values
    return names, table
