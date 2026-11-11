"""The second recording of the match: its tracking brought onto the tactical video's frames.

The match was filmed twice. The tactical recording is the reference (row i of a submission is its
frame i). The broadcast recording has its own automatic tracking, "not frame-aligned to the
tactical video, and no mapping table is provided. It has cuts, replays and zooms" (DATA.md). It
is offered as optional extra input, and it matters for one thing: the tactical tracking has no
ball on four frames in ten, and a second camera does not lose the ball at the same moments.

Nothing here uses a label or a video. Both trackings place the same players on the same pitch
at the same moments, so the record of one that belongs to a frame of the other is the one whose
players stand where that frame's players stand. That is found in three steps:

1. **Roughly, a minute at a time.** For each minute of broadcast records, the stretch of the
   tactical tracking in which the ball follows the same path, tried at every possible shift.
2. **Exactly, frame by frame.** A broadcast record is kept only if its players have tactical
   players of the same team standing where they stand, at the shift its minute found. A replay
   shows an earlier moment, so its players are in the wrong places and it is dropped.
3. **The ball.** Where the tactical tracking has no ball and a kept broadcast record has one,
   the broadcast ball is used, moved by however much the players around it are displaced
   between the two trackings (the two cameras' pitch maps differ by a metre or so).

The thresholds are distances a person can check against the size of a pitch and a player; none
was fitted to anything.
"""

import os

import numpy as np

from . import epts

CHUNK = 1500            # records in a minute
STEP = 750
NEAR = 2.5              # m: a broadcast player has a tactical partner if one of his team stands this near
ENOUGH = 0.6            # share of a record's players that must have a partner
LEAST_PLAYERS = 4       # a record with fewer players in view says too little
AROUND_BALL = 20.0      # m: the players whose displacement moves the broadcast ball


def paths(data_dir):
    stem = os.path.join(data_dir, "broadcast", "BAR-ATH_broadcast")
    return {"meta": stem + "_atd.xml", "raw": stem + "_atd.txt", "homography": stem + "_homography.csv"}


def in_metres(xy, pitch):
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * pitch[0]
    out[..., 1] = (xy[..., 1] - 0.5) * pitch[1]
    return out


def load(data_dir, pitch):
    """The broadcast tracking in metres: players (N, Q, 2), their team (Q,), ball (N, 2)."""
    files = paths(data_dir)
    data = epts.read(files["meta"], files["raw"])
    return {"xy": in_metres(data.xy, pitch), "team": data.team.copy(), "ball": in_metres(data.ball, pitch),
            "frame": data.frame.copy(), "team_names": data.team_names}


def _correlate(long, short):
    """sum_j short[j] * long[j + d] for every shift d at which the short one fits inside."""
    size = 1 << int(np.ceil(np.log2(len(long) + len(short))))
    spectrum = np.fft.rfft(long, size) * np.conj(np.fft.rfft(short, size))
    return np.fft.irfft(spectrum, size)[:len(long) - len(short) + 1]


def ball_shift(tactical_ball, piece, least=250):
    """For one stretch of broadcast ball positions, how far apart the two balls are, on average,
    at each shift: (root mean square distance (D,), frames compared (D,)).

    Shift d puts the stretch's first record on tactical frame d.
    """
    seen_t = np.isfinite(tactical_ball[:, 0])
    seen_b = np.isfinite(piece[:, 0])
    a = np.where(seen_t[:, None], tactical_ball, 0.0).astype(np.float64)
    b = np.where(seen_b[:, None], piece, 0.0).astype(np.float64)
    mask_t, mask_b = seen_t.astype(np.float64), seen_b.astype(np.float64)
    count = _correlate(mask_t, mask_b)
    total = np.zeros_like(count)
    for axis in (0, 1):
        total += _correlate(mask_t, mask_b * b[:, axis] ** 2)
        total -= 2.0 * _correlate(a[:, axis], b[:, axis])
        total += _correlate(a[:, axis] ** 2, mask_b)
    count = np.round(count)
    with np.errstate(invalid="ignore", divide="ignore"):
        apart = np.sqrt(np.maximum(total, 0.0) / count)
    apart[count < least] = np.inf
    return apart, count


def rough_shifts(tactical_ball, broadcast_ball, chunk=CHUNK, step=STEP):
    """Minute by minute: [(first record, best shift, distance there, distance at the best shift
    elsewhere, frames compared)]. "Elsewhere" is more than ten seconds from the best shift: a
    minute whose ball path fits two places equally well has not been placed."""
    out = []
    for start in range(0, len(broadcast_ball) - chunk + 1, step):
        apart, count = ball_shift(tactical_ball, broadcast_ball[start:start + chunk])
        if not np.isfinite(apart).any():
            out.append((start, -1, np.inf, np.inf, 0))
            continue
        best = int(np.argmin(apart))
        other = apart.copy()
        other[max(0, best - 250):best + 251] = np.inf
        out.append((start, best, float(apart[best]), float(other.min()), int(count[best])))
    return out


def partners(tactical_xy, tactical_team, broadcast_xy, broadcast_team, near=NEAR):
    """For records and frames already paired (same length): for each record, how many of its
    players are in view and how many of them have a tactical player of their team within `near`
    metres. Returns (in view (N,), with a partner (N,), mean displacement (N, 2))."""
    count = np.zeros(len(broadcast_xy), dtype=np.int32)
    matched = np.zeros(len(broadcast_xy), dtype=np.int32)
    moved = np.zeros((len(broadcast_xy), 2), dtype=np.float64)
    for team in (0, 1):
        ours = broadcast_xy[:, broadcast_team == team]            # (N, q, 2)
        theirs = tactical_xy[:, tactical_team == team]            # (N, p, 2)
        if ours.shape[1] == 0 or theirs.shape[1] == 0:
            continue
        delta = theirs[:, None, :, :] - ours[:, :, None, :]       # (N, q, p, 2)
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        distance = np.where(np.isfinite(distance), distance, np.inf)
        nearest = distance.argmin(axis=2)                          # (N, q)
        least = np.take_along_axis(distance, nearest[..., None], axis=2)[..., 0]
        in_view = np.isfinite(ours[..., 0])
        good = in_view & (least <= near)
        count += in_view.sum(axis=1)
        matched += good.sum(axis=1)
        step = np.take_along_axis(delta, nearest[..., None, None].repeat(2, axis=-1), axis=2)[:, :, 0, :]
        moved += np.where(good[..., None], step, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        moved = moved / np.maximum(matched, 1)[:, None]
    return count, matched, moved


def flips(xy, flip):
    """The same positions with the pitch turned: flip is (turn x, turn y)."""
    out = xy.copy()
    if flip[0]:
        out[..., 0] = -out[..., 0]
    if flip[1]:
        out[..., 1] = -out[..., 1]
    return out


def paired(tactical_xy, tactical_team, broadcast_xy, broadcast_team, shift, block=3000, records=None):
    """`partners` for every broadcast record against tactical frame record + shift.

    Returns (in view, with a partner, mean displacement), each with one row per record; a record
    whose frame would fall outside the tactical video has nobody in view. `records` limits the
    work to some records (an index array); the others are left at zero.
    """
    total = len(broadcast_xy)
    count = np.zeros(total, dtype=np.int32)
    matched = np.zeros(total, dtype=np.int32)
    moved = np.zeros((total, 2), dtype=np.float64)
    rows = np.arange(total) if records is None else np.asarray(records)
    rows = rows[(rows + shift >= 0) & (rows + shift < len(tactical_xy))]
    for start in range(0, len(rows), block):
        part = rows[start:start + block]
        c, m, d = partners(tactical_xy[part + shift], tactical_team, broadcast_xy[part], broadcast_team)
        count[part], matched[part], moved[part] = c, m, d
    return count, matched, moved


def best_shift(tactical_xy, tactical_team, broadcast_xy, broadcast_team, around, reach=15, every=7):
    """The shift near `around` at which most broadcast players have a partner, from every
    `every`-th record. Returns (shift, {shift: share of players with a partner})."""
    rows = np.arange(0, len(broadcast_xy), every)
    shares = {}
    for shift in range(around - reach, around + reach + 1):
        count, matched, _ = paired(tactical_xy, tactical_team, broadcast_xy, broadcast_team, shift, records=rows)
        shares[shift] = float(matched.sum()) / max(1, int(count.sum()))
    return max(shares, key=shares.get), shares


def place_records(tactical_xy, tactical_team, broadcast_xy, broadcast_team, shifts):
    """For every broadcast record, the tactical frame it shows, or -1.

    `shifts` are the few shifts the rough step found (one per stretch of the broadcast that runs
    without a break). A record takes the shift at which the larger share of its players have a
    partner, and is kept if that share is at least ENOUGH of at least LEAST_PLAYERS players.
    Returns (tactical frame (N,), mean displacement of the record's players (N, 2)).
    """
    total = len(broadcast_xy)
    frame = np.full(total, -1, dtype=np.int64)
    best_share = np.zeros(total)
    displacement = np.zeros((total, 2), dtype=np.float64)
    for shift in shifts:
        count, matched, moved = paired(tactical_xy, tactical_team, broadcast_xy, broadcast_team, shift)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(count >= LEAST_PLAYERS, matched / np.maximum(count, 1), 0.0)
        better = (share >= ENOUGH) & (share > best_share)
        frame[better] = np.flatnonzero(better) + shift
        displacement[better] = moved[better]
        best_share[better] = share[better]
    return frame, displacement
