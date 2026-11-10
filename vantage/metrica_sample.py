"""Load Metrica Sports' public sample games (other matches) into labelled Half objects.

Source: https://github.com/metrica-sports/sample-data — three anonymised matches of Metrica's
corrected tracking with synchronised events, published for the analytics community ("If you use
it for anything public, please acknowledge the source"). They matter here for one reason: they
are corrected tracking from the same provider that produced the challenge's ground truth, so
labels built from them follow that provider's own idea of possession and of a dead ball.

How the labels are built (no hand labelling; everything comes from the published files):

* **Dead ball.** In this corrected tracking the ball has no position while play is stopped: it
  disappears when the ball goes out or a foul is given and reappears at the restart. A run of
  frames without a ball is therefore `D`. Runs shorter than half a second are ordinary momentary
  gaps inside live play and are left alive.
* **Team.** The event stream names the team for every action on the ball. While play is live the
  ball belongs to the team of the most recent such action (set piece, pass, carry, shot, recovery,
  ball lost, ball out). Challenges, cards and fouls received do not move possession by
  themselves, which is the provider's published definition: an opponent's touch that does not
  change the outcome is not a change of possession.
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import A, D, FPS, H, X
from . import epts
from .half import Half, home_attacks_positive_x

GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)
MIN_DEAD_FRAMES = 12
ON_BALL = {"SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}


def _metres(xy):
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (xy[..., 1] - 0.5) * PITCH[1]
    return out


def _read_csv_tracking(path):
    """(period, frame, players (T, S, 2), ball (T, 2)) from a Sample Game 1/2 tracking CSV."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))[3:]
    data = np.array([[float(cell) if cell not in ("", "NaN") else np.nan for cell in row]
                     for row in rows], dtype=np.float64)
    period = data[:, 0].astype(np.int64)
    frame = data[:, 1].astype(np.int64)
    values = data[:, 3:]
    players = values[:, :-2].reshape(len(values), -1, 2)
    ball = values[:, -2:]
    return period, frame, players, ball


def _read_csv_events(path):
    events = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            events.append({"team": row["Team"].strip().lower(), "type": row["Type"].strip(),
                           "frame": int(row["Start Frame"])})
    return events


def _labels(frame, ball, events):
    """H/A/D for one period from ball presence and the event stream."""
    t = len(frame)
    missing = np.isnan(ball[:, 0])
    dead = np.zeros(t, dtype=bool)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= MIN_DEAD_FRAMES:
            dead[start:stop] = True

    owner = np.full(t, -1, dtype=np.int8)
    first, last = int(frame[0]), int(frame[-1])
    for event in sorted(events, key=lambda e: e["frame"]):
        if event["type"] not in ON_BALL or not first <= event["frame"] <= last:
            continue
        owner[event["frame"] - first:] = H if event["team"] == "home" else A

    labels = np.where(dead, D, owner).astype(np.int8)
    # Live frames before the first action of a stretch take the team of that first action.
    unknown = labels == -1
    if unknown.any():
        index = np.arange(t)
        known = np.where(~unknown & ~dead, index, t)
        nxt = np.minimum.accumulate(known[::-1])[::-1]
        fill = unknown & (nxt < t)
        labels[fill] = labels[nxt[fill]]
        labels[labels == -1] = X
    return labels


def _bridge_short_gaps(ball):
    """Fill momentary gaps in the ball (shorter than a stoppage) by straight lines."""
    out = ball.copy()
    missing = np.isnan(out[:, 0])
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < MIN_DEAD_FRAMES and start > 0 and stop < len(out):
            weight = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
            out[start:stop] = (1 - weight) * out[start - 1] + weight * out[stop]
    return out


def _half(game, period, frame, home, away, ball, events):
    labels = _labels(frame, ball, events)
    ball = _bridge_short_gaps(ball)
    home_xy, away_xy, ball_xy = _metres(home), _metres(away), _metres(ball)
    flipped = not home_attacks_positive_x(home_xy, away_xy)
    if flipped:
        home_xy, away_xy, ball_xy = -home_xy, -away_xy, -ball_xy
    return Half(match="metrica{}".format(game), period=period, pitch=PITCH, ball_xy=ball_xy,
                home_xy=home_xy, away_xy=away_xy, labels=labels,
                first_frame=int(frame[0]), meta={"source": "metrica-sample", "flipped": flipped}
                ).check()


def load_game(game, root):
    """[Half, Half] for one sample game. `root` is the cloned sample-data repository."""
    folder = os.path.join(root, "data", "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_".format(game))
    halves = []
    if game in (1, 2):
        period, frame, home, ball = _read_csv_tracking(stem + "RawTrackingData_Home_Team.csv")
        _, _, away, _ = _read_csv_tracking(stem + "RawTrackingData_Away_Team.csv")
        events = _read_csv_events(stem + "RawEventsData.csv")
        for number in (1, 2):
            rows = period == number
            halves.append(_half(game, number, frame[rows], home[rows], away[rows], ball[rows],
                                events))
    else:
        data = epts.read(stem + "metadata.xml", stem + "tracking.txt")
        root = ET.parse(stem + "metadata.xml").getroot()
        home_id = next(root.iter("Team")).get("id")
        with open(stem + "events.json") as handle:
            raw = json.load(handle)["data"]
        events = [{"team": "home" if e["team"]["id"] == home_id else "away",
                   "type": e["type"]["name"], "frame": int(e["start"]["frame"])} for e in raw]
        is_home = data.team == 0
        for number, key in ((1, "first_half"), (2, "second_half")):
            start = int(data.parameters[key + "_start"])
            end = int(data.parameters[key + "_end"])
            rows = (data.frame >= start) & (data.frame <= end)
            halves.append(_half(game, number, data.frame[rows],
                                data.xy[rows][:, is_home].astype(np.float64),
                                data.xy[rows][:, ~is_home].astype(np.float64),
                                data.ball[rows].astype(np.float64), events))
    return halves


def with_dead_ball_track(half, rng):
    """Give the ball a plausible position while play is stopped.

    The sample tracking has no ball at all during stoppages, which automatic tracking does not
    share: a camera can still see a ball rolling out or sitting on the corner spot. Left as it
    is, "no ball" would mean "dead ball" far more reliably here than in any real recording.
    So each stoppage gets a simple, physical stand-in: the ball rolls on from where play stopped
    and comes to rest, is out of sight while it is fetched, and then sits still on the spot the
    restart is taken from. Stoppages under three seconds are joined by a straight line. The
    noise simulator then hides and jitters this track like any other.
    """
    ball = half.ball_xy.astype(np.float32).copy()
    labels = half.labels
    t = half.n_frames
    edges = np.diff(np.concatenate(([0], (labels == D).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        before = ball[start - 1] if start > 0 else None
        after = ball[stop] if stop < t else None
        if before is not None and not np.isfinite(before[0]):
            before = None
        if after is not None and not np.isfinite(after[0]):
            after = None
        length = stop - start
        if before is not None and after is not None and length < 3 * FPS:
            weight = (np.arange(1, length + 1) / float(length + 1))[:, None]
            ball[start:stop] = (1 - weight) * before + weight * after
            continue
        if before is not None:
            roll = min(int(rng.uniform(1.0, 4.0) * FPS), int(0.4 * length))
            if roll > 0:
                velocity = np.zeros(2, dtype=np.float32)
                if start > 2 and np.isfinite(ball[start - 3, 0]):
                    velocity = (ball[start - 1] - ball[start - 3]) * (FPS / 2.0)
                seconds = np.arange(1, roll + 1) / float(FPS)
                ball[start:start + roll] = before + velocity * (1.0 - np.exp(-seconds))[:, None]
        if after is not None:
            rest = min(int(rng.uniform(1.5, 12.0) * FPS), int(0.6 * length))
            if rest > 0:
                ball[stop - rest:stop] = after
    return Half(match=half.match, period=half.period, pitch=half.pitch, ball_xy=ball,
                home_xy=half.home_xy, away_xy=half.away_xy, labels=half.labels,
                ball_z=np.zeros(t, dtype=np.float32), first_frame=half.first_frame,
                meta=dict(half.meta, dead_ball="synthetic"))
