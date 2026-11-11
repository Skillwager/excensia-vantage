"""Load SkillCorner's open A-League matches (other games, MIT licence) into labelled Half objects.

Source: https://github.com/SkillCorner/opendata (fetched by scripts/fetch_skillcorner.py). Twenty
matches of the 2024/2025 Australian A-League, each with

    {id}_match.json                   teams, players, pitch size, the frames each half spans
    {id}_tracking_extrapolated.jsonl  ten frames a second, made automatically from the broadcast
                                      picture: players on screen are detected, the others are
                                      extrapolated and marked; the ball where it is seen
    {id}_dynamic_events.csv           among others every "player possession": a player in control
                                      of the ball, with its first frame and his team
    {id}_phases_of_play.csv           the phases of play, which exist only while the ball is in play

They matter for one reason: this is real automatic tracking, with the noise such tracking has.
The other external matches are corrected tracking put through a simulator of that noise.

How the labels are built (no hand labelling; everything comes from the published files):

* **Dead ball.** A frame inside a half that no phase of play covers is `D`: "when the ball is
  out of play there is no phase of play" (the source's README).
* **Team.** While play is live the ball belongs to the team of the latest player possession. A
  touch that does not give a player control is not a player possession in this data, which is
  the challenge's own rule that such a touch does not change possession. The frames of a
  stretch of play before its first player possession belong to the team that took the restart:
  the source says of that first possession whether the restart was for its team or against it.
"""

import csv
import json
import os

import numpy as np

from . import A, D, FPS, H, X
from .half import Half, home_attacks_positive_x

IDS = (2017461, 2016236, 2015213, 2013725, 2011166, 2010085, 2007721, 2007448, 2006363, 2006229,
       1996435, 1996436, 1986691, 1959846, 1953632, 1927964, 1925299, 1899585, 1886347, 1874553)
MATCHES = tuple("sc{}".format(number) for number in IDS)
SOURCE_FPS = 10
CACHE_VERSION = 2
POSSESSION = "player_possession"


def _stem(match, root):
    number = match[2:] if match.startswith("sc") else str(match)
    return os.path.join(root, "data", "matches", number, number)


def _sheet(match, root):
    """Teams, players and the frames of each half, from the match file."""
    with open(_stem(match, root) + "_match.json", encoding="utf-8") as handle:
        sheet = json.load(handle)
    home, away = int(sheet["home_team"]["id"]), int(sheet["away_team"]["id"])
    slots = {H: {}, A: {}}
    for player in sheet["players"]:
        side = H if int(player["team_id"]) == home else A
        slots[side].setdefault(int(player["id"]), len(slots[side]))
    periods = {int(period["period"]): (int(period["start_frame"]), int(period["end_frame"]))
               for period in sheet["match_periods"] if int(period["period"]) in (1, 2)}
    return {"home": home, "away": away, "slots": slots, "periods": periods,
            "pitch": (float(sheet["pitch_length"]), float(sheet["pitch_width"]))}


def _read_tracking(path, sheet):
    """{period: arrays at the source's ten frames a second}."""
    out = {}
    for period, (first, last) in sheet["periods"].items():
        t = last - first + 1
        out[period] = {
            "first": first,
            "home": np.full((t, len(sheet["slots"][H]), 2), np.nan, dtype=np.float32),
            "away": np.full((t, len(sheet["slots"][A]), 2), np.nan, dtype=np.float32),
            "home_seen": np.zeros((t, len(sheet["slots"][H])), dtype=bool),
            "away_seen": np.zeros((t, len(sheet["slots"][A])), dtype=bool),
            "ball": np.full((t, 2), np.nan, dtype=np.float32),      # where the ball is detected
            "ball_any": np.full((t, 2), np.nan, dtype=np.float32),  # wherever a position is given
            "group": np.full(t, -1, dtype=np.int8),                 # the per-frame possession flag
        }
    spans = [(first, last, period) for period, (first, last) in sheet["periods"].items()]
    home_slot, away_slot = sheet["slots"][H], sheet["slots"][A]
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            number = frame["frame"]
            period = next((p for first, last, p in spans if first <= number <= last), None)
            if period is None:
                continue
            arrays = out[period]
            row = number - arrays["first"]
            for player in frame.get("player_data") or ():
                x, y = player.get("x"), player.get("y")
                if x is None or y is None:
                    continue
                identity = player.get("player_id")
                if identity in home_slot:
                    arrays["home"][row, home_slot[identity]] = (x, y)
                    arrays["home_seen"][row, home_slot[identity]] = bool(player.get("is_detected"))
                elif identity in away_slot:
                    arrays["away"][row, away_slot[identity]] = (x, y)
                    arrays["away_seen"][row, away_slot[identity]] = bool(player.get("is_detected"))
            ball = frame.get("ball_data") or {}
            if ball.get("x") is not None and ball.get("y") is not None:
                arrays["ball_any"][row] = (ball["x"], ball["y"])
                if ball.get("is_detected"):
                    arrays["ball"][row] = (ball["x"], ball["y"])
            group = (frame.get("possession") or {}).get("group")
            if group == "home team":
                arrays["group"][row] = H
            elif group == "away team":
                arrays["group"][row] = A
    return out


def _read_events(match, root, sheet):
    """(phases, possessions): [(first frame, last frame, team)] and
    [(first frame, last frame, team, the interruption before it or "")], team as H or A."""
    side = {sheet["home"]: H, sheet["away"]: A}
    stem = _stem(match, root)
    phases = []
    with open(stem + "_phases_of_play.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phases.append((int(row["frame_start"]), int(row["frame_end"]),
                           side.get(int(float(row["team_in_possession_id"])), X)))
    possessions = []
    with open(stem + "_dynamic_events.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] != POSSESSION:
                continue
            possessions.append((int(row["frame_start"]), int(row["frame_end"]),
                                side.get(int(float(row["team_id"])), X),
                                row.get("game_interruption_before") or ""))
    return sorted(phases), sorted(possessions)


def restart_team(team, interruption):
    """Who took the restart before a stretch's first player possession by `team`.

    The source names the interruption from that team's side: "throw_in_for" was its own throw,
    "throw_in_against" the other team's, which it then won. A goal is the other way round: after
    a "goal_against" the team that conceded kicks off.
    """
    other = A if team == H else H
    if interruption in ("goal_for", "goal_against"):       # not "goal_kick_...", which is a restart like any other
        return team if interruption == "goal_against" else other
    return other if interruption.endswith("_against") else team


def labels_at_source_rate(first, last, phases, possessions):
    """H/A/D for the frames first..last of one half, at the source's frame rate."""
    t = last - first + 1
    live = np.zeros(t, dtype=bool)
    for start, stop, _ in phases:
        if stop < first or start > last:
            continue
        live[max(start, first) - first:min(stop, last) - first + 1] = True
    owner = np.full(t, -1, dtype=np.int8)
    # every stretch of live play starts with nobody: possession does not carry over a stoppage
    edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    stretches = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    inside = [(start - first, team, interruption) for start, _, team, interruption in possessions
              if first <= start <= last and team in (H, A)]
    index = 0
    for begin, end in stretches:
        while index < len(inside) and inside[index][0] < begin:
            index += 1
        opened = False
        while index < len(inside) and inside[index][0] < end:
            row, team, interruption = inside[index]
            if not opened:
                owner[begin:row] = restart_team(team, interruption)
                opened = True
            owner[row:end] = team
            index += 1
    labels = np.where(live, owner, D).astype(np.int8)
    labels[labels == -1] = X        # a stretch of play in which the source names no possession
    return labels


def _to_engine_rate(values):
    from .anymatch import resample

    return resample(values, SOURCE_FPS, FPS)


def _build_half(match, period, arrays, sheet, phases, possessions):
    first = arrays["first"]
    last = first + len(arrays["ball"]) - 1
    coarse = labels_at_source_rate(first, last, phases, possessions)
    home_xy, away_xy = _to_engine_rate(arrays["home"]), _to_engine_rate(arrays["away"])
    ball_xy = _to_engine_rate(arrays["ball"])
    count = len(ball_xy)
    nearest = np.minimum(np.floor(np.arange(count) * (SOURCE_FPS / FPS) + 0.5).astype(np.int64),
                         len(coarse) - 1)
    labels = coarse[nearest]
    flipped = not home_attacks_positive_x(home_xy, away_xy)
    if flipped:
        home_xy, away_xy, ball_xy = -home_xy, -away_xy, -ball_xy
    return Half(match=match, period=period, pitch=sheet["pitch"], ball_xy=ball_xy, home_xy=home_xy,
                away_xy=away_xy, labels=labels, first_frame=first,
                meta={"source": "skillcorner", "flipped": flipped,
                      "detected": float(np.mean(np.concatenate([arrays["home_seen"].ravel(),
                                                                arrays["away_seen"].ravel()])))}).check()


def load_match(match, root, cache_dir=None):
    """[Half, Half] for one match. Cached after the first read."""
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "skillcorner_{}_v{}.npz".format(match, CACHE_VERSION))
        if os.path.isfile(cache_path):
            return _load_cache(cache_path)
    sheet = _sheet(match, root)
    raw = _read_tracking(_stem(match, root) + "_tracking_extrapolated.jsonl", sheet)
    phases, possessions = _read_events(match, root, sheet)
    halves = [_build_half(match, period, raw[period], sheet, phases, possessions)
              for period in sorted(raw)]
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        payload = {"match": np.array(match)}
        for half in halves:
            key = "p{}_".format(half.period)
            payload[key + "ball_xy"] = half.ball_xy
            payload[key + "home_xy"] = half.home_xy
            payload[key + "away_xy"] = half.away_xy
            payload[key + "labels"] = half.labels
            payload[key + "info"] = np.array([half.pitch[0], half.pitch[1], half.first_frame,
                                              1.0 if half.meta["flipped"] else 0.0,
                                              half.meta["detected"]])
        np.savez_compressed(cache_path, **payload)
    return halves


def _load_cache(path):
    with np.load(path) as data:
        match = str(data["match"])
        halves = []
        for period in (1, 2):
            key = "p{}_".format(period)
            if key + "ball_xy" not in data:
                continue
            info = data[key + "info"]
            halves.append(Half(
                match=match, period=period, pitch=(float(info[0]), float(info[1])),
                ball_xy=data[key + "ball_xy"], home_xy=data[key + "home_xy"],
                away_xy=data[key + "away_xy"], labels=data[key + "labels"],
                first_frame=int(info[2]),
                meta={"source": "skillcorner", "flipped": bool(info[3]),
                      "detected": float(info[4])}).check())
    return halves
