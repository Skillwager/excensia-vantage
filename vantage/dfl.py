"""Load the open DFL matches (other games, CC BY 4.0) into Half objects.

Source: Bassek, Rein, Weber, Memmert (2025), doi:10.6084/m9.figshare.28196177.v1. Each match has

    DFL_02_01_matchinformation_*.xml       teams (home / guest) and pitch size
    DFL_04_03_positions_raw_observed_*.xml 25 Hz positions for every player and the ball

The ball's frames carry two flags that are the labels this project learns from:

    BallStatus      1 = ball in play, 0 = dead ball
    BallPossession  1 = home team, 2 = guest team

mapped to the competition's alphabet as: dead -> D, otherwise home -> H, guest -> A.

The position file is ~400 MB of XML, so it is read as a stream and cached as a small .npz.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import A, D, H, X
from .half import Half, home_attacks_positive_x

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
BALL_TEAM = "BALL"
CACHE_VERSION = 2


def _find(folder, prefix, match):
    hits = glob.glob(os.path.join(folder, "{}*{}.xml".format(prefix, match)))
    if len(hits) != 1:
        raise IOError("expected one {}*{}.xml in {}, found {}".format(
            prefix, match, folder, len(hits)))
    return hits[0]


def _team_roles(info_path):
    """{team_id: 'home' | 'guest'} and the pitch size, from the match information file."""
    root = ET.parse(info_path).getroot()
    roles = {team.get("TeamId"): team.get("Role") for team in root.iter("Team")}
    if sorted(roles.values()) != ["guest", "home"]:
        raise ValueError("{}: team roles {}".format(info_path, roles))
    environment = next(root.iter("Environment"))
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    return roles, pitch


def _read_positions(path):
    """{period: {'ball': {...}, 'players': [(team_id, person_id, n, x, y)]}} from the stream."""
    out = {}
    current = None
    n, x, y, z, status, possession = [], [], [], [], [], []

    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = element.tag
        if event == "start":
            if tag == "FrameSet":
                current = (element.get("GameSection"), element.get("TeamId"),
                           element.get("PersonId"))
                n, x, y, z, status, possession = [], [], [], [], [], []
            continue

        if tag == "Frame":
            get = element.get
            n.append(int(get("N")))
            x.append(float(get("X")))
            y.append(float(get("Y")))
            if current[1] == BALL_TEAM:
                z.append(float(get("Z", "nan")))
                status.append(int(get("BallStatus")))
                possession.append(int(get("BallPossession")))
            element.clear()
        elif tag == "FrameSet":
            section, team_id, person_id = current
            period = SECTIONS.get(section)
            if period is not None:
                bucket = out.setdefault(period, {"ball": None, "players": []})
                frames = np.asarray(n, dtype=np.int64)
                xy = np.stack([np.asarray(x, dtype=np.float32),
                               np.asarray(y, dtype=np.float32)], axis=1)
                if team_id == BALL_TEAM:
                    bucket["ball"] = {
                        "n": frames, "xy": xy,
                        "z": np.asarray(z, dtype=np.float32),
                        "status": np.asarray(status, dtype=np.int8),
                        "possession": np.asarray(possession, dtype=np.int8),
                    }
                else:
                    bucket["players"].append((team_id, person_id, frames, xy))
            element.clear()
            current = None
    return out


def _build_half(match, period, bucket, roles, pitch):
    ball = bucket["ball"]
    if ball is None:
        raise ValueError("{} period {}: no ball frames".format(match, period))
    first, last = int(ball["n"][0]), int(ball["n"][-1])
    t = last - first + 1
    if len(ball["n"]) != t or not np.array_equal(ball["n"], np.arange(first, last + 1)):
        raise ValueError("{} period {}: ball frames are not contiguous".format(match, period))

    sides = {"home": [], "guest": []}
    for team_id, _person, frames, xy in bucket["players"]:
        role = roles.get(team_id)
        if role is None:
            continue  # officials, if a file ever carries them
        track = np.full((t, 2), np.nan, dtype=np.float32)
        inside = (frames >= first) & (frames <= last)
        track[frames[inside] - first] = xy[inside]
        sides[role].append(track)

    home_xy = np.stack(sides["home"], axis=1)
    away_xy = np.stack(sides["guest"], axis=1)
    ball_xy = ball["xy"].copy()

    labels = np.full(t, X, dtype=np.int8)
    alive = ball["status"] == 1
    labels[~alive] = D
    labels[alive & (ball["possession"] == 1)] = H
    labels[alive & (ball["possession"] == 2)] = A
    if (labels == X).any():
        raise ValueError("{} period {}: {} alive frames without a possessing team".format(
            match, period, int((labels == X).sum())))

    flipped = not home_attacks_positive_x(home_xy, away_xy)
    if flipped:
        home_xy, away_xy, ball_xy = -home_xy, -away_xy, -ball_xy

    return Half(match=match, period=period, pitch=pitch, ball_xy=ball_xy, home_xy=home_xy,
                away_xy=away_xy, labels=labels, ball_z=ball["z"].copy(), first_frame=first,
                meta={"source": "dfl", "flipped": flipped}).check()


def load_match(match, folder, cache_dir=None):
    """[Half, Half] for one DFL match. Cached after the first read."""
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "dfl_{}_v{}.npz".format(match, CACHE_VERSION))
        if os.path.isfile(cache_path):
            return _load_cache(cache_path)

    roles, pitch = _team_roles(_find(folder, "DFL_02_01_matchinformation", match))
    raw = _read_positions(_find(folder, "DFL_04_03_positions_raw_observed", match))
    halves = [_build_half(match, period, raw[period], roles, pitch) for period in sorted(raw)]
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        _save_cache(cache_path, halves)
    return halves


def _save_cache(path, halves):
    payload = {}
    for half in halves:
        key = "p{}_".format(half.period)
        payload[key + "ball_xy"] = half.ball_xy
        payload[key + "ball_z"] = half.ball_z
        payload[key + "home_xy"] = half.home_xy
        payload[key + "away_xy"] = half.away_xy
        payload[key + "labels"] = half.labels
        payload[key + "info"] = np.array([half.pitch[0], half.pitch[1], half.first_frame,
                                          1.0 if half.meta.get("flipped") else 0.0])
    payload["match"] = np.array(halves[0].match)
    np.savez_compressed(path, **payload)


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
                ball_z=data[key + "ball_z"], first_frame=int(info[2]),
                meta={"source": "dfl", "flipped": bool(info[3])}).check())
    return halves
