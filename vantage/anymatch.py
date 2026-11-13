"""The engine on any match's tracking, and its result written for the viewer.

The competition entry reads the organisers' files (competition.py). This is the same engine for
everything else: a provider's tracking of another match, or positions worked out from a video.
It takes plain arrays, in a tracking file: a NumPy .npz with

    fps         frames per second of the arrays. The engine works at 25; other rates are
                resampled.
    pitch       (length, width) of the pitch in metres
    home, away  (T, S, 2) positions in metres from the centre spot, x along the pitch, NaN where
                a slot is empty. A slot is a track, not a person: a team may be spread over any
                number of slots, and a slot may hold different players at different times.
    ball        (T, 2), NaN where the ball is not seen
    periods     optional (K, 2): first and last frame of each stretch of play (a half). The whole
                file is one stretch when this is absent.
    period_names  optional (K,) text: what the viewer calls each stretch (for clips joined into
                one file, the clip's time in the match and what happened in it)
    camera      optional (T, 3, 3): for each frame, the matrix taking a pitch position in metres
                (x, y, 1) to the pixel of the video that shows it; NaN where it is not known.
                With it the viewer draws the tracking on the video.
    camera_size (width, height) of that video in pixels, needed with `camera`
    title, home_name, away_name   optional text

    python -m vantage tracking match.npz --out match.run.json --page match.html

Nothing here knows about the competition, and nothing in the competition entry uses this module.
"""

import numpy as np

from . import FPS, X
from . import decode as decoding
from . import model, smoothing, viewer_export
from .features import build as build_features
from .half import Half, home_attacks_positive_x


def resample(values, fps, target=FPS):
    """Positions sampled `fps` times a second, at `target` times a second.

    values is (T, ..., 2). Between two samples that both have a position it is the straight line
    between them; next to a missing one it is the nearer sample, so a track is neither stretched
    across a gap nor cut short by half a step.
    """
    values = np.asarray(values, dtype=np.float32)
    if float(fps) == float(target):
        return values.copy()
    frames = len(values)
    count = int(np.floor((frames - 1) * target / fps)) + 1
    at = np.arange(count) * (fps / target)
    lo = np.minimum(np.floor(at).astype(np.int64), frames - 1)
    hi = np.minimum(lo + 1, frames - 1)
    u = (at - lo).astype(np.float32).reshape((-1,) + (1,) * (values.ndim - 1))
    a, b = values[lo], values[hi]
    out = a + (b - a) * u
    only_a = np.isnan(b) & ~np.isnan(a)
    only_b = np.isnan(a) & ~np.isnan(b)
    near_a = np.broadcast_to(u <= 0.5, out.shape)
    near_b = np.broadcast_to(u >= 0.5, out.shape)
    out = np.where(only_a & near_a, a, out)
    out = np.where(only_b & near_b, b, out)
    out = np.where((only_a & ~near_a) | (only_b & ~near_b), np.nan, out)
    return out.astype(np.float32)


def load(path):
    """A tracking file as a dict of arrays at the engine's 25 frames a second."""
    with np.load(path, allow_pickle=False) as data:
        fps = float(data["fps"])
        pitch = tuple(float(v) for v in data["pitch"])
        home = resample(data["home"], fps)
        away = resample(data["away"], fps)
        ball = resample(data["ball"], fps)
        if not (len(home) == len(away) == len(ball)):
            raise ValueError("home, away and ball cover different numbers of frames")
        frames = len(ball)
        scale = FPS / fps
        if "periods" in data.files:
            periods = [(int(round(first * scale)), min(frames - 1, int(round(last * scale))))
                       for first, last in np.asarray(data["periods"]).reshape(-1, 2)]
        else:
            periods = [(0, frames - 1)]
        camera = None
        if "camera" in data.files:
            if "camera_size" not in data.files:
                raise ValueError("a tracking file with a camera also needs camera_size")
            matrices = np.asarray(data["camera"], dtype=np.float64)
            # the camera of the nearest source frame: a matrix is not a thing to average
            nearest = np.minimum(np.round(np.arange(frames) * (fps / FPS)).astype(np.int64),
                                 len(matrices) - 1)
            camera = {"matrices": matrices[nearest],
                      "size": tuple(int(v) for v in data["camera_size"])}

        def text(key, fallback):
            return str(data[key]) if key in data.files else fallback

        period_names = None
        if "period_names" in data.files:
            period_names = [str(name) for name in np.asarray(data["period_names"]).reshape(-1)]
            if len(period_names) != len(periods):
                raise ValueError("period_names has {} names for {} periods".format(len(period_names), len(periods)))
        return {"pitch": pitch, "home": home, "away": away, "ball": ball, "periods": periods,
                "period_names": period_names, "camera": camera, "title": text("title", "Match"),
                "home_name": text("home_name", "Home"), "away_name": text("away_name", "Away")}


def halves_of(tracking, match="match"):
    """The stretches of play as the engine's halves. A stretch played the other way round is
    turned, so that the home team attacks towards +x in every one, and remembers that it was."""
    halves = []
    for index, (first, last) in enumerate(tracking["periods"]):
        rows = slice(first, last + 1)
        home_xy = tracking["home"][rows]
        away_xy = tracking["away"][rows]
        ball_xy = tracking["ball"][rows].copy()
        flipped = not home_attacks_positive_x(home_xy, away_xy)
        if flipped:
            home_xy, away_xy, ball_xy = -home_xy, -away_xy, -ball_xy
        halves.append(Half(match=match, period=index + 1, pitch=tracking["pitch"],
                           ball_xy=ball_xy, home_xy=home_xy, away_xy=away_xy, first_frame=first,
                           meta={"source": "tracking", "flipped": flipped}).check())
    return halves


def analyse(halves, frames, state_model, boundary_model, card):
    """(labels for every frame, state probabilities per half). Frames outside the stretches of
    play are X."""
    config = decoding.DecodeConfig(**card["decode"])
    labels = np.full(frames, X, dtype=np.int8)
    states = []
    for half in halves:
        half, _ = smoothing.for_model(half, card)
        state, boundary = model.predict_half_both(state_model, boundary_model, half,
                                                  card["features"])
        labels[half.first_frame:half.first_frame + half.n_frames] = decoding.decode(
            state, config, boundary)
        states.append(state)
    return labels, states


def run_file(tracking, halves, labels, states, state_model=None, card=None, every=5):
    """The viewer's run file for an analysed match: labels, confidence, evidence when the model
    is given, the tracking thinned to one sample in `every`, and the camera when there is one."""
    evidence = None
    if state_model is not None and card is not None:
        evidence = []
        for half in halves:
            half, _ = smoothing.for_model(half, card)
            built, table = build_features(half)
            names = card["features"]
            table = table[:, [built.index(name) for name in names]]
            codes = labels[half.first_frame:half.first_frame + half.n_frames]
            evidence.append(viewer_export.evidence_shares(
                state_model, table, names, np.minimum(codes, 2).astype(np.int64), every=every))
    camera = None
    if tracking["camera"] is not None:
        matrices = tracking["camera"]["matrices"]
        valid = np.isfinite(matrices).all(axis=(1, 2))
        camera = viewer_export.camera(matrices, valid, tracking["pitch"], tracking["camera"]["size"],
                                      0, len(labels), every=every, metres=True)
    names = tracking.get("period_names") or (["1st half", "2nd half"] if len(halves) == 2 else [
        "Part {}".format(index + 1) for index in range(len(halves))])
    return viewer_export.build(tracking["title"], tracking["home_name"], tracking["away_name"],
                               labels, halves, states, evidence, every=every, period_names=names,
                               camera=camera)
