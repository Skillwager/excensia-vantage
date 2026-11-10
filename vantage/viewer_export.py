"""Write a run file for the viewer (viewer/, schema `excensia.vantage.run/1`).

A run file is what the viewer opens next to a match video: the labels, how sure the model was of
each possession and on what kind of evidence, and the tracking it worked from, thinned out.

A run file of the competition match holds frame-level tracking. It is derived match data: keep
it on the machine that has the match, never commit it and never publish it. `.gitignore` excludes
`runs/`, which is where the command line writes it by default.
"""

import json

import numpy as np

from . import A, D, FPS, H, LABELS, X, __version__
from .labels import segments

SCHEMA = "excensia.vantage.run/1"

# Which kind of evidence a feature is, for the viewer's three chips. A feature that looks at
# another moment (a window, a time since or until, the other end of a gap) is timing; a
# feature of a team's layout is shape; what is left describes the ball and who is at it now.
_WINDOWS = ("_d25", "_d75", "_d250", "_m12", "_m37", "_m125")
_TIMING_PREFIXES = ("gap_", "last_", "next_", "out_", "run_", "still_", "tch_", "vis_")
_TIMING_WORDS = ("_past", "_next", "_since", "_until", "_last", "_run")


def evidence_group(name):
    if name.endswith(_WINDOWS) or name.startswith(_TIMING_PREFIXES):
        return "timing"
    if name.startswith("ctl_") and any(word in name for word in _TIMING_WORDS):
        return "timing"
    if name.startswith(("h_", "a_", "t_")):
        return "shape"
    return "ball"


def evidence_shares(booster, table, names, classes, every=5):
    """Per sampled row, how much of the model's call rests on ball, shape and timing features.

    Uses the model's own per-feature contributions to the score of the class it was asked about
    (`classes`, one code per row of `table`), in absolute value, summed by kind of evidence.
    Returns (rows, (n, 3)) with the shares in the order ball, shape, timing.
    """
    rows = np.arange(0, len(table), every)
    n_features = len(names)
    contributions = booster.predict(table[rows], pred_contrib=True)
    contributions = np.asarray(contributions).reshape(len(rows), -1, n_features + 1)
    chosen = np.abs(contributions[np.arange(len(rows)), classes[rows], :n_features])
    groups = np.array([("ball", "shape", "timing").index(evidence_group(name)) for name in names])
    shares = np.stack([chosen[:, groups == index].sum(axis=1) for index in range(3)], axis=1)
    total = shares.sum(axis=1, keepdims=True)
    return rows, shares / np.where(total > 0, total, 1.0)


def _run_length(labels):
    return [[LABELS[code], stop - start] for start, stop, code in segments(labels)]


def _point(xy):
    if not np.isfinite(xy[0]) or not np.isfinite(xy[1]):
        return None
    return [round(float(xy[0]), 2), round(float(xy[1]), 2)]


def build(title, home, away, labels, halves, states, evidence=None, every=5, period_names=None):
    """The run file as a dict.

    labels    (N,) label codes for every video frame
    halves    the Half objects the labels were decoded from; `first_frame` places each in the video
    states    per half, the (T, 3) probabilities of home, away and dead ball
    evidence  per half, None or (rows, (n, 3) shares) from `evidence_shares`
    every     keep one tracking sample in this many frames
    """
    n_frames = len(labels)
    period_names = period_names or ["1st half", "2nd half", "Extra time 1", "Extra time 2"]

    confidence = np.full(n_frames, np.nan)
    seen = np.zeros(n_frames, dtype=bool)
    covered = np.zeros(n_frames, dtype=bool)
    shares = np.full((n_frames, 3), np.nan)
    for index, (half, state) in enumerate(zip(halves, states)):
        span = slice(half.first_frame, half.first_frame + half.n_frames)
        codes = labels[span]
        inside = codes < 3
        picked = np.where(inside, state[np.arange(half.n_frames), np.minimum(codes, 2)], np.nan)
        confidence[span] = picked
        seen[span] = np.isfinite(half.ball_xy[:, 0])
        covered[span] = True
        if evidence is not None and evidence[index] is not None:
            rows, values = evidence[index]
            shares[half.first_frame + rows] = values

    spells, transitions = [], []
    previous = None
    for start, stop, code in segments(labels):
        if code != X:
            spell = {"start": start, "end": stop - 1, "label": LABELS[code]}
            if covered[start:stop].all():
                spell["confidence"] = round(float(np.nanmean(confidence[start:stop])), 3)
                spell["ball_seen"] = round(float(np.mean(seen[start:stop])), 3)
                known = shares[start:stop]
                known = known[np.isfinite(known[:, 0])]
                if len(known):
                    ball, shape, timing = (round(float(v), 3) for v in known.mean(axis=0))
                    spell["evidence"] = {"ball": ball, "shape": shape, "timing": timing}
            spells.append(spell)
            if previous is not None and previous != X:
                change = {"frame": start, "from": LABELS[previous], "to": LABELS[code]}
                if covered[start] and np.isfinite(confidence[start:stop]).any():
                    # how sure the model is of the new state over its first second
                    change["confidence"] = round(
                        float(np.nanmean(confidence[start:min(stop, start + FPS)])), 3)
                transitions.append(change)
        previous = code

    # Tracking: one sample in `every`, in the pitch's own orientation for the whole match. The
    # engine turns a half round so that home attacks +x; a viewer beside the video wants the
    # camera's left and right, so a turned half is turned back.
    count = (n_frames + every - 1) // every
    home_slots = max(half.home_xy.shape[1] for half in halves)
    away_slots = max(half.away_xy.shape[1] for half in halves)
    samples = [{"b": None, "h": [None] * home_slots, "a": [None] * away_slots}
               for _ in range(count)]
    for half in halves:
        sign = -1.0 if half.meta.get("flipped") else 1.0
        for row in range(half.n_frames):
            frame = half.first_frame + row
            if frame % every:
                continue
            sample = samples[frame // every]
            sample["b"] = _point(sign * half.ball_xy[row])
            sample["h"] = [_point(sign * xy) for xy in half.home_xy[row]] + [None] * (
                home_slots - half.home_xy.shape[1])
            sample["a"] = [_point(sign * xy) for xy in half.away_xy[row]] + [None] * (
                away_slots - half.away_xy.shape[1])

    graded = np.flatnonzero(labels != X)
    periods = []
    for index, half in enumerate(halves):
        inside = graded[(graded >= half.first_frame) & (graded < half.first_frame + half.n_frames)]
        if len(inside):
            periods.append({"name": period_names[index], "start": int(inside[0]),
                            "end": int(inside[-1])})
    return {
        "schema": SCHEMA,
        "generator": {"name": "excensia-vantage", "version": __version__},
        "match": {"title": title, "home": {"name": home}, "away": {"name": away}, "fps": FPS,
                  "frames": int(n_frames),
                  "pitch": {"length": float(halves[0].pitch[0]), "width": float(halves[0].pitch[1])},
                  "periods": periods},
        "labels": _run_length(labels),
        "spells": spells,
        "transitions": transitions,
        "tracking": {"every": int(every), "samples": samples},
        "video": {"offset_frames": 0},
    }


def write(path, run):
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        json.dump(run, handle, separators=(",", ":"))
    return path


def summary(run):
    """A few totals of a run file, safe to print: no positions and no frame numbers."""
    labels = run["labels"]
    frames = sum(length for _, length in labels)
    spells = run.get("spells", [])
    with_evidence = [s for s in spells if "evidence" in s]
    samples = run.get("tracking", {}).get("samples", [])
    return {
        "frames": frames,
        "label_runs": len(labels),
        "possessions": sum(1 for s in spells if s["label"] in ("H", "A")),
        "stoppages": sum(1 for s in spells if s["label"] == "D"),
        "spells_with_evidence": len(with_evidence),
        "mean_confidence": round(float(np.mean([s["confidence"] for s in spells
                                                 if "confidence" in s] or [0.0])), 3),
        "tracking_samples": len(samples),
        "samples_with_ball": sum(1 for s in samples if s["b"] is not None),
        "periods": len(run["match"]["periods"]),
    }
