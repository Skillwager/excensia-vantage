"""The external matches: where they are, how they are labelled, and how training tables are made.

Ten matches from two open sources are used for everything that is learned or tuned:

* seven from the open DFL dataset (vantage/dfl.py), and
* Metrica's three public sample games (vantage/metrica_sample.py).

The competition match is not part of this module.
"""

import os
import zlib

import numpy as np

from dataclasses import replace

from . import A, D, FPS, H, X, dfl, metrica_sample, skillcorner
from .degrade import DegradeConfig, degrade, sample_config
from .features import build
from .labels import CONTROL_SECONDS, change_distances, harmonise, run_lengths

# The repository root, where data/ and runs/ live. VANTAGE_ROOT overrides it, so that a frozen
# copy of the code can be run against the same data while the working copy is being edited.
ROOT = os.environ.get("VANTAGE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
METRICA_DIR = os.path.join(ROOT, "data", "external", "metrica-sample")
CACHE = os.path.join(ROOT, "data", "cache")

METRICA_MATCHES = tuple("metrica{}".format(game) for game in metrica_sample.GAMES)
ALL_MATCHES = tuple(dfl.MATCHES) + METRICA_MATCHES

# A third source, kept apart from the ten above: SkillCorner's open A-League matches, which are
# automatic tracking as it was recorded (vantage/skillcorner.py). Nothing is simulated on them.
SKILLCORNER_DIR = os.path.join(ROOT, "data", "external", "skillcorner")
SKILLCORNER_MATCHES = skillcorner.MATCHES

# The simulated recording conditions a held-out match is scored under. None is the corrected
# tracking as published; "random" draws a setting from the same ranges the training passes use.
CONDITIONS = {
    "clean": None,
    "typical": DegradeConfig(),
    "easy": DegradeConfig(live_loss_rate=0.2, long_miss=0.0, player_missing=0.03, blackout=0.0,
                          extra_share=0.03, false_rate=0.0),
    "harsh": DegradeConfig(dead_hidden=0.7, lose_shift=-1.0, lose_spread=1.5, find_lag=1.5,
                           long_miss=0.12, live_loss_rate=1.6, live_loss_median=10.0,
                           until_restart=0.5, false_rate=0.15, player_missing=0.1,
                           player_drift=0.35, blackout=0.5, extra_share=0.2),
    # stoppages mostly tracked right through, and the ball lost more often in live play
    "watched": DegradeConfig(dead_hidden=0.3, dead_watched=0.65, dead_watched_in=0.85,
                             hold_at_line=0.7, early_find=0.4, live_loss_rate=1.4,
                             live_loss_median=8.0),
    "random1": "random",
    "random2": "random",
}


def condition_half(half, match, name):
    """One labelled half as it would be recorded under the named condition. Deterministic."""
    rng = np.random.default_rng(seed_for("eval", match, half.period, name))
    setting = CONDITIONS[name]
    noisy = observed(half, rng, sample_config(rng) if setting == "random" else setting)
    return with_tags(noisy, np.random.default_rng(seed_for("tags", "eval", match, half.period, name)))


def seed_for(*parts):
    """A stable 32-bit seed from names, so every table can be rebuilt bit for bit."""
    return zlib.crc32("|".join(str(part) for part in parts).encode("utf-8"))


def source_of(match):
    if match.startswith("metrica"):
        return "metrica"
    return "skillcorner" if match.startswith("sc") else "dfl"


def labelled_halves(match):
    """The match's halves with labels brought to the challenge's definition of possession."""
    source = source_of(match)
    if source == "metrica":
        halves = metrica_sample.load_game(int(match[-1]), METRICA_DIR)
    elif source == "skillcorner":
        halves = skillcorner.load_match(match, SKILLCORNER_DIR, CACHE)
    else:
        halves = dfl.load_match(match, DFL_DIR, CACHE)
    for half in halves:
        half.labels = harmonise(half.labels, CONTROL_SECONDS[source])
    return halves


def observed(half, rng, config):
    """What automatic tracking might have recorded of this half."""
    if half.meta.get("source") == "skillcorner":
        return half         # it is automatic tracking: what was recorded is what there is
    if half.meta.get("source") == "metrica-sample":
        half = metrica_sample.with_dead_ball_track(half, rng)
        if config is not None:
            half = metrica_sample.with_loose_ball(half, rng)
    return degrade(half, rng, config)


def tagged_restarts(half, rng):
    """Restart tags of the kind Smart Tagging gives for set pieces, drawn from a half's labels.

    The provided recording comes with about sixty set-piece tags, each of which places a restart
    to within a second or two and, for corners, goal kicks and free kicks, names the team. No
    external match has such tags, so they are imitated: some of the half's real restarts are
    tagged, with an error in time, and some of those name the team. How many, how exact and how
    often with a team are drawn afresh each time, and a few tags point at nothing, because the
    real tags are automatic and none of this can be measured on the recording.
    """
    labels = half.labels
    t = len(labels)
    coverage = rng.uniform(0.25, 0.75)
    with_team = rng.uniform(0.3, 0.7)
    spread = rng.uniform(0.3, 1.5)
    shift = rng.uniform(-1.0, 1.0)
    false_share = rng.uniform(0.0, 0.1)
    out = [(int(np.clip(round((shift + rng.normal(0.0, spread)) * FPS), 0, t - 1)), None)]  # kick-off
    dead = np.concatenate(([0], (labels == D).astype(np.int8), [0]))
    edges = np.diff(dead)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop >= t or stop - start < 2 * FPS or rng.random() >= coverage:
            continue
        row = stop + int(round((shift + rng.normal(0.0, spread)) * FPS))
        team = None
        if labels[stop] in (H, A) and rng.random() < with_team:
            team = int(labels[stop])
            if rng.random() < 0.03:
                team = A if team == H else H
        out.append((int(np.clip(row, 0, t - 1)), team))
    for _ in range(int(rng.poisson(false_share * len(out)))):
        out.append((int(rng.integers(0, t)), (H, A, None)[int(rng.integers(0, 3))]))
    return sorted(out, key=lambda pair: pair[0])


def with_tags(half, rng):
    """The half with imitated restart tags attached (see `tagged_restarts`)."""
    return replace(half, meta=dict(half.meta, restarts=tagged_restarts(half, rng)))


def train_table_path(tag, match, period, variant, view):
    return os.path.join(CACHE, "train_" + tag,
                        "{}_p{}_k{}_{}.npz".format(match, period, variant, view))


def prepare_match(match, tag, variants, stride):
    """Build and cache the training tables of one match. Existing tables are kept."""
    os.makedirs(os.path.join(CACHE, "train_" + tag), exist_ok=True)
    for half in labelled_halves(match):
        for variant in range(variants):
            rng = np.random.default_rng(seed_for("train", match, half.period, variant))
            noisy = observed(half, rng, sample_config(rng))
            # the tags have their own stream, so adding them moved nothing else
            noisy = with_tags(noisy, np.random.default_rng(
                seed_for("tags", "train", match, half.period, variant)))
            for view, version in (("own", noisy), ("swap", noisy.swapped())):
                path = train_table_path(tag, match, half.period, variant, view)
                if os.path.isfile(path):
                    continue
                names, table = build(version)
                phase = int(rng.integers(0, stride))
                rows = np.arange(phase, version.n_frames, stride)
                rows = rows[version.labels[rows] != X]
                if version.meta.get("source") == "skillcorner":
                    # The broadcast cuts away during stoppages, and its tracking with it: half
                    # of this source's dead-ball frames have no players at all. The recording
                    # the method is for never does that, so those frames are not learned from.
                    seen = (np.isfinite(version.home_xy[..., 0]).sum(axis=1)
                            + np.isfinite(version.away_xy[..., 0]).sum(axis=1))
                    rows = rows[seen[rows] >= 14]
                np.savez(path, table=table[rows], target=version.labels[rows],
                         change=change_distances(version.labels)[:, rows],
                         run=run_lengths(version.labels)[rows], names=np.array(names))


def load_training(tag, matches, variants):
    """The cached training tables of the given matches, as one set of arrays.

    names    the feature names
    table    (N, F) float32 features
    target   (N,) home, away or dead ball
    change   (3, N) frames to the nearest turnover, stoppage start and restart: the boundary
             model's targets
    run      (N,) the length in frames of the possession or stoppage the row belongs to
    metrica  (N,) True for rows from Metrica's sample games
    """
    parts = {"table": [], "target": [], "change": [], "run": [], "metrica": []}
    names = None
    for match in matches:
        for period in (1, 2):
            for variant in range(variants):
                for view in ("own", "swap"):
                    with np.load(train_table_path(tag, match, period, variant, view)) as data:
                        for key in ("table", "target", "change", "run"):
                            parts[key].append(data[key])
                        parts["metrica"].append(
                            np.full(len(data["target"]), source_of(match) == "metrica"))
                        names = [str(name) for name in data["names"]]
    out = {key: np.concatenate(value, axis=1 if key == "change" else 0)
           for key, value in parts.items()}
    out["names"] = names
    return out


def row_weights(training, metrica_weight=1.0, short_weight=1.0, short_seconds=8.0):
    """A weight per training row, or None when every weight would be one.

    `metrica_weight` multiplies the rows from Metrica's sample games. `short_weight` raises the
    rows of short possessions: the ranking metric counts every possession once, however long it
    is, while an unweighted fit counts every frame once, so a forty-second possession outweighs
    a one-second one forty to one. A possession of `short_seconds` or longer keeps weight one; a
    shorter one gets short_seconds divided by its length, up to `short_weight`. Dead-ball rows
    keep weight one.
    """
    if metrica_weight == 1.0 and short_weight == 1.0:
        return None
    weight = np.ones(len(training["target"]), dtype=np.float32)
    if short_weight != 1.0:
        seconds = np.maximum(training["run"], 1) / float(FPS)
        boost = np.clip(short_seconds / seconds, 1.0, short_weight)
        weight = np.where(training["target"] == D, 1.0, boost).astype(np.float32)
    if metrica_weight != 1.0:
        weight = weight * np.where(training["metrica"], metrica_weight, 1.0).astype(np.float32)
    return weight
