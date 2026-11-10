"""The external matches: where they are, how they are labelled, and how training tables are made.

Ten matches from two open sources are used for everything that is learned or tuned:

* seven from the open DFL dataset (vantage/dfl.py), and
* Metrica's three public sample games (vantage/metrica_sample.py).

The competition match is not part of this module.
"""

import os
import zlib

import numpy as np

from . import X, dfl, metrica_sample
from .degrade import degrade, sample_config
from .features import build
from .labels import CONTROL_SECONDS, harmonise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
METRICA_DIR = os.path.join(ROOT, "data", "external", "metrica-sample")
CACHE = os.path.join(ROOT, "data", "cache")

METRICA_MATCHES = tuple("metrica{}".format(game) for game in metrica_sample.GAMES)
ALL_MATCHES = tuple(dfl.MATCHES) + METRICA_MATCHES


def seed_for(*parts):
    """A stable 32-bit seed from names, so every table can be rebuilt bit for bit."""
    return zlib.crc32("|".join(str(part) for part in parts).encode("utf-8"))


def source_of(match):
    return "metrica" if match.startswith("metrica") else "dfl"


def labelled_halves(match):
    """The match's halves with labels brought to the challenge's definition of possession."""
    source = source_of(match)
    if source == "metrica":
        halves = metrica_sample.load_game(int(match[-1]), METRICA_DIR)
    else:
        halves = dfl.load_match(match, DFL_DIR, CACHE)
    for half in halves:
        half.labels = harmonise(half.labels, CONTROL_SECONDS[source])
    return halves


def observed(half, rng, config):
    """What automatic tracking might have recorded of this half."""
    if half.meta.get("source") == "metrica-sample":
        half = metrica_sample.with_dead_ball_track(half, rng)
    return degrade(half, rng, config)


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
            for view, version in (("own", noisy), ("swap", noisy.swapped())):
                path = train_table_path(tag, match, half.period, variant, view)
                if os.path.isfile(path):
                    continue
                names, table = build(version)
                phase = int(rng.integers(0, stride))
                rows = np.arange(phase, version.n_frames, stride)
                rows = rows[version.labels[rows] != X]
                np.savez(path, table=table[rows], target=version.labels[rows],
                         names=np.array(names))


def load_training(tag, matches, variants):
    """(feature names, table, target) concatenated over the given matches."""
    tables, targets, names = [], [], None
    for match in matches:
        for period in (1, 2):
            for variant in range(variants):
                for view in ("own", "swap"):
                    with np.load(train_table_path(tag, match, period, variant, view)) as data:
                        tables.append(data["table"])
                        targets.append(data["target"])
                        names = [str(name) for name in data["names"]]
    return names, np.concatenate(tables), np.concatenate(targets)
