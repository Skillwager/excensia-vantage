"""From per-frame probabilities to a label sequence made of whole possessions.

The ranking metric matches possessions as intervals, so a label sequence that is right on 97% of
frames but flickers scores badly. Decoding is therefore a search for the best *sequence*, not the
best label per frame:

1. A three-state Viterbi pass (home, away, dead ball). Every change of state pays a cost, so the
   path only switches where the evidence for the new state outweighs it for long enough. The
   cost is not the same everywhere: a second model says, frame by frame, how likely it is that
   a turnover, a stoppage or a restart is happening right there, and a change is cheap where
   that model sees one and expensive where it does not. That is what lets a short possession
   survive without letting flicker through.
2. A minimum-length pass: a possession or stoppage shorter than its minimum is absorbed by what
   came before it.

The costs, the weight of the boundary model and the minimum lengths are the only numbers here.
They are chosen by cross-validation on other matches (scripts/cv.py), never on the competition
match.
"""

from dataclasses import dataclass

import numpy as np

from . import A, D, FPS, H


@dataclass
class DecodeConfig:
    cost_team: float = 12.0         # home <-> away
    cost_to_dead: float = 10.0      # team -> dead ball
    cost_from_dead: float = 10.0    # dead ball -> team
    boundary_weight: float = 0.0    # how far the boundary model moves those costs (0 = not used)
    min_possession: float = 1.0     # seconds
    min_dead: float = 2.0           # seconds
    floor: float = 1e-4             # probabilities are clipped here before the log
    cost_floor: float = 0.25        # a change is never cheaper than this


# Which kind of change each transition is: 0 turnover, 1 stoppage, 2 restart.
_KIND = np.zeros((3, 3), dtype=np.int64)
_KIND[H, D] = _KIND[A, D] = 1
_KIND[D, H] = _KIND[D, A] = 2
_OFF_DIAGONAL = 1.0 - np.eye(3)


def boundary_log_odds(boundary, floor=1e-3):
    """(T, 3, 3): log-odds that the change from state i to state j happens at each frame.

    `boundary` is the boundary model's output, (T, 4) probabilities of no change, turnover,
    stoppage and restart. None gives None.
    """
    if boundary is None:
        return None
    p = np.clip(np.asarray(boundary, dtype=np.float64)[:, 1:4], floor, 1.0 - floor)
    odds = np.log(p / (1.0 - p))
    return odds[:, _KIND] * _OFF_DIAGONAL


def _base_cost(config):
    cost = np.zeros((3, 3), dtype=np.float64)        # cost[from, to]
    cost[H, A] = cost[A, H] = config.cost_team
    cost[H, D] = cost[A, D] = config.cost_to_dead
    cost[D, H] = cost[D, A] = config.cost_from_dead
    return cost


def viterbi(log_prob, config, odds=None):
    """Most likely state path under switching costs. log_prob is (T, 3) for H, A, D."""
    return viterbi_many(log_prob, [config], odds)[0]


def viterbi_many(log_prob, configs, odds=None):
    """Viterbi paths for several settings in one pass. Returns (len(configs), T) int8.

    `odds` is the output of boundary_log_odds, or None to use fixed costs throughout.
    """
    t = log_prob.shape[0]
    n = len(configs)
    if t == 0:
        return np.zeros((n, 0), dtype=np.int8)
    base = np.stack([_base_cost(config) for config in configs])
    weight = np.array([config.boundary_weight for config in configs], dtype=np.float64)
    floor = np.array([config.cost_floor for config in configs], dtype=np.float64)
    floor = floor[:, None, None] * _OFF_DIAGONAL
    varying = odds is not None and bool(np.any(weight != 0.0))

    emission = log_prob.astype(np.float64)
    score = np.repeat(emission[0][None, :], n, axis=0)
    back = np.zeros((t, n, 3), dtype=np.int8)
    cost = base
    for index in range(1, t):
        if varying:
            cost = np.maximum(base - weight[:, None, None] * odds[index], floor)
        candidate = score[:, :, None] - cost
        best = np.argmax(candidate, axis=1)
        score = np.take_along_axis(candidate, best[:, None, :], axis=1)[:, 0, :] + emission[index]
        back[index] = best

    rows = np.arange(n)
    paths = np.empty((n, t), dtype=np.int8)
    state = np.argmax(score, axis=1)
    paths[:, -1] = state
    for index in range(t - 1, 0, -1):
        state = back[index, rows, state]
        paths[:, index - 1] = state
    return paths


def enforce_minimums(path, config, fps=FPS):
    """Absorb possessions and stoppages that are shorter than their minimum length."""
    out = path.copy()
    min_possession = int(round(config.min_possession * fps))
    min_dead = int(round(config.min_dead * fps))
    for _ in range(6):                              # merging can expose a new short run
        breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(out)]))
        changed = False
        for i in range(1, len(bounds) - 1):         # the first run has nothing before it
            start, stop = int(bounds[i]), int(bounds[i + 1])
            code = int(out[start])
            minimum = min_dead if code == D else min_possession
            if stop - start < minimum:
                out[start:stop] = out[start - 1]
                changed = True
        if not changed:
            break
    return out


def decode(prob, config, boundary=None, extra_log_prob=None):
    """Label path for one half.

    prob is (T, 3): P(home), P(away), P(dead ball). `boundary` is the boundary model's (T, 4)
    output, if used. `extra_log_prob` is added to the log-probabilities first (the Smart Tagging
    votes).
    """
    log_prob = np.log(np.clip(prob, config.floor, 1.0))
    if extra_log_prob is not None:
        log_prob = log_prob + extra_log_prob
    path = viterbi(log_prob, config, boundary_log_odds(boundary))
    return enforce_minimums(path, config)
