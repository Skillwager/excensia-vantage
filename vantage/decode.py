"""From per-frame probabilities to a label sequence made of whole possessions.

The ranking metric matches possessions as intervals, so a label sequence that is right on 97% of
frames but flickers scores badly. Decoding is therefore a search for the best *sequence*, not the
best label per frame:

1. A three-state Viterbi pass (home, away, dead ball). Every change of state pays a fixed cost,
   so the path only switches where the evidence for the new state outweighs it for long enough.
2. A minimum-length pass: a possession or stoppage shorter than its minimum is absorbed by what
   came before it.

The costs and minimum lengths are the only numbers here, and they are chosen by cross-validation
on other matches (scripts/cv.py), never on the competition match.
"""

from dataclasses import dataclass

import numpy as np

from . import A, D, FPS, H


@dataclass
class DecodeConfig:
    cost_team: float = 12.0         # home <-> away
    cost_to_dead: float = 10.0      # team -> dead ball
    cost_from_dead: float = 10.0    # dead ball -> team
    min_possession: float = 1.0     # seconds
    min_dead: float = 2.0           # seconds
    floor: float = 1e-4             # probabilities are clipped here before the log


def viterbi(log_prob, config):
    """Most likely state path under switching costs. log_prob is (T, 3) for H, A, D."""
    t = log_prob.shape[0]
    if t == 0:
        return np.zeros(0, dtype=np.int8)
    cost = np.zeros((3, 3), dtype=np.float64)        # cost[from, to]
    cost[H, A] = cost[A, H] = config.cost_team
    cost[H, D] = cost[A, D] = config.cost_to_dead
    cost[D, H] = cost[D, A] = config.cost_from_dead

    score = log_prob[0].astype(np.float64).copy()
    back = np.zeros((t, 3), dtype=np.int8)
    for index in range(1, t):
        candidate = score[:, None] - cost             # (from, to)
        best = np.argmax(candidate, axis=0)
        score = candidate[best, (0, 1, 2)] + log_prob[index]
        back[index] = best

    path = np.empty(t, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for index in range(t - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return path


def viterbi_many(log_prob, configs):
    """Viterbi paths for several cost settings in one pass. Returns (len(configs), T) int8.

    Used by the cross-validation grid, where the same probabilities are decoded under many
    settings; the result for each setting is identical to calling `viterbi` with it.
    """
    t = log_prob.shape[0]
    n = len(configs)
    cost = np.zeros((n, 3, 3), dtype=np.float64)
    for index, config in enumerate(configs):
        cost[index, H, A] = cost[index, A, H] = config.cost_team
        cost[index, H, D] = cost[index, A, D] = config.cost_to_dead
        cost[index, D, H] = cost[index, D, A] = config.cost_from_dead
    if t == 0:
        return np.zeros((n, 0), dtype=np.int8)

    emission = log_prob.astype(np.float64)
    score = np.repeat(emission[0][None, :], n, axis=0)
    back = np.zeros((t, n, 3), dtype=np.int8)
    for index in range(1, t):
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


def decode(prob, config):
    """Label path for one half. prob is (T, 3): P(home), P(away), P(dead ball)."""
    log_prob = np.log(np.clip(prob, config.floor, 1.0))
    return enforce_minimums(viterbi(log_prob, config), config)
