"""The frame classifier: gradient-boosted trees over the feature table.

Trees are used on purpose. Their prediction is a sum of constants selected by threshold tests, so
the same model file gives the same numbers on any machine, which is what lets someone else
regenerate the submitted file exactly. They also accept missing values as they are, and missing
values are the central fact of this data.
"""

import numpy as np

from . import A, H
from .features import build, is_extended

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 127,
    "num_threads": 8,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
ROUNDS = 400


def fit(table, target, names, seed=0, rounds=ROUNDS, params=None, weight=None):
    import lightgbm

    settings = dict(PARAMS, seed=seed)
    if params:
        settings.update(params)
    data = lightgbm.Dataset(table, label=target, feature_name=list(names), weight=weight,
                            free_raw_data=False)
    return lightgbm.train(settings, data, num_boost_round=rounds)


BOUNDARY_WIDTH = 6           # frames either side of a change that count as "at the change"
BOUNDARY_PARAMS = {"num_class": 4, "min_data_in_leaf": 150}


def fit_boundary(table, change, names, seed=0, rounds=ROUNDS, width=BOUNDARY_WIDTH, weight=None):
    """The boundary model: is a turnover, a stoppage or a restart happening right here?

    Same features, different question. The frame classifier is dominated by the long middles of
    possessions; this model is trained on the moments between them.
    """
    from .labels import change_classes

    return fit(table, change_classes(change, width), names, seed=seed, rounds=rounds,
               params=BOUNDARY_PARAMS, weight=weight)


def predict_table(booster, table):
    return booster.predict(table, num_threads=PARAMS["num_threads"])


def _tables(half, names):
    """The feature tables of a half and of its mirror image, with the columns a model expects.

    `names` is the model's own feature list. Features the builder has gained since the model was
    trained are left out; a feature the model needs and the builder lacks is an error. The
    extended columns are built only for a model that reads one of them.
    """
    extended = names is None or any(is_extended(name) for name in names)
    own_names, table = build(half, extended)
    _, mirrored = build(half.swapped(), extended)
    if names is None or list(names) == list(own_names):
        return table, mirrored
    missing = [name for name in names if name not in own_names]
    if missing:
        raise ValueError("the model needs features that are not built: " + ", ".join(missing[:5]))
    columns = [own_names.index(name) for name in names]
    return table[:, columns], mirrored[:, columns]


def predict_half(booster, half, names=None):
    """(T, 3) probabilities for home, away, dead ball, averaged over the two mirror views.

    The half is scored as it is and again with the teams exchanged and the pitch rotated; the
    second answer is mapped back and the two are averaged. A model that is exactly symmetric
    between the teams cannot favour one of them, and the average is also slightly less noisy.
    """
    table, mirrored_table = _tables(half, names)
    direct = predict_table(booster, table)
    mirrored = predict_table(booster, mirrored_table)
    mirrored = mirrored[:, [A, H, 2]]
    return (0.5 * (direct + mirrored)).astype(np.float64)


def predict_half_both(booster, boundary_booster, half, names=None):
    """(state probabilities (T, 3), boundary probabilities (T, 4)), each averaged over both views.

    The kinds of change (turnover, stoppage, restart) read the same from either side, so the
    mirrored boundary answer needs no remapping.
    """
    table, mirrored_table = _tables(half, names)
    state = 0.5 * (predict_table(booster, table)
                   + predict_table(booster, mirrored_table)[:, [A, H, 2]])
    if boundary_booster is None:
        return state.astype(np.float64), None
    boundary = 0.5 * (predict_table(boundary_booster, table)
                      + predict_table(boundary_booster, mirrored_table))
    return state.astype(np.float64), boundary.astype(np.float64)
