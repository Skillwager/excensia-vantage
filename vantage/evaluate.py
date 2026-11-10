"""Score a prediction for one external match with the challenge's own metric.

The two halves are laid end to end with a block of ungraded frames between them, which is how the
competition file is laid out, so the official code sees the same structure it will see on the
real match: possessions never run across half-time.
"""

import numpy as np

from . import X
from . import _official_metrics as official

HALF_TIME_GAP = 2000


def stitch(parts):
    """One array from per-half arrays, separated by ungraded frames."""
    gap = np.full(HALF_TIME_GAP, X, dtype=np.int8)
    pieces = []
    for index, part in enumerate(parts):
        if index:
            pieces.append(gap)
        pieces.append(np.asarray(part, dtype=np.int8))
    return np.concatenate(pieces)


def score(truth_halves, predicted_halves):
    """The official result dict for one match given per-half truth and prediction arrays."""
    truth = stitch(truth_halves)
    predicted = stitch(predicted_halves)
    predicted[truth == X] = X
    return official.evaluate(truth, predicted)


def headline(result):
    return {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "n_true": result["n_gt_spells"],
        "n_pred": result["n_pred_spells"],
        "alive_dead": result["alive_dead_f1"],
        "team_acc": result["team_acc_both_alive"],
        "trans25": result["transitions"][25]["f1"],
    }


def quick(truth_halves, predicted_halves):
    """spellF1, macroF1 and the possession counts only, by the official functions.

    The same numbers `score` returns for those fields, without the transition and bootstrap
    work, for use inside a search over many decoding settings.
    """
    truth = stitch(truth_halves)
    predicted = stitch(predicted_halves)
    predicted[truth == X] = X
    runs = official.valid_runs(truth)
    filled = official.fill_abstentions(predicted, runs)
    spells = official.spell_scores(truth, filled, runs)
    macro = official.macro_f1(official.scored_confusion(official.confusion(truth, predicted)))
    return {"spellF1": spells["mean_f1"], "macroF1": macro, "n_true": spells["n_gt"],
            "n_pred": spells["n_pred"]}
