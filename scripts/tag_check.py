"""Label-free check of how the Smart Tagging is read: do the tags agree with tracking-only labels?

Needs a tracking-only run (python -m vantage run ... --no-tagging) and the data folder. Prints
shares only. If a kind of tag were read the wrong way round, its agreement would sit well below
one half.
"""
import sys

import numpy as np

from vantage import A, D, H, X, competition, epts, tagging
from vantage import _official_metrics as official

csv_path, data_dir = sys.argv[1], sys.argv[2]
labels = official.label_codes(official.read_label_csv(csv_path))
paths = competition.tactical_paths(data_dir)
home, away = epts.read_team_names(paths["meta"])
instances = tagging.load(paths["tagging"])
print("{} instances; team names in tracking: {!r}, {!r}".format(len(instances), home, away))
result = tagging.agreement(instances, labels, home, away)
for kind, value in result.items():
    print("  {:<18} agreement with tracking-only labels on live frames: {:.3f}".format(kind, value))

# How much of each kind of tagged time the tracking-only labels call dead ball.
for name, codes in (("in possession", tagging.IN_POSSESSION), ("out of possession", tagging.OUT_OF_POSSESSION)):
    frames = np.zeros(len(labels), dtype=bool)
    for item in instances:
        if item["code"] in codes:
            frames[max(0, item["start"]):max(0, item["end"])] = True
    graded = frames & (labels != X)
    print("  {:<18} tagged graded frames {:>6}; called dead ball by tracking {:.3f}".format(
        name, int(graded.sum()), float(np.mean(labels[graded] == D)) if graded.any() else float("nan")))

# Set pieces: how long are the tagged windows? (A fixed length means a clip, not a measurement.)
import collections
by_type = collections.defaultdict(list)
for item in instances:
    if item["code"] == tagging.SET_PIECE:
        by_type[item["tags"].get("Type", "?")].append((item["end"] - item["start"]) / 25.0)
for kind, seconds in sorted(by_type.items()):
    print("  set piece {:<24} n={:>2}  window seconds min/median/max {:.1f} {:.1f} {:.1f}".format(
        kind, len(seconds), min(seconds), float(np.median(seconds)), max(seconds)))
unknown = sum(1 for item in instances if tagging.team_code(item["tags"].get("Team"), home, away) is None)
print("  instances without a usable team: {}".format(unknown))
