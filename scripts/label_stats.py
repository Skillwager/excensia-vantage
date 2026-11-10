"""How the control threshold changes possession statistics on the open DFL matches."""
import sys, time
import numpy as np
from vantage import D
from vantage.dfl import MATCHES, load_match
from vantage.labels import harmonise, spell_stats

folder, cache = sys.argv[1], sys.argv[2]
thresholds = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
totals = {th: [0, 0.0] for th in thresholds}
minutes = 0.0
dead_share = []
for match in MATCHES:
    t0 = time.time()
    halves = load_match(match, folder, cache)
    frames = sum(h.n_frames for h in halves)
    minutes += frames / 25.0 / 60.0
    dead = sum(int((h.labels == D).sum()) for h in halves) / frames
    dead_share.append(dead)
    row = []
    for th in thresholds:
        n = 0; secs = 0.0
        for h in halves:
            lab = harmonise(h.labels, th) if th > 0 else h.labels
            c, mean, _, _ = spell_stats(lab)
            n += c; secs += c * mean
        totals[th][0] += n; totals[th][1] += secs
        row.append("{:>4d}/{:>4.1f}s".format(n, secs / n))
    print("{} {:>6.1f}min dead {:.3f} | ".format(match, frames / 1500.0, dead) + "  ".join(row) + "   ({:.0f}s load)".format(time.time() - t0))
print("thresholds (s):      " + "  ".join("{:>10}".format(th) for th in thresholds))
print("all: possessions per match / mean s: " + "  ".join("{:>5.0f}/{:>4.1f}".format(totals[th][0] / len(MATCHES), totals[th][1] / totals[th][0]) for th in thresholds))
print("possessions per 5 live+dead minutes: " + "  ".join("{:>10.1f}".format(totals[th][0] / minutes * 5) for th in thresholds))
print("mean dead share {:.3f}".format(float(np.mean(dead_share))))
