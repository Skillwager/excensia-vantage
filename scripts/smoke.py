import time, sys
import numpy as np
from vantage import H, A, D
from vantage.dfl import load_match
from vantage.degrade import degrade, sample_config, DegradeConfig
from vantage.features import build
from vantage.labels import harmonise
from vantage import decode as dec

half = load_match("J03WOH", "data/external/dfl", "data/cache")[0]
half.labels = harmonise(half.labels, 3.0)
t0 = time.time(); names, tab = build(half); print("clean features", tab.shape, "%.1fs" % (time.time() - t0))
print("NaN fraction overall %.3f; all-NaN columns: %s" % (np.isnan(tab).mean(), [n for n, c in zip(names, tab.T) if np.isnan(c).all()]))
rng = np.random.default_rng(1)
for label, cfg in (("typical", DegradeConfig()), ("random", sample_config(rng))):
    t0 = time.time(); noisy = degrade(half, rng, cfg); dt = time.time() - t0
    seen = np.isfinite(noisy.ball_xy[:, 0])
    dead = half.labels == D
    gaps = np.diff(np.concatenate(([0], (~seen).astype(int), [0])))
    lens = (np.flatnonzero(gaps == -1) - np.flatnonzero(gaps == 1)) / 25.0
    hp = np.isfinite(noisy.home_xy[..., 0]).sum(1).mean() + np.isfinite(noisy.away_xy[..., 0]).sum(1).mean()
    print("%s: degrade %.1fs | ball hidden %.3f (alive %.3f, dead %.3f) | gaps n=%d mean %.2fs med %.2fs p90 %.2fs max %.1fs | players seen %.2f of 22 | slots %d+%d" % (
        label, dt, 1 - seen.mean(), 1 - seen[~dead].mean(), 1 - seen[dead].mean(), len(lens), lens.mean(), np.median(lens), np.percentile(lens, 90), lens.max(), hp, noisy.home_xy.shape[1], noisy.away_xy.shape[1]))
    t0 = time.time(); n2, tab2 = build(noisy); print("   features %.1fs NaN %.3f" % (time.time() - t0, np.isnan(tab2).mean()))
    assert n2 == names
# decoder equivalence
rng = np.random.default_rng(0)
p = rng.dirichlet([1, 1, 1], size=3000)
cfgs = [dec.DecodeConfig(cost_team=a, cost_to_dead=b, cost_from_dead=b) for a in (2.0, 9.0) for b in (3.0, 7.0)]
many = dec.viterbi_many(np.log(p), cfgs)
for i, c in enumerate(cfgs):
    assert np.array_equal(many[i], dec.viterbi(np.log(p), c)), i
print("viterbi_many == viterbi for", len(cfgs), "settings; features:", len(names))
sw = half.swapped(); n3, tab3 = build(sw)
print("swap check: h_cx(own) vs a_cx(swapped) max abs diff %.4f" % np.nanmax(np.abs(tab[:, names.index("h_cx")] - tab3[:, names.index("a_cx")])))
