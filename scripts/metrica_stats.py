"""Possession statistics of Metrica's public sample games under the event-derived labels."""
import sys, time
import numpy as np
from vantage import H, A, D, X
from vantage.metrica_sample import GAMES, load_game
from vantage.labels import harmonise, spell_stats, segments

root = sys.argv[1]
tot = {}
for game in GAMES:
    t0 = time.time()
    halves = load_game(game, root)
    frames = sum(h.n_frames for h in halves)
    for h in halves:
        lab = h.labels
        seen = np.isfinite(h.ball_xy[:, 0])
        segs = segments(lab)
        dead = [(b - a) / 25.0 for a, b, k in segs if k == D]
        print("game {} p{} frames {} flipped {} slots {}+{} | share H {:.3f} A {:.3f} D {:.3f} X {:.4f} | dead stretches {} mean {:.1f}s | players present {:.2f}".format(
            game, h.period, h.n_frames, h.meta["flipped"], h.home_xy.shape[1], h.away_xy.shape[1],
            *(np.mean(lab == k) for k in (H, A, D, X)), len(dead), np.mean(dead),
            np.isfinite(h.home_xy[..., 0]).sum(1).mean() + np.isfinite(h.away_xy[..., 0]).sum(1).mean()))
    row = []
    for th in (0.0, 1.0, 1.5, 2.0, 3.0):
        n = 0; secs = 0.0; short = 0
        for h in halves:
            lab = harmonise(h.labels, th) if th > 0 else h.labels
            c, mean, med, under2 = spell_stats(lab)
            n += c; secs += c * mean; short += under2 * c
        tot.setdefault(th, [0, 0.0, 0.0]); tot[th][0] += n; tot[th][1] += secs
        row.append("th{:.1f}: {}/{:.1f}s(<2s {:.2f})".format(th, n, secs / n, short / n))
    live_min = sum(int((h.labels != D).sum()) for h in halves) / 1500.0
    tot.setdefault("live", 0.0); tot["live"] += live_min
    print("   game {} ({:.0f}s load, {:.1f} live min): ".format(game, time.time() - t0, live_min) + "  ".join(row))
print("ALL per live minute: " + "  ".join("th{}: {:.2f}/min mean {:.1f}s".format(th, tot[th][0] / tot["live"], tot[th][1] / tot[th][0]) for th in (0.0, 1.0, 1.5, 2.0, 3.0)))
