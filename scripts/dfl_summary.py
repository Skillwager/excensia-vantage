"""Print a sanity summary of one open DFL match (external data, safe to inspect)."""
import sys, time
import numpy as np
from vantage import H, A, D
from vantage.dfl import load_match

match, folder, cache = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
halves = load_match(match, folder, cache)
print("loaded {} in {:.1f}s".format(match, time.time() - t0))
for half in halves:
    lab = half.labels
    runs = np.flatnonzero(np.diff(lab) != 0) + 1
    bounds = np.concatenate(([0], runs, [len(lab)]))
    segs = [(bounds[i], bounds[i + 1], lab[bounds[i]]) for i in range(len(bounds) - 1)]
    spells = [(a, b) for a, b, k in segs if k in (H, A)]
    dead = [(a, b) for a, b, k in segs if k == D]
    dur = np.array([b - a for a, b in spells]) / 25.0
    ddur = np.array([b - a for a, b in dead]) / 25.0
    L, W = half.pitch
    out = (np.abs(half.ball_xy[:, 0]) > L / 2) | (np.abs(half.ball_xy[:, 1]) > W / 2)
    print("period {} frames {} first_frame {} flipped {} pitch {}".format(
        half.period, half.n_frames, half.first_frame, half.meta["flipped"], half.pitch))
    print("  slots home {} away {} | mean present home {:.2f} away {:.2f}".format(
        half.home_xy.shape[1], half.away_xy.shape[1],
        np.isfinite(half.home_xy[..., 0]).sum(1).mean(), np.isfinite(half.away_xy[..., 0]).sum(1).mean()))
    print("  share H {:.3f} A {:.3f} D {:.3f}".format(*(np.mean(lab == k) for k in (H, A, D))))
    print("  possessions {} mean {:.1f}s median {:.1f}s p10 {:.1f}s p90 {:.1f}s | <2s: {:.2f}".format(
        len(spells), dur.mean(), np.median(dur), np.percentile(dur, 10), np.percentile(dur, 90), np.mean(dur < 2)))
    print("  dead stretches {} mean {:.1f}s median {:.1f}s".format(len(dead), ddur.mean(), np.median(ddur)))
    print("  ball NaN {:.4f} | out of pitch: all {:.3f}, alive {:.3f}, dead {:.3f}".format(
        np.isnan(half.ball_xy[:, 0]).mean(), out.mean(), out[lab != D].mean(), out[lab == D].mean()))
    z = half.ball_z
    print("  ball z: mean {:.2f} p50 {:.2f} p90 {:.2f} p99 {:.2f} | >1.5m when alive {:.3f}".format(
        np.nanmean(z), np.nanmedian(z), np.nanpercentile(z, 90), np.nanpercentile(z, 99), np.mean(z[lab != D] > 1.5)))
    print("  home mean x {:.1f}, away mean x {:.1f}".format(np.nanmean(half.home_xy[..., 0]), np.nanmean(half.away_xy[..., 0])))
