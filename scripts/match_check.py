"""Check the competition loader against the figures the organisers publish in DATA.md.

Prints totals only. Every number it prints is either structural (how many slots) or has a
published counterpart to compare with:

    ball present on 113,488 of 199,319 frames, and on 94,214 of 158,857 graded frames
    mean of 20.5 tracked objects over graded frames
    46 track slots
"""
import sys
import time

import numpy as np

from vantage import competition, epts

data_dir = sys.argv[1]
paths = competition.tactical_paths(data_dir)
started = time.time()
data = epts.read(paths["meta"], paths["raw"])
print("read in {:.0f}s: {} records, frame ids {}..{} contiguous {}".format(
    time.time() - started, len(data.frame), int(data.frame[0]), int(data.frame[-1]),
    bool(np.array_equal(data.frame, np.arange(1, len(data.frame) + 1)))))
print("teams: {}  slots: home {} away {} (total {})  speed unit: {!r}  speed present: {}".format(
    data.team_names, int((data.team == 0).sum()), int((data.team == 1).sum()), len(data.team),
    data.speed_unit, bool(np.isfinite(data.speed).any())))

seen = np.isfinite(data.ball[:, 0])
graded = np.zeros(len(seen), dtype=bool)
for first, last in competition.GRADED.values():
    graded[first:last + 1] = True
present = np.isfinite(data.xy[..., 0]).sum(axis=1)
print("ball present: {} of {} frames (published 113488 of 199319)".format(int(seen.sum()), len(seen)))
print("ball present on graded frames: {} of {} (published 94214 of 158857)".format(
    int(seen[graded].sum()), int(graded.sum())))
print("mean tracked objects on graded frames: {:.2f} (published 20.5)".format(present[graded].mean()))

halves = competition.load_halves(data_dir)
for half in halves:
    print("half {}: rows {} first video frame {} flipped {} slots {}+{}".format(
        half.period, half.n_frames, half.first_frame, half.meta["flipped"],
        half.home_xy.shape[1], half.away_xy.shape[1]))
