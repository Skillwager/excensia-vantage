"""A run file of the organisers' public five minutes for the viewer: their labels, not ours.

The organisers' clarification of the rules (3 November 2026) allows "inspecting the provided
data and the video to better understand what counts as possession", and does not allow output
to be checked against the video. This script serves the first and keeps clear of the second. It
writes the public ground truth of the first five minutes of play, with the provided tracking, as
a run file the viewer can play beside the tactical video. Nothing in the file comes from the
model: there is no confidence and no evidence in it, and it is named for what it is. The run
files of the model's own output cannot be used this way, because the viewer opens no video for
them (vantage.competition.VIDEO_WITHHELD).

    python scripts/public_truth_run.py --data-dir /path/to/data \\
        --official external/more-than-a-hack-2027 \\
        --out runs/match/public/public-five-minutes.run.json \\
        --page runs/match/public/public-five-minutes.html
    node viewer/scripts/show.mjs runs/match/public/public-five-minutes.html \\
        --video=/path/to/BAR-ATH_tactical.mp4

The files hold match tracking and stay on the machine that has the match. Totals only are
printed.
"""

import argparse
import os

import numpy as np

from vantage import A, FPS, H, X, competition, epts, viewer_export
from vantage import _official_metrics as official

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="folder containing tactical/")
    parser.add_argument("--official", required=True, help="clone of the official repository")
    parser.add_argument("--out", required=True, help="where to write the run file (.json)")
    parser.add_argument("--page", default=None,
                        help="also write the viewer with this run built in, as one .html file")
    parser.add_argument("--viewer", default=os.path.join(ROOT, "viewer", "dist", "index.html"))
    parser.add_argument("--every", type=int, default=5,
                        help="keep one tracking sample in this many frames")
    parser.add_argument("--margin", type=float, default=5.0,
                        help="seconds of the match kept before and after the labelled stretch")
    args = parser.parse_args()

    truth_path = os.path.join(args.official, "evaluation", "gt",
                              "BAR-ATH_possession_gt_public.csv")
    truth = official.label_codes(official.read_label_csv(truth_path))
    if len(truth) != competition.N_FRAMES:
        raise SystemExit("the public ground truth has {} rows, not {}".format(
            len(truth), competition.N_FRAMES))
    labelled = np.flatnonzero(truth != X)
    margin = int(round(args.margin * FPS))
    start = max(0, int(labelled[0]) - margin)
    stop = min(len(truth), int(labelled[-1]) + 1 + margin)

    halves = competition.load_halves(args.data_dir)
    home, away = epts.read_team_names(competition.tactical_paths(args.data_dir)["meta"])
    run = viewer_export.window(
        "{} v {}, first five minutes: the organisers' labels".format(home, away), home, away,
        truth, halves, start, stop, every=args.every,
        generator={"name": "organisers-public-ground-truth", "version": "more-than-a-hack-2027"},
        period="First five minutes of play")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    viewer_export.write(args.out, run)
    print("wrote {}  ({:.1f} MB)".format(args.out, os.path.getsize(args.out) / 1e6))
    if args.page:
        viewer_export.write_page(args.page, run, args.viewer, name=os.path.basename(args.out))
        print("wrote {}  (the viewer with these five minutes built in)".format(args.page))
    inside = truth[start:stop]
    spells = [code for _, _, code in viewer_export.segments(inside) if code in (H, A)]
    print("  {} frames ({:.0f} s), {} of them labelled, {} possessions".format(
        stop - start, (stop - start) / FPS, int(np.sum(inside != X)), len(spells)))
    print("  the video starts {:.1f} s in (video.offset_frames {})".format(start / FPS, start))
    print("  these are the organisers' labels; nothing here comes from the model")
    print("  this file holds match tracking: keep it on this machine")


if __name__ == "__main__":
    main()
