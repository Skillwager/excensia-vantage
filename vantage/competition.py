"""The competition match: load its automatic tracking, and write the submission file.

Every constant below is published by the organisers (DATA.md, FAQ.md, games.py in the official
repository). None of them was read off the match by this project.

The match data itself is licensed for the competition only. This module reads it from a folder
outside the repository and never writes any of it back out: the only output is the label file.
"""

import os

import numpy as np

from . import D, LABELS, X, epts
from .half import Half, home_attacks_positive_x

N_FRAMES = 199319
PITCH = (105.3, 68.0)                       # games.py: real pitch size in metres
# Graded video frames, inclusive (DATA.md, "Which frames are graded").
GRADED = {1: (11185, 90651), 2: (110808, 190197)}
# The second half is graded from 110808 but kicks off at 113742; the frames in between are graded
# and all dead ball (FAQ.md, "Which frames are actually graded?").
KICKOFF = {1: 11185, 2: 113742}
# The organisers' clarification of the rules (3 November 2026) does not allow output to be
# checked against the video. A run file of this match carries this sentence, and the viewer
# then opens no video beside its labels.
VIDEO_WITHHELD = ("This is the test match of More than a Hack 2027. Its rules do not allow "
                  "output to be checked against the video, so no video is played beside "
                  "these labels.")


def tactical_paths(data_dir):
    folder = os.path.join(data_dir, "tactical")
    stem = os.path.join(folder, "BAR-ATH_tactical")
    return {"meta": stem + "_atd.xml", "raw": stem + "_atd.txt",
            "tagging": stem + "_smart_tagging.xml", "homography": stem + "_homography.csv"}


def load_halves(data_dir):
    """[Half, Half]: the two halves from kick-off to the last graded frame, in metres.

    Row i of a half is video frame `half.first_frame + i`. The tracking counts frames from 1 and
    the video from 0 (DATA.md, "Frame alignment"), so record k of the file is video frame k - 1;
    the file is checked to be exactly one record per video frame before anything else is done.
    """
    paths = tactical_paths(data_dir)
    data = epts.read(paths["meta"], paths["raw"])
    if len(data.frame) != N_FRAMES or not np.array_equal(data.frame, np.arange(1, N_FRAMES + 1)):
        raise ValueError("tracking is not one record per video frame numbered from 1")

    metres = np.empty_like(data.xy)
    metres[..., 0] = (data.xy[..., 0] - 0.5) * PITCH[0]
    metres[..., 1] = (data.xy[..., 1] - 0.5) * PITCH[1]
    ball = np.empty_like(data.ball)
    ball[:, 0] = (data.ball[:, 0] - 0.5) * PITCH[0]
    ball[:, 1] = (data.ball[:, 1] - 0.5) * PITCH[1]

    halves = []
    for period in (1, 2):
        first, last = KICKOFF[period], GRADED[period][1]
        rows = slice(first, last + 1)
        home_xy = metres[rows][:, data.team == 0]
        away_xy = metres[rows][:, data.team == 1]
        ball_xy = ball[rows].copy()
        flipped = not home_attacks_positive_x(home_xy, away_xy)
        if flipped:
            home_xy, away_xy, ball_xy = -home_xy, -away_xy, -ball_xy
        halves.append(Half(match="BAR-ATH", period=period, pitch=PITCH, ball_xy=ball_xy,
                           home_xy=home_xy, away_xy=away_xy, first_frame=first,
                           meta={"source": "competition", "flipped": flipped}).check())
    return halves


def assemble(decoded_halves):
    """Full-length label codes from the decoded halves [(first_video_frame, codes)].

    Ungraded frames are X. The graded second-half frames before its kick-off are D, which the
    organisers state outright.
    """
    labels = np.full(N_FRAMES, X, dtype=np.int8)
    labels[GRADED[2][0]:KICKOFF[2]] = D
    for first, codes in decoded_halves:
        labels[first:first + len(codes)] = codes
    graded = np.zeros(N_FRAMES, dtype=bool)
    for first, last in GRADED.values():
        graded[first:last + 1] = True
    labels[~graded] = X
    if (labels[graded] == X).any():
        raise ValueError("a graded frame was left without a label")
    return labels


def write_submission(path, labels):
    """frame,label with one row per video frame, LF line ends, exactly as SUBMISSION.md asks."""
    if len(labels) != N_FRAMES:
        raise ValueError("{} labels, expected {}".format(len(labels), N_FRAMES))
    letters = np.array(LABELS)[labels]
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n", encoding="ascii") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(index, letter)
                             for index, letter in enumerate(letters)))
    return path
