"""Use the provided Smart Tagging as extra evidence.

Smart Tagging is Metrica's automatic annotation of the tactical video: 638 intervals, each with a
code and tags. The organisers describe it as "a strong free prior" and also as fallible, and both
halves of that are respected here: a tag never sets a label. Each tag casts one bounded *vote*,
spread over the frames it covers, which is added to the model's own evidence before decoding.

What each code is taken to say comes from the organisers' published definitions (OVERVIEW.md):

* The team **performing** an attacking phase has the ball: build up, progression, creating
  chances, attacking transition, long ball, ball in final third, ball in the box, players in the
  box.
* The team performing a **defending** phase does not: defending in any third, defensive
  transition. The vote goes to the other team.
* A **set piece** is a restart. The tag is a window around the restart, so the seconds before the
  window are voted dead ball and the seconds after it are voted live, for the tagged team when it
  is named (throw-ins are tagged without a team).
* After a tagged **goal**, the ball is dead until the following tagged kick-off.

The size of a vote is fixed by reasoning, not fitted: there is no other match with these tags to
fit it on, and fitting it on this match is not allowed. One tag is worth `VOTE` in log-odds in
total however long it is, about the weight of a single source that is right nineteen times in
twenty, so a wrong tag can only overturn tracking evidence that was weak to begin with.
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import A, D, FPS, H

VOTE = 3.0                    # total log-odds of one tag, spread over its frames
RESTART_BEFORE = 3.0          # seconds before a set-piece window voted dead ball
RESTART_AFTER = 2.0           # seconds after it voted live
GOAL_MAX_WAIT = 180.0         # a goal's dead ball lasts until a kick-off within this many seconds
GOAL_VOTE_PER_FRAME = 0.5     # the laws of the game, not an estimate: stronger than a phase tag

IN_POSSESSION = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                 "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"}
OUT_OF_POSSESSION = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                     "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
SET_PIECE = "SET PIECES"
GOAL = "GOALS"
KICK_OFF_AFTER_GOAL = "kick off (after goal)"


def load(path, fps=FPS):
    """[{code, start, end, tags}] with frames; end is the first frame after the interval."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start": int(round(float(instance.findtext("start")) * fps)),
            "end": int(round(float(instance.findtext("end")) * fps)),
            "tags": {(label.findtext("group") or "").strip(): (label.findtext("text") or "").strip()
                     for label in instance.findall("label")},
        })
    return out


def team_code(name, home_name, away_name):
    """H, A or None for a Team tag, matching the tag text to the tracking's two team names.

    The tagging spells one club slightly differently from the tracking metadata, so this compares
    letters rather than whole strings. A name that fits neither team (the published 'N/A')
    returns None.
    """
    import difflib

    def clean(text):
        return "".join(ch for ch in (text or "").lower() if ch.isalnum())

    target = clean(name)
    if not target or target == "na":
        return None
    scores = [difflib.SequenceMatcher(None, target, clean(candidate)).ratio()
              for candidate in (home_name, away_name)]
    if max(scores) < 0.6 or abs(scores[0] - scores[1]) < 0.1:
        return None
    return H if scores[0] > scores[1] else A


def votes(instances, n_frames, home_name, away_name, fps=FPS):
    """(n_frames, 3) log-odds to add to log-probabilities of home, away, dead ball."""
    out = np.zeros((n_frames, 3), dtype=np.float64)

    def spread(start, stop, column, total):
        start, stop = max(0, start), min(n_frames, stop)
        if stop > start:
            out[start:stop, column] += total / float(stop - start)

    kick_offs = sorted(item["start"] for item in instances
                       if item["code"] == SET_PIECE
                       and item["tags"].get("Type", "").lower() == KICK_OFF_AFTER_GOAL)

    for item in instances:
        code, start, end = item["code"], item["start"], item["end"]
        team = team_code(item["tags"].get("Team"), home_name, away_name)

        if code in IN_POSSESSION and team is not None:
            spread(start, end, team, VOTE)
        elif code in OUT_OF_POSSESSION and team is not None:
            spread(start, end, A if team == H else H, VOTE)
        elif code == SET_PIECE:
            spread(start - int(RESTART_BEFORE * fps), start, D, VOTE)
            after = (end, end + int(RESTART_AFTER * fps))
            if team is not None:
                spread(after[0], after[1], team, VOTE)
            else:
                spread(after[0], after[1], H, VOTE / 2.0)
                spread(after[0], after[1], A, VOTE / 2.0)
        elif code == GOAL:
            following = [frame for frame in kick_offs if end <= frame <= end + GOAL_MAX_WAIT * fps]
            if following:
                stop = min(n_frames, following[0])
                out[max(0, end):stop, D] += GOAL_VOTE_PER_FRAME
    return out


def agreement(instances, labels, home_name, away_name):
    """Share of tagged frames on which a label sequence agrees with the tags, per kind of tag.

    A check that needs no ground truth: it compares the tags with the tracking-only labels. If
    the meaning assumed for a code were wrong (say, the team of a defending phase), agreement for
    that kind would sit well under one half instead of well over it.
    """
    kinds = {"in_possession": [0, 0], "out_of_possession": [0, 0]}
    for item in instances:
        team = team_code(item["tags"].get("Team"), home_name, away_name)
        if team is None:
            continue
        if item["code"] in IN_POSSESSION:
            kind, expected = "in_possession", team
        elif item["code"] in OUT_OF_POSSESSION:
            kind, expected = "out_of_possession", (A if team == H else H)
        else:
            continue
        window = labels[max(0, item["start"]):max(0, item["end"])]
        live = window[(window == H) | (window == A)]
        kinds[kind][0] += int(np.sum(live == expected))
        kinds[kind][1] += int(len(live))
    return {kind: (hits / total if total else float("nan")) for kind, (hits, total) in kinds.items()}
