"""Use the provided Smart Tagging as extra evidence.

Smart Tagging is Metrica's automatic annotation of the tactical video: 638 intervals, each with a
code and tags. The organisers describe it as "a strong free prior" and also as fallible. Both
are respected here: a tag never sets a label. Each tag casts one bounded *vote* for a team,
spread over the frames it covers, which is added to the model's own evidence before decoding.

Which tags vote, and why only these
-----------------------------------
Two label-free checks on the provided files decided it (scripts/tag_check.py prints both; they
compare the tags with the ball's visibility and with tracking-only labels, never with ground
truth):

* The intervals are clips, not measurements: every corner, throw-in, goal kick and free kick is
  exactly 35 seconds long and every kick-off exactly 5. A clip says a set piece is in there
  somewhere, not when the ball was put back in play, so set pieces and goals do not vote.
* The four attacking phases (build up, progression, creating chances, attacking transition)
  carry the team that has the ball, by the organisers' definitions (OVERVIEW.md). Each is tagged
  a second time from the defending side with the same interval; that copy is the same evidence
  and is not counted again. The remaining codes (ball in final third, ball in the box, players
  in the box, long ball) cover long stretches that run through stoppages and through the other
  team's possession, and do not vote.

The size of a vote is fixed by reasoning, not fitted: there is no other match with these tags to
fit it on, and fitting it on this match is not allowed. One tag is worth `VOTE` in log-odds in
total however long it is, about the weight of a single source that is right nineteen times in
twenty, so a wrong tag can only overturn tracking evidence that was weak to begin with. In
practice that means a tag matters where the ball is untracked and the players alone leave the
answer open, and nowhere else.
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import A, FPS, H

VOTE = 3.0                    # total log-odds of one tag, spread over its frames

IN_POSSESSION = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION"}
# The same intervals seen from the other side. Read for the checks, never voted twice.
OUT_OF_POSSESSION = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                     "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
SET_PIECE = "SET PIECES"


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

    Compares letters rather than whole strings, so a small spelling difference between the two
    files does not matter. A name that fits neither team (the published 'N/A') returns None.
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


def votes(instances, n_frames, home_name, away_name):
    """(n_frames, 3) log-odds to add to log-probabilities of home, away, dead ball."""
    out = np.zeros((n_frames, 3), dtype=np.float64)
    for item in instances:
        if item["code"] not in IN_POSSESSION:
            continue
        team = team_code(item["tags"].get("Team"), home_name, away_name)
        start, stop = max(0, item["start"]), min(n_frames, item["end"])
        if team is None or stop <= start:
            continue
        out[start:stop, team] += VOTE / float(stop - start)
    return out


def agreement(instances, labels, home_name, away_name):
    """Share of tagged live frames on which a label sequence agrees with the tags, by kind.

    A check that needs no ground truth: it compares the tags with tracking-only labels. If the
    meaning assumed for a kind were wrong (say, the team of a defending phase), its agreement
    would sit well under one half instead of well over it.
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
