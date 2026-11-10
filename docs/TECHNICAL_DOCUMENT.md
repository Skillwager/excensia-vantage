# Excensia Vantage: possession from automatic tracking, learned on other matches

Technical document, More than a Hack 2027, Phase 1. Team: **Excensia**.
Repository: https://github.com/Skillwager/excensia-vantage

## 1. Approach

The task is a sequence labelling problem with one hard constraint: nothing may be learned or
tuned on the match being labelled. So the method is built the other way round. Everything is
learned on **ten other matches** whose possession is known, after those matches have been made
to look like automatic tracking, and the result is applied to the competition match unchanged.

1. **Other matches, labelled the way the ground truth is.** Seven matches of the open DFL
   dataset and Metrica's three public sample games give frame-level possession and dead-ball
   labels (section 3). They are used as the sources give them, short possessions included,
   because the ground truth has no minimum length for a possession (section 4).
2. **A simulator of automatic tracking.** Corrected tracking has a ball on every frame and every
   player all game; automatic tracking does not. Each training half is passed through a simulator
   that removes the ball the way a tracker loses it (at many stoppages, and for long spells in
   live play), follows it through other stoppages while the players move it about, drops and
   duplicates players, and adds slow position error. The simulator's *shape* was set from
   label-free totals of the provided tracking (how long the ball stays visible, how long it stays
   away, what it does while the players are slow, how far it is from the nearest player, how
   often a team has an extra body). The quantities that cannot be measured without labels, such
   as how often a missing ball means a stoppage and how often a stoppage keeps its ball, are
   drawn from wide ranges on every pass, so the model cannot rely on one answer.
3. **Features that do not need identities or a ball on every frame.** For each frame: the ball
   and who is near it when it is seen; when it is not, how long it has been gone, how long until
   it returns, and what the picture was at both ends of the gap; which team last controlled it
   and which controls it next; which team last touched it and which touches it next, a touch
   being a change in the ball's velocity next to a player; where each team is, how spread, how
   fast, moving forward or back; the same signals a second, three seconds and ten seconds
   either side; and how far the frame is from a tagged restart (section 2). 255 numbers, all
   properties of sets of players, so track slots and their order never matter.
4. **Two tree models.** A frame classifier gives the probability of home, away and dead ball.
   A boundary model answers a different question from the same features: is a turnover, a
   stoppage or a restart happening right here? Both are gradient-boosted trees (LightGBM). Each
   half is scored twice, as it is and with the teams exchanged and the pitch rotated, and the two
   answers are averaged, so the model cannot favour the side called "home".
5. **Decoding into possessions.** The ranking metric counts possessions as intervals, so the
   labels come from a Viterbi path over home, away and dead ball, not from the best label per
   frame. Changing state costs something everywhere, less where the boundary model sees a
   change and more where it does not; runs shorter than a minimum are absorbed. Smart Tagging
   enters here as bounded votes (section 2).

Why this rather than something else. Rules written by hand on nearest-player distance are the
published baseline, and its failure is the one the metric punishes: flicker, and no notion of a
stoppage. A neural sequence model on raw positions would need far more than ten matches. Trees on
engineered features learn from ten matches, take missing values as they are, and give the same
output on any machine, which the reproduction rule needs.

## 2. Assets: what is used, what is not

| Asset | Use |
|---|---|
| Tactical tracking (`_atd.xml`, `_atd.txt`) | The input. Positions and team membership of the track slots, and the ball. The file's per-slot speed is not used; speeds are recomputed from positions the same way for every match. |
| Smart Tagging: set pieces | The 59 set-piece tags are used as restarts: four of the model's inputs say how long until and since a tagged restart and, where the tag names a team, whose it is. Each tag is a clip of fixed length, and the clip starts a fixed time before the restart it shows: the two start-of-half kick-off tags begin 3.52 and 3.28 seconds before the kick-off frames the organisers publish, and a label-free profile of the tracking around the other set pieces has its step at the same place (`scripts/set_piece_profile.py`). So a tag puts a restart 3.5 seconds into its clip. No other match has these tags, so for training they are imitated on the external matches from their own labels: a share of the real restarts is tagged, with an error in time, some with the team, and a few tags point at nothing. How many, how exact and how often with a team are drawn from wide ranges (`external.tagged_restarts`). The model was therefore validated using tags the same way it uses them on the match. A tag never sets a label. |
| Smart Tagging: phases | A light vote for the team, added before decoding, from the four attacking phases only (build up, progression, creating chances, attacking transition). One tag is worth a fixed total of 3 in log-odds however long it is; that weight is a reasoned constant, because no other match has these tags to tune it on. A label-free check (`scripts/tag_check.py`) decided what votes. The phases are tagged where the ball is tracked (it is missing on 7% of their frames against 41% overall), so they add little where help is needed; on live frames they agree with tracking-only labels 92 to 98% of the time for build up and progression. The defending-phase tags repeat the attacking ones interval for interval and are not counted twice. Ball in final third, ball in the box, players in the box and long ball run through stoppages and through the other team's possession and are not used. |
| Published constants | The graded frame ranges, the second-half kick-off frame and the fact that the graded frames before it are dead ball (DATA.md, FAQ.md). Ungraded frames are written `X`. |
| Public 5-minute ground truth | Used to read the definition and to check the pipeline, as CHALLENGE.md and EVALUATION.md ask. No number was fitted to it, and no version was chosen by its score. The full account is below the table. |
| Homography, tactical video | Not used. |
| Broadcast video, tracking and homography | Not used. Aligning its timeline to the tactical one was considered as a way to recover the ball while the tactical tracker has lost it, and left out for lack of a way to validate it on other matches. |

**What the public five minutes were used for, in full.** Three things were learned from them,
and each changed the method.

1. *The definition has no minimum length.* They contain possessions of 0.8 and 1.3 seconds in
   the middle of live play. A first version had read "keeps the ball for a few seconds" as a
   rule and merged shorter spells in its training labels. That reading was wrong and was dropped
   (section 4).
2. *A pipeline check failed on dead ball.* Run on a finished file, the check found frames aligned
   and home and away the right way round, but only a quarter of the dead-ball frames marked dead.
   Totals over the window showed why: two of its four stoppages have a ball on every frame, and
   the simulator had assumed a stoppage almost always hides the ball. The assumption was removed:
   the chance that a stoppage keeps its ball is now drawn from 0 to 0.7.
3. *What a ball does in such a stoppage* was then taken from the whole recording without labels
   (`scripts/slow_report.py`): while the players are close to standing still, the tracked ball is
   at rest on only a fifth of the frames. The simulator was changed to move it, and compared with
   the recording on those label-free totals, not on the window.

The check itself is pass or fail, on criteria written down before the final file was checked:
teams agree where both files say the ball is alive, dead ball is found in stoppages with and
without a tracked ball, and the number of possessions is of the right order. Its spellF1 is
reported in section 5 for completeness. Seventeen possessions cannot rank two versions, and no
version was chosen that way; versions were compared on other matches only.

**How the provided match was handled.** It is read at inference, and it was read for label-free
totals: `scripts/match_check.py` compares the loader against the figures in DATA.md (it
reproduces them exactly: the ball on 94,214 of 158,857 graded frames, 20.5 tracked players on
average), and `scripts/domain_report.py` prints percentiles and shares for a whole half, never a
frame. Those totals shaped the simulator's ranges. No frame of the match was labelled, corrected
or inspected by a person for this method, no constant refers to a frame or a moment of the
match, and no output was compared with the video.

**AI assistance.** The code and this document were written with a hosted AI coding assistant
(Claude). The tracking rows, tags, homography and video of the match were kept out of its
context: it worked through scripts that run on the local machine and print totals (counts,
shares, percentiles) over a half, a five-minute block, the public window, a stoppage of that
window or all the tags of one kind, never frames. The scripts are in `scripts/` and print the
same totals for anyone who runs them. The finest things it saw are the share of each of the four
public stoppages that has a tracked ball, and how far the two kick-off tags start before the
published kick-off frames (3.52 and 3.28 seconds). One exception, before the team had read that
rule: a file search on a cloud drive returned to the assistant the opening lines, about a
thousand characters each, of the two tracking metadata files, and the bundle's README. They
describe the file layout, and the method takes nothing from them that DATA.md does not publish.

## 3. Training data

All of it is from other matches. None of it is from the provided game.

| Dataset | Matches | Labels | Who produced the labels |
|---|---|---|---|
| Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, doi:10.6084/m9.figshare.28196177.v1, CC BY 4.0 | 7 (2 Bundesliga, 5 2. Bundesliga, 2022/23) | Per-frame `BallStatus` (in play / dead) and `BallPossession` (home / guest) in the official position data | The league's official data provider, as published. About 1.0 million frames. |
| Metrica Sports sample data, https://github.com/metrica-sports/sample-data | 3 anonymised matches | Derived by code from the published files: a stretch with no ball position of half a second or more is a dead ball (in this corrected tracking the ball has no position while play is stopped); while play is live the ball belongs to the team of the latest on-ball event (set piece, pass, carry, shot, recovery, ball lost, ball out) | Metrica's tracking and events as published; the derivation is `vantage/metrica_sample.py`. About 0.43 million frames. |

We added no labels of our own to any match. `scripts/fetch_external.py` downloads the first
dataset and checks every file against its published checksum; the second is a `git clone`.

## 4. The possession rules implemented

Most rules here are learned from the labelled matches, so the rules are in how those labels are
defined and in how the decoder is constrained.

- **Three states.** Home, away, dead ball. Abstentions (`X`) are written only on ungraded frames.
- **No minimum length.** The challenge text asks for controlled possession and offers "a player
  keeps the ball for a few seconds" as a proxy. It also says the ground truth is taken
  mechanically from corrected tracking with nothing smoothed, and that where a reading of the
  text disagrees with the public five minutes, the ground truth is right. Those five minutes
  contain possessions of 0.8 and 1.3 seconds between two spells of the other team. So a short
  possession is a possession, and the labels of the external matches are used as given:
  in Metrica's sample games the ball belongs to the team of the latest on-ball event (28% of
  those possessions are under two seconds), and the DFL flag is the provider's own. The decoder's
  minimum length is whatever cross-validation against such labels chooses (section 5).
- **Touches that do not count.** A challenge, a deflection and a duel won without control are
  not on-ball events in Metrica's data, so they do not change possession in the labels the model
  learns from. That is the challenge's own list.
- **Loose and contested balls** stay with the last team on the ball. That is how the labels are
  built, and the features give the model the last and the next team to control or touch it.
- **A ball that is not tracked.** Never interpolated. A gap is described by its length and by
  what the ball and the players were doing at both ends and throughout; the model decides
  whether it is a stoppage, live play, or one then the other.
- **Dead ball** is whatever the labelled matches mark as dead: from the ball leaving play or the
  whistle to the restart. Stoppages shorter than the decoder's minimum are not created.
- **Dead ball with the ball in view.** A stoppage is not the same thing as a missing ball. The
  tracker follows the ball through some stoppages, and then the evidence is elsewhere: the
  players slow to a walk, the ball is near a line or is carried to a spot and waits there. The
  model is trained on stoppages of both kinds.
- **Restarts** are placed where the evidence says play resumed: near a tagged set piece where
  there is one (section 2), and otherwise a little before the tracker picks the ball up again.
- **Minimum lengths and switching costs** are chosen by cross-validation on the external matches
  (section 5) and stored in `models/vantage.json`.

## 5. Validation, on other matches only

Five folds over the ten matches; every held-out match is scored whole, with the organisers' own
metric code, under several simulated recording conditions. One decoding setting is chosen for all
folds and conditions together.

<!-- RESULTS -->
*Table filled in from `python scripts/cvb.py summary --tag v4` before submission.*
<!-- /RESULTS -->

A model trained only on the seven DFL matches scores the same on Metrica's three sample games as
on held-out DFL matches, so what it learns is not one provider's labelling habit.

## 6. Limitations

- The simulator is the weakest link. How often the real tracker loses the ball in live play, and
  for how long, is not known and cannot be measured without labels, so the model is trained to
  be reasonable across a range instead of right for this recording.
- Possession while the ball is untracked rests on the players alone and is the least reliable
  part of the output, as is the exact moment of a stoppage that happens with the ball out of
  sight.
- What counts as a possession in the ground truth is known from the challenge text and from
  seventeen public possessions, not from a specification. The external labels are the closest
  thing available (Metrica's own events, and another provider's flag), and a systematic
  difference between them and the ground truth would cost short possessions first. Short
  possessions are also the hardest to find: in the external labels more than a quarter of all
  possessions last under two seconds, the method finds about a third of those under three, and
  each one missed also spoils the match of the two possessions either side of it.
- Ten matches. Rare situations (a drop ball, a long VAR stoppage with play behind it) are barely
  represented.
- The phase tags are weighted by a reasoned constant, not a fitted one. The set-piece tags are
  used the way imitated tags were used in training; how good the imitation is cannot be checked
  on the recording. One sign that they are imperfect: before two of the ten tagged goal kicks
  the model still sees live play, which is what a goalkeeper playing the ball out of his hands
  would look like.
- With more time: aligning the broadcast tracking to recover the ball in live play, a learned
  model of the tags once more tagged matches exist, and more external matches.

## 7. Reproduction

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m vantage run --data-dir /path/to/data --out Excensia.csv
```

Python 3.12; numpy, scipy and lightgbm at the versions pinned in `requirements.txt`. On Linux,
LightGBM needs the OpenMP runtime (`apt-get install libgomp1`). About a minute on a laptop, no
GPU, no network. The trained models are in `models/` with their SHA-256 in `models/vantage.json`;
the run refuses a model file that does not match. The command prints the SHA-256 of the file it
writes and saves a receipt next to it with the hashes of the inputs.

The same file, byte for byte, was produced on Windows 11 (Python 3.12.13) and on Ubuntu 24.04
(Python 3.12.3). The feature code uses only exact arithmetic (no transcendental functions and no
linear-algebra library), so the result does not depend on the processor's maths library.

The submitted file has SHA-256 `<filled in at submission>`.

Retraining from the public sources is documented in the README. It is not needed to reproduce
the file.
