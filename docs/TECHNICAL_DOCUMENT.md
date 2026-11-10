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
   In its training a frame inside a short possession counts for more: four times for a
   possession of two seconds or less, falling to once at eight seconds. The ranking metric
   counts a short possession as one possession like any other, and it has far fewer frames to
   be learned from. A boundary model answers a different question from the same features: is a
   turnover, a stoppage or a restart happening right here? Both are gradient-boosted trees
   (LightGBM). Each half is scored twice, as it is and with the teams exchanged and the pitch
   rotated, and the two answers are averaged, so the model cannot favour the side called "home".
5. **Decoding into possessions.** The ranking metric counts possessions as intervals, so the
   labels come from a Viterbi path over home, away and dead ball, not from the best label per
   frame. Changing state costs something everywhere, less where the boundary model sees a
   change and more where it does not; runs shorter than a minimum are absorbed. The phase tags
   of Smart Tagging enter here as bounded votes (section 2).

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
| Homography, tactical video | Not used by the method. The viewer in this repository can play the tactical video beside the finished labels on the user's own machine; nothing is read from it. |
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
without a tracked ball, and the number of possessions is of the right order. Its spellF1 was
read on four finished files, and all four readings are in section 5, including one that went
against the final file. Seventeen possessions cannot rank two versions, and no version
was chosen that way; versions were compared on other matches only.

**How the provided match was handled.** It is read at inference, and it was read for label-free
totals: `scripts/match_check.py` compares the loader against the figures in DATA.md (it
reproduces them exactly: the ball on 94,214 of 158,857 graded frames, 20.5 tracked players on
average), and `scripts/domain_report.py` prints percentiles and shares for a whole half, never a
frame. Those totals shaped the simulator's ranges. No frame of the match was labelled, corrected
or inspected by a person for this method, and no constant refers to a frame or a moment of the
match. The method never compares its output with the video. The viewer shows the finished labels
beside the tactical video on the team's own machine, for the team to watch, as the rules allow;
no label, correction, constant or choice between versions came from watching.

**AI assistance.** The code and this document were written with a hosted AI coding assistant
(Claude). The tracking rows, tags, homography and video of the match were kept out of its
context: it worked through scripts that run on the local machine and print totals (counts,
shares, percentiles) over a half, a five-minute block, the public window, a stoppage of that
window or all the tags of one kind, never frames. The scripts are in `scripts/` and print the
same totals for anyone who runs them. The finest things it saw are the share of each of the four
public stoppages that has a tracked ball, and how far the two kick-off tags start before the
published kick-off frames (3.52 and 3.28 seconds). Of the video it saw the length and the frame
size, from a local check that the viewer plays it in step with the labels and that takes no
picture (`viewer/scripts/tape-check.mjs`). One exception, before the team had read that
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

Five folds over the ten matches, two held out at a time. Every held-out match is scored whole,
with the organisers' own metric code, against its labels as its source gives them, under six
simulated recording conditions: easy, typical, watched (most stoppages followed right through),
harsh, and two drawn at random from the ranges used in training. The corrected tracking as
published ("clean") is reported and not used for any choice. One decoding setting serves every
fold and condition. It is chosen on Metrica's three sample games, because their labels come from
the provider that made the ground truth; the seven DFL matches are reported beside them.

spellF1, mean over the held-out matches (`python scripts/cv.py summary --tag v11b --sources metrica`):

| Held-out matches | easy | typical | watched | random 1 | random 2 | harsh | mean of the six | clean |
|---|---|---|---|---|---|---|---|---|
| Metrica sample games (3) | 0.746 | 0.748 | 0.721 | 0.748 | 0.731 | 0.662 | **0.726** | 0.781 |
| DFL matches (7) | 0.712 | 0.704 | 0.672 | 0.691 | 0.682 | 0.634 | 0.683 | 0.752 |
| All ten | 0.722 | 0.717 | 0.687 | 0.708 | 0.697 | 0.642 | 0.696 | 0.760 |

macroF1 on the same runs is 0.920 on Metrica's games and 0.926 on the DFL matches. These labels
keep every short possession, which makes them hard: 42% of the true possessions are under three
seconds. The setting chosen lets the decoder write possessions down to 0.8 seconds and stoppages
down to 0.5; floors of 0.2 and 0.4 seconds for a possession scored within 0.002 of it.

**What each step was worth**, each measured on held-out matches at its own best decoding setting:

| Step | Measured on | Before | After |
|---|---|---|---|
| Labels as given instead of merged below two to three seconds | Metrica games 1 and 2, five conditions, both models scored against the unmerged labels | 0.645 | 0.739 |
| The boundary model in the decoder | Metrica's three games, final models | 0.694 | 0.726 |
| Touch features | Four matches, five conditions, under the earlier labels | 0.754 | 0.761 |
| Tagged restarts as inputs | Metrica games 1 and 2, six conditions | 0.708 | 0.712 |
| More weight on short possessions in the frame classifier | Metrica's three games, six conditions (the seven DFL matches: 0.674 to 0.683) | 0.714 | 0.726 |

The last step is small, and it is in the file because it is consistent: taken match by match
over the six conditions, eight of the ten held-out matches gain between 0.003 and 0.022, one is
unchanged and one loses 0.001.

The simulator that follows the ball through stoppages cannot be scored this way, because it
changes the test itself: the simulated recordings it is tested on became harder and more like
the real one. It was judged on the provided recording without labels. Just before the 57 tagged
restarts that have five seconds of recording ahead of them, the file called dead ball at 41 with
the earlier simulator, at 51 with the new one, and at 52 once the tags became inputs, where the
final file still is.

**Where it still fails** (the ten held-out matches, typical condition, a possession counted as
found at an overlap of one half): possessions of twelve seconds or more are found 94 to 96% of
the time, those of six to twelve seconds 86%, three to six seconds 71%, and those under three
seconds 40%. Nearly half of the misses are short possessions swallowed by the other team's
possession around them. Stoppages of three seconds or more are found 89 to 100% of the time.

**The submitted file, without labels** (`scripts/pipeline_check.py`, `scripts/behaviour_report.py`).
Beside each total is the range of the same total for the model's own output on the ten held-out
matches over the six conditions. A file that behaved unlike anything seen in validation would
show here without a single label of the match.

| From kick-off to the end of each half | The submitted file | Held-out matches |
|---|---|---|
| Frames with no tracked ball | 41% | 20 to 62% |
| Of those, called dead ball | 90% | 44 to 93% |
| Frames with a tracked ball that are called dead ball | 15% | 10 to 45% |
| Dead ball, share of all frames | 46% | 31 to 53% |
| Possessions per minute of live play | 5.2 | 4.8 to 8.7 |
| Median possession | 6.8 s | 4.3 to 7.8 s |
| Possessions under three seconds | 21% | 20 to 35% |
| Median stoppage | 18.2 s | 10.5 to 24.8 s |

The file has 294 possessions and 108 stoppages. It sits inside the validation range on every
total, at the low end for short possessions. On the held-out matches the method finds fewer
possessions than there are (6.3 a minute against 7.8), and that is the bias to expect here too.

**Pipeline check on the public five minutes**, final file: all four criteria pass (teams agree
on 93% of the frames both call live; 16 possessions for 17; dead ball found on 81% of the
dead frames that have a tracked ball and 73% of those that do not).

For completeness, every spellF1 read on that window, each on a finished file: 0.614
(the first file, trained on merged labels), 0.578 (labels as given; the file that failed the
dead-ball criterion), 0.759 (the new simulator and the tagged restarts) and 0.594 (the final
file, which adds the weight on short possessions; its macroF1 there is 0.894). The last step
was adopted on the ten other matches before its file was read here, and it was kept when the
reading came out lower than the one before. The rules do not allow a version to be chosen on
this window, and the window could not choose in any case: the organisers' own bootstrap over
its 17 possessions (5th to 95th percentile of 1,000 resamples, printed by
`scripts/pipeline_check.py`) gives 0.40 to 0.78 for the final file and 0.54 to 0.90 for the one
before it, each reading inside the other's interval. One possession moves spellF1 there by
about six points.

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
  possessions last under two seconds, the method finds four in ten of those under three, and
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

The submitted file has SHA-256 `4684384a2d27b9b3641e79e8388e40e7537b63f27b8fa52c491caa4c66737b7b`.

Retraining from the public sources is documented in the README. It is not needed to reproduce
the file.
