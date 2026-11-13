"""The external matches: where they are, how they are labelled, and how training tables are made.

Ten matches from two open sources are used for everything that is learned or tuned:

* seven from the open DFL dataset (vantage/dfl.py), and
* Metrica's three public sample games (vantage/metrica_sample.py).

The competition match is not part of this module.
"""

import os
import zlib

import numpy as np

from dataclasses import replace

from . import A, D, FPS, H, X, dfl, metrica_sample, skillcorner
from .degrade import (SEVENTH_ONLY, SIXTH_ONLY, DegradeConfig, degrade, sample_config,
                      sample_config_fifth, sample_config_found_at_kicks, sample_config_seventh,
                      sample_config_sixth)
from .features import build
from .labels import CONTROL_SECONDS, change_distances, harmonise, run_lengths

# The repository root, where data/ and runs/ live. VANTAGE_ROOT overrides it, so that a frozen
# copy of the code can be run against the same data while the working copy is being edited.
ROOT = os.environ.get("VANTAGE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
METRICA_DIR = os.path.join(ROOT, "data", "external", "metrica-sample")
CACHE = os.path.join(ROOT, "data", "cache")

METRICA_MATCHES = tuple("metrica{}".format(game) for game in metrica_sample.GAMES)
ALL_MATCHES = tuple(dfl.MATCHES) + METRICA_MATCHES

# A third source, kept apart from the ten above: SkillCorner's open A-League matches, which are
# automatic tracking as it was recorded (vantage/skillcorner.py). Nothing is simulated on them.
SKILLCORNER_DIR = os.path.join(ROOT, "data", "external", "skillcorner")
SKILLCORNER_MATCHES = skillcorner.MATCHES

# The simulated recording conditions a held-out match is scored under. None is the corrected
# tracking as published; "random" draws a setting from the same ranges the training passes use.
CONDITIONS = {
    "clean": None,
    "typical": DegradeConfig(),
    "easy": DegradeConfig(live_loss_rate=0.2, long_miss=0.0, player_missing=0.03, blackout=0.0,
                          extra_share=0.03, false_rate=0.0),
    "harsh": DegradeConfig(dead_hidden=0.7, lose_shift=-1.0, lose_spread=1.5, find_lag=1.5,
                           long_miss=0.12, live_loss_rate=1.6, live_loss_median=10.0,
                           until_restart=0.5, false_rate=0.15, player_missing=0.1,
                           player_drift=0.35, blackout=0.5, extra_share=0.2),
    # stoppages mostly tracked right through, and the ball lost more often in live play
    "watched": DegradeConfig(dead_hidden=0.3, dead_watched=0.65, dead_watched_in=0.85,
                             hold_at_line=0.7, early_find=0.4, live_loss_rate=1.4,
                             live_loss_median=8.0),
    "random1": "random",
    "random2": "random",
}


# The fourth generation loses and finds the ball as the recording's tracker does, read off the
# recording without labels (scripts/track_report.py). There a track of the ball begins at a
# player's feet (0.66 m from the nearest player in the middle case, within 1.5 m at two starts
# in three): the tracker picks the ball up as it is struck. It ends near a player more often
# than not. And the ball is lost more often and for longer in live play than the first
# generations drew it: 1.4 tracks a minute, half of them under 14 seconds, half of the gaps over
# 11. After a restart the ball is not picked up at once: over the 57 tagged set pieces the
# nearest beginning of a track is 68 frames after the restart in the middle case, within a
# second at about one in four, and before the restart at one in ten (a restart delivered
# through the air is not tracked until it is played on the ground). The settings below are the
# ones whose totals come nearest to all of that.
FOUND_AT_KICKS = dict(find_at_kick=0.9, find_at_kick_restart=0.25, lose_at_player=0.85,
                      lose_at_player_dead=0.3, find_offset=0, live_loss_rate=2.2,
                      live_loss_median=8.0, live_loss_spread=0.45, ball_blips=0.15, until_restart=0.15,
                      dead_watched=0.35, early_find=0.1, find_quick=0.28, find_lag_quick=0.6,
                      find_lag=2.9, find_lag_spread=0.45, soften_first=False)
CONDITIONS_G4 = {
    "clean": None,
    "typical": DegradeConfig(**FOUND_AT_KICKS),
    "easy": DegradeConfig(**dict(FOUND_AT_KICKS, live_loss_rate=0.5 * FOUND_AT_KICKS["live_loss_rate"],
                                 long_miss=0.0, player_missing=0.03, blackout=0.0, extra_share=0.03,
                                 false_rate=0.0)),
    "harsh": DegradeConfig(**dict(FOUND_AT_KICKS, live_loss_rate=1.2 * FOUND_AT_KICKS["live_loss_rate"],
                                  live_loss_median=9.5, until_restart=0.25, find_at_kick=0.6,
                                  find_at_kick_restart=0.15, lose_at_player=0.6, lose_at_player_dead=0.15,
                                  dead_hidden=0.7, lose_shift=-1.0, lose_spread=1.5,
                                  find_lag=3.6, find_quick=0.15, long_miss=0.12, false_rate=0.15,
                                  player_missing=0.1,
                                  player_drift=0.35, blackout=0.5, extra_share=0.2)),
    # stoppages mostly tracked right through, and the ball lost more often in live play
    "watched": DegradeConfig(**dict(FOUND_AT_KICKS, live_loss_rate=1.25 * FOUND_AT_KICKS["live_loss_rate"],
                                    live_loss_median=9.0, dead_hidden=0.3, dead_watched=0.65,
                                    dead_watched_in=0.85, hold_at_line=0.7, early_find=0.25)),
    "random1": "random",
    "random2": "random",
}


# The fifth generation. The fourth read the recording's set-piece tags as the moment play
# restarts and concluded that its tracker finds the ball some three seconds late. More totals,
# again read off the recording without a label (scripts/track_report.py), say otherwise, and
# each of them the fourth generation got wrong:
#
#   * After a tagged throw-in the track begins at the touchline (half a metre outside it in the
#     middle case) with the ball moving in at 12 m/s: the tracker picks the ball up as it is
#     thrown. It is the tag that is early, by 2.7 seconds in the middle case.
#   * The ball is found about ten metres from where it was lost (4/10/20 m), as far as a restart
#     is from where play stopped in the open matches' labels (3/7/13 m). The fourth generation
#     found it 21 to 27 metres away.
#   * No gap lasts between 0.6 and 5.16 seconds: 14 are shorter, 129 longer. A ball the tracker
#     loses for less it bridges.
#   * More than half of the gaps of five to ten seconds begin and end in open field, with the
#     ball a few metres from where it was and the players at a stoppage's pace: a ball that
#     lies still next to a player for five seconds is lost, and found as it is struck.
#
# So here a stoppage's ball is picked up as the restart is struck; a ball that lies still within
# a player's reach is lost until it is played, by one rule that does not ask whether play is
# live (the keeper with the ball in his hands, and the ball that waits with whoever takes a
# restart in a stoppage the tracker follows); no gap is shorter than 5.36 seconds except the
# blips, which fall where the ball is fast; and the imitated tags run early as the real ones do
# (`tagged_restarts`). A pick-up at a kick and a loss at a player have the same chance in play
# and out of it. An independent review of the first cut of this generation found two ways it
# still gave a stoppage away (a resting ball was lost by different rules in play and out of it,
# and a short gap was bridged with the source's own motionless ball); both are gone, and
# tests/test_tracks.py holds them out.
FIFTH = dict(find_at_kick=0.9, find_at_kick_restart=0.9, find_offset=0, find_lag_spread=0.4,
             find_quick=0.85, find_lag_quick=0.1, find_lag=1.0, restart_back=12, restart_wait=1.0,
             early_find=0.1, until_restart_as_restart=True,
             lose_at_player=0.85, lose_at_player_dead=0.85,
             hold_loss=0.85, hold_delay=0.5, dead_watched=0.3, dead_watched_in=0.4, out_margin=3.0,
             live_loss_rate=1.2, live_loss_median=7.0, live_loss_spread=0.7, until_restart=0.1,
             min_gap=5.36, ball_blips=0.14, blip_frames=(6, 12), blip_fast=100.0, soften_first=False)
CONDITIONS_G5 = {
    "clean": None,
    "typical": DegradeConfig(**FIFTH),
    "easy": DegradeConfig(**dict(FIFTH, live_loss_rate=0.5, dead_watched=0.5, long_miss=0.0,
                                 player_missing=0.03, blackout=0.0, extra_share=0.03, false_rate=0.0)),
    "harsh": DegradeConfig(**dict(FIFTH, live_loss_rate=1.8, live_loss_median=9.0, until_restart=0.2,
                                  find_quick=0.6, find_at_kick=0.6, find_at_kick_restart=0.6,
                                  lose_at_player=0.6, lose_at_player_dead=0.6, dead_watched=0.2,
                                  hold_loss=1.0, lose_shift=-1.0, lose_spread=1.5, long_miss=0.12,
                                  false_rate=0.15, player_missing=0.1, player_drift=0.35, blackout=0.5,
                                  extra_share=0.2)),
    # stoppages mostly tracked right through, and the ball lost more often in live play
    "watched": DegradeConfig(**dict(FIFTH, live_loss_rate=1.4, dead_watched=0.7, dead_watched_in=0.85,
                                    hold_loss=0.6, hold_at_line=0.7, early_find=0.25)),
    "random1": "random",
    "random2": "random",
}


# The sixth generation is the fifth without what a fifth independent review found the ball
# itself still saying about a stoppage, things only a simulator supplies:
#
#   * In the DFL matches the published ball stands still from the first dead frame on: 85% of
#     their stoppages have a change of velocity of 3 m/s or more across that frame, where live
#     play has it at 23% and another tracker's real recordings (SkillCorner's) at 33% across
#     their own stoppages' first frame. A ball rolls on when the whistle goes. So the DFL
#     matches' stoppages get the drawn track Metrica's sample games always had (`observed`), and
#     the path drawn for a stoppage the tracker follows begins with the ball's own motion.
#   * A stoppage whose own gap would come out under the floor is followed, not bridged with
#     what the source has there.
#   * Two false tracks in one gap leave no piece under the floor between them, and a blip may
#     show as a gap of up to eleven frames.
SIXTH = dict(FIFTH, **SIXTH_ONLY)
CONDITIONS_G6 = {name: (replace(setting, **SIXTH_ONLY) if isinstance(setting, DegradeConfig) else setting)
                 for name, setting in CONDITIONS_G5.items()}


# The seventh generation is the sixth without what a sixth independent review found the drawn
# dead ball still saying: its roll-on followed one law at every stoppage (speed falling off by
# 1/e a second from the first dead frame on, so that the speed twelve frames on was always 0.62
# of the speed now), the path drawn for a followed stoppage had steps of metres a frame and a
# hop where its wait began, and a ball in the air came down in one frame. Now the ball rolls on
# as a free ball does, slowed at a steady rate drawn for each stoppage; no drawn pass is faster
# than 20 m/s; and the height falls over half a second. This is the generation the shipped
# models were trained on.
SEVENTH = dict(FIFTH, **SEVENTH_ONLY)
CONDITIONS_G7 = {name: (replace(setting, **SEVENTH_ONLY) if isinstance(setting, DegradeConfig) else setting)
                 for name, setting in CONDITIONS_G5.items()}


def conditions():
    """The scoring conditions of the generation in use."""
    return {"g4": CONDITIONS_G4, "g5": CONDITIONS_G5, "g6": CONDITIONS_G6,
            "g7": CONDITIONS_G7}.get(generation(), CONDITIONS)


def sampled(rng):
    """A random noise setting of the generation in use."""
    made = generation()
    if made == "g7":
        return sample_config_seventh(rng)
    if made == "g6":
        return sample_config_sixth(rng)
    if made == "g5":
        return sample_config_fifth(rng)
    return sample_config_found_at_kicks(rng) if made == "g4" else sample_config(rng)


def condition_half(half, match, name):
    """One labelled half as it would be recorded under the named condition. Deterministic."""
    rng = np.random.default_rng(seed_for("eval", match, half.period, name))
    setting = conditions()[name]
    noisy = observed(half, rng, sampled(rng) if setting == "random" else setting)
    return with_tags(noisy, np.random.default_rng(seed_for("tags", "eval", match, half.period, name)))


def seed_for(*parts):
    """A stable 32-bit seed from names, so every table can be rebuilt bit for bit."""
    return zlib.crc32("|".join(str(part) for part in parts).encode("utf-8"))


def source_of(match):
    if match.startswith("metrica"):
        return "metrica"
    return "skillcorner" if match.startswith("sc") else "dfl"


def labelled_halves(match):
    """The match's halves with labels brought to the challenge's definition of possession."""
    source = source_of(match)
    if source == "metrica":
        halves = metrica_sample.load_game(int(match[-1]), METRICA_DIR)
    elif source == "skillcorner":
        halves = skillcorner.load_match(match, SKILLCORNER_DIR, CACHE)
    else:
        halves = dfl.load_match(match, DFL_DIR, CACHE)
    for half in halves:
        half.labels = harmonise(half.labels, CONTROL_SECONDS[source])
    return halves


# How much the recording's tracker smooths its paths, measured on the recording without labels
# (scripts/smoothing_report.py): a strong change of the ball's velocity is spread evenly over 21
# frames, which is what a centred moving average of 21 frames does to a kick; the start and
# end of that spread are rounded, as a second average of 5 frames rounds them; about half a
# millimetre of frame-to-frame noise is left; and the players' paths are as smooth as the
# open matches' after a moving average of 7. The first generation of simulated recordings
# smoothed the ball over 3 to 9 frames and the players not at all. VANTAGE_SIM=g1 gives those
# again, draw for draw.
BALL_SMOOTH_G2 = {3: 19, 5: 21, 7: 21, 9: 23}
BALL_SMOOTH2_G2 = 5
BALL_JITTER_G2 = 0.0005
PLAYER_SMOOTH_G2 = 7
# The third generation treats the two ends of a track as the recording's tracker does, read off
# the recording without labels (scripts/edge_report.py): a track begins at full speed, and a
# ball or a player that is lost is still reported for 5 frames where it was last seen, after
# which the average takes that place for everything that follows, so every track slows to a
# stop over its last 16 frames. The same reading shows the players' paths averaged over 21
# frames, as the ball's is, not over 7. Nothing is averaged across frames the tracker did not
# have: the second generation did that, and an independent review caught it.
PLAYER_SMOOTH_G3 = 21
TRACK_HOLD_G3 = 5
GENERATIONS = ("g1", "g2", "g3", "g4", "g5", "g6", "g7")


_STAMP = []


def code_stamp():
    """A short hash of this package as it was when the process first asked: what made a table."""
    if not _STAMP:
        import hashlib
        folder = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha1()
        for name in sorted(os.listdir(folder)):
            if name.endswith(".py"):
                with open(os.path.join(folder, name), "rb") as handle:
                    digest.update(handle.read().replace(b"\r\n", b"\n"))
        _STAMP.append(digest.hexdigest()[:12])
    return _STAMP[0]


def generation():
    value = os.environ.get("VANTAGE_SIM", "g7")
    if value not in GENERATIONS:
        raise SystemExit("VANTAGE_SIM is {!r}; it has to be one of {}".format(value, ", ".join(GENERATIONS)))
    return value


def observed(half, rng, config):
    """What automatic tracking might have recorded of this half."""
    if half.meta.get("source") == "skillcorner":
        return half         # it is automatic tracking: what was recorded is what there is
    if config is not None and generation() == "g2":
        config = replace(config, ball_smooth=BALL_SMOOTH_G2.get(config.ball_smooth, config.ball_smooth),
                         ball_smooth2=BALL_SMOOTH2_G2, ball_jitter=BALL_JITTER_G2,
                         player_smooth=PLAYER_SMOOTH_G2)
    if config is not None and generation() in ("g3", "g4", "g5", "g6", "g7"):
        config = replace(config, ball_smooth=BALL_SMOOTH_G2.get(config.ball_smooth, config.ball_smooth),
                         ball_smooth2=BALL_SMOOTH2_G2, ball_jitter=BALL_JITTER_G2,
                         player_smooth=PLAYER_SMOOTH_G3, edges="keep", track_hold=TRACK_HOLD_G3)
    friction = generation() == "g7"          # the dead ball rolls on as a free ball does
    if half.meta.get("source") == "metrica-sample":
        half = metrica_sample.with_dead_ball_track(half, rng, friction)
        if config is not None:
            half = metrica_sample.with_loose_ball(half, rng)
    elif config is not None and generation() in ("g6", "g7") and half.labels is not None:
        # The DFL matches publish a ball that stands still from the first dead frame of a
        # stoppage to its last. No ball does that, and a model would learn the stop as the
        # stoppage. Their stoppages get the same drawn track: the ball rolls on and dies away,
        # is carried to the restart spot and waits there. Its height is taken as nought there.
        drawn = metrica_sample.with_dead_ball_track(half, rng, friction)
        height = None
        if half.ball_z is not None:
            height = np.nan_to_num(half.ball_z.astype(np.float32), nan=0.0)
            dead = half.labels == D
            if friction:
                # a ball in the air when play stops comes down over half a second
                edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
                for start, stop in zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()):
                    frames = min(SEVENTH_ONLY["height_fall"], stop - start)
                    top = height[start - 1] if start > 0 else 0.0
                    height[start:stop] = 0.0
                    height[start:start + frames] = top * (1.0 - np.arange(1, frames + 1, dtype=np.float32)
                                                          / np.float32(SEVENTH_ONLY["height_fall"]))
            else:
                height[dead] = 0.0
        half = replace(drawn, ball_z=height, meta=dict(half.meta, dead_ball="synthetic"))
    return degrade(half, rng, config)


def tagged_restarts(half, rng):
    """Restart tags of the kind Smart Tagging gives for set pieces, drawn from a half's labels.

    The provided recording comes with about sixty set-piece tags, each of which places a restart
    to within a second or two and, for corners, goal kicks and free kicks, names the team. No
    external match has such tags, so they are imitated: some of the half's real restarts are
    tagged, with an error in time, and some of those name the team. How many, how exact and how
    often with a team are drawn afresh each time, and a few tags point at nothing, because the
    real tags are automatic and none of this can be measured on the recording.
    """
    labels = half.labels
    t = len(labels)
    if generation() in ("g5", "g6", "g7"):
        return _tagged_early(labels, rng)
    coverage = rng.uniform(0.25, 0.75)
    with_team = rng.uniform(0.3, 0.7)
    spread = rng.uniform(0.3, 1.5)
    shift = rng.uniform(-1.0, 1.0)
    false_share = rng.uniform(0.0, 0.1)
    out = [(int(np.clip(round((shift + rng.normal(0.0, spread)) * FPS), 0, t - 1)), None)]  # kick-off
    dead = np.concatenate(([0], (labels == D).astype(np.int8), [0]))
    edges = np.diff(dead)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop >= t or stop - start < 2 * FPS or rng.random() >= coverage:
            continue
        row = stop + int(round((shift + rng.normal(0.0, spread)) * FPS))
        team = None
        if labels[stop] in (H, A) and rng.random() < with_team:
            team = int(labels[stop])
            if rng.random() < 0.03:
                team = A if team == H else H
        out.append((int(np.clip(row, 0, t - 1)), team))
    for _ in range(int(rng.poisson(false_share * len(out)))):
        out.append((int(rng.integers(0, t)), (H, A, None)[int(rng.integers(0, 3))]))
    return sorted(out, key=lambda pair: pair[0])


def _tagged_early(labels, rng):
    """The fifth generation's tags. A real tag is exact for a kick-off (that is where its place
    in the clip was read off, vantage/tagging.py), but at the set pieces the tracker does not
    follow, its restart comes 68 frames before the tracker picks the ball up in the middle case
    (24 and 92 at the quartiles), and the ball is picked up as the restart is taken. So most
    tags run early, by about three seconds, and the rest are about right. How many of each, how
    early and how exactly are drawn afresh for every half around those totals."""
    t = len(labels)
    coverage = rng.uniform(0.25, 0.75)
    with_team = rng.uniform(0.3, 0.7)
    right = rng.uniform(0.1, 0.4)             # share of the tags that put the restart where it is
    early = rng.uniform(2.4, 3.6)             # s by which the others come before it, in the mean
    spread = rng.uniform(0.7, 1.3)
    false_share = rng.uniform(0.0, 0.1)
    out = [(int(np.clip(round(rng.normal(0.0, 0.3) * FPS), 0, t - 1)), None)]            # kick-off
    dead = np.concatenate(([0], (labels == D).astype(np.int8), [0]))
    edges = np.diff(dead)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop >= t or stop - start < 2 * FPS or rng.random() >= coverage:
            continue
        before = rng.normal(0.0, 0.5) if rng.random() < right else rng.normal(early, spread)
        row = stop - int(round(before * FPS))
        team = None
        if labels[stop] in (H, A) and rng.random() < with_team:
            team = int(labels[stop])
            if rng.random() < 0.03:
                team = A if team == H else H
        out.append((int(np.clip(row, 0, t - 1)), team))
    for _ in range(int(rng.poisson(false_share * len(out)))):
        out.append((int(rng.integers(0, t)), (H, A, None)[int(rng.integers(0, 3))]))
    return sorted(out, key=lambda pair: pair[0])


def with_tags(half, rng):
    """The half with imitated restart tags attached (see `tagged_restarts`)."""
    return replace(half, meta=dict(half.meta, restarts=tagged_restarts(half, rng)))


def train_table_path(tag, match, period, variant, view):
    return os.path.join(CACHE, "train_" + tag,
                        "{}_p{}_k{}_{}.npz".format(match, period, variant, view))


def prepare_match(match, tag, variants, stride, extended=True):
    """Build and cache the training tables of one match. Existing tables are kept. With
    extended=False the tables hold the first 255 columns only, as the first models' tables did."""
    os.makedirs(os.path.join(CACHE, "train_" + tag), exist_ok=True)
    for half in labelled_halves(match):
        for variant in range(variants):
            rng = np.random.default_rng(seed_for("train", match, half.period, variant))
            noisy = observed(half, rng, sampled(rng))
            # the tags have their own stream, so adding them moved nothing else
            noisy = with_tags(noisy, np.random.default_rng(
                seed_for("tags", "train", match, half.period, variant)))
            phases = [int(rng.integers(0, stride)) for _ in range(2)]     # drawn whether or not a
            for (view, version), phase in zip((("own", noisy), ("swap", noisy.swapped())), phases):
                path = train_table_path(tag, match, half.period, variant, view)   # table is there
                if os.path.isfile(path):
                    continue
                names, table = build(version, extended)
                rows = np.arange(phase, version.n_frames, stride)
                rows = rows[version.labels[rows] != X]
                if version.meta.get("source") == "skillcorner":
                    # The broadcast cuts away during stoppages, and its tracking with it: half
                    # of this source's dead-ball frames have no players at all. The recording
                    # the method is for never does that, so those frames are not learned from.
                    seen = (np.isfinite(version.home_xy[..., 0]).sum(axis=1)
                            + np.isfinite(version.away_xy[..., 0]).sum(axis=1))
                    rows = rows[seen[rows] >= 14]
                np.savez(path, table=table[rows], target=version.labels[rows],
                         change=change_distances(version.labels)[:, rows],
                         run=run_lengths(version.labels)[rows], names=np.array(names),
                         generation=np.array(generation()), code=np.array(code_stamp()))


def load_training(tag, matches, variants):
    """The cached training tables of the given matches, as one set of arrays.

    names    the feature names
    generation  which generation of the simulator made the tables (they must all agree, and so
             must the code that made them and the features they hold)
    table    (N, F) float32 features
    target   (N,) home, away or dead ball
    change   (3, N) frames to the nearest turnover, stoppage start and restart: the boundary
             model's targets
    run      (N,) the length in frames of the possession or stoppage the row belongs to
    metrica  (N,) True for rows from Metrica's sample games
    """
    paths = [(match, train_table_path(tag, match, period, variant, view))
             for match in matches for period in (1, 2) for variant in range(variants)
             for view in ("own", "swap")]
    # First the sizes, from the small arrays only; then every table is read once, straight into
    # its place. Joining lists of arrays would hold the whole set in memory twice.
    sizes, made_by, stamps, columns = [], set(), set(), set()
    for _, path in paths:
        with np.load(path) as data:
            sizes.append(len(data["target"]))
            # tables from before the generation was written down are the first generation's
            made_by.add(str(data["generation"]) if "generation" in data.files else "g1")
            stamps.add(str(data["code"]) if "code" in data.files else "not recorded")
            columns.add(tuple(str(name) for name in data["names"]))
    if len(made_by) > 1:
        raise SystemExit("the tables of {} were made by different simulators ({}): build them again "
                         "under one tag".format(tag, ", ".join(sorted(made_by))))
    if len(stamps) > 1 or len(columns) > 1:
        raise SystemExit("the tables of {} were made by different code ({}): build them again under "
                         "one tag".format(tag, ", ".join(sorted(stamps))))
    with np.load(paths[0][1]) as data:
        names = [str(name) for name in data["names"]]
        width, table_type = data["table"].shape[1], data["table"].dtype
        target_type, change_type, run_type = data["target"].dtype, data["change"].dtype, data["run"].dtype
        change_rows = data["change"].shape[0]
    total = int(sum(sizes))
    out = {"table": np.empty((total, width), dtype=table_type),
           "target": np.empty(total, dtype=target_type),
           "change": np.empty((change_rows, total), dtype=change_type),
           "run": np.empty(total, dtype=run_type),
           "metrica": np.empty(total, dtype=bool)}
    start = 0
    for (match, path), size in zip(paths, sizes):
        with np.load(path) as data:
            out["table"][start:start + size] = data["table"]
            out["target"][start:start + size] = data["target"]
            out["change"][:, start:start + size] = data["change"]
            out["run"][start:start + size] = data["run"]
        out["metrica"][start:start + size] = source_of(match) == "metrica"
        start += size
    out["names"] = names
    out["generation"] = made_by.pop()
    return out


def row_weights(training, metrica_weight=1.0, short_weight=1.0, short_seconds=8.0):
    """A weight per training row, or None when every weight would be one.

    `metrica_weight` multiplies the rows from Metrica's sample games. `short_weight` raises the
    rows of short possessions: the ranking metric counts every possession once, however long it
    is, while an unweighted fit counts every frame once, so a forty-second possession outweighs
    a one-second one forty to one. A possession of `short_seconds` or longer keeps weight one; a
    shorter one gets short_seconds divided by its length, up to `short_weight`. Dead-ball rows
    keep weight one.
    """
    if metrica_weight == 1.0 and short_weight == 1.0:
        return None
    weight = np.ones(len(training["target"]), dtype=np.float32)
    if short_weight != 1.0:
        seconds = np.maximum(training["run"], 1) / float(FPS)
        boost = np.clip(short_seconds / seconds, 1.0, short_weight)
        weight = np.where(training["target"] == D, 1.0, boost).astype(np.float32)
    if metrica_weight != 1.0:
        weight = weight * np.where(training["metrica"], metrica_weight, 1.0).astype(np.float32)
    return weight
