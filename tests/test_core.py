"""Checks on generated data only. No match data, external or competition, is needed to run them."""

import os

import numpy as np
import pytest

from vantage import A, D, H, X, competition, epts, evaluate, tagging
from vantage import decode as decoding
from vantage.features import build
from vantage.half import Half
from vantage.labels import harmonise, segments, spell_stats


def codes(text):
    return np.array([{"H": H, "A": A, "D": D, "X": X}[ch] for ch in text], dtype=np.int8)


# -- labels ------------------------------------------------------------------------------------

def test_harmonise_gives_a_short_touch_back_to_the_team_that_had_the_ball():
    labels = codes("H" * 100 + "A" * 20 + "H" * 100)
    assert np.array_equal(harmonise(labels, 2.0), codes("H" * 220))


def test_harmonise_keeps_a_controlled_possession():
    labels = codes("H" * 100 + "A" * 60 + "H" * 100)
    assert np.array_equal(harmonise(labels, 2.0), labels)


def test_harmonise_keeps_the_first_spell_after_a_restart_whatever_its_length():
    labels = codes("D" * 50 + "A" * 10 + "H" * 100)
    assert np.array_equal(harmonise(labels, 2.0), labels)


def test_harmonise_never_changes_dead_ball_frames():
    labels = codes("H" * 30 + "D" * 5 + "A" * 10 + "D" * 40 + "H" * 60)
    out = harmonise(labels, 2.0)
    assert np.array_equal(out == D, labels == D)


def test_harmonise_chain_of_short_spells_collapses_to_the_owner():
    labels = codes("H" * 80 + "A" * 30 + "H" * 30 + "A" * 30 + "H" * 80)
    assert np.array_equal(harmonise(labels, 2.0), codes("H" * 250))


def test_spell_stats_counts_possessions_not_stoppages():
    count, mean, _, _ = spell_stats(codes("H" * 50 + "D" * 25 + "A" * 100))
    assert count == 2 and mean == pytest.approx(3.0)


# -- decoding ----------------------------------------------------------------------------------

def test_viterbi_many_matches_viterbi():
    rng = np.random.default_rng(0)
    log_prob = np.log(rng.dirichlet([1.0, 1.0, 1.0], size=2000))
    configs = [decoding.DecodeConfig(cost_team=a, cost_to_dead=b, cost_from_dead=c)
               for a in (1.0, 6.0) for b in (2.0, 9.0) for c in (3.0, 5.0)]
    many = decoding.viterbi_many(log_prob, configs)
    for index, config in enumerate(configs):
        assert np.array_equal(many[index], decoding.viterbi(log_prob, config))


def test_viterbi_ignores_a_flicker_and_follows_a_real_change():
    prob = np.full((300, 3), 0.05)
    prob[:150, H] = 0.9
    prob[150:, A] = 0.9
    prob[60:63] = [0.05, 0.9, 0.05]          # three frames of away in the middle of home
    path = decoding.decode(prob, decoding.DecodeConfig(cost_team=8.0))
    assert np.array_equal(path, codes("H" * 150 + "A" * 150))


def test_minimum_lengths_absorb_short_runs():
    config = decoding.DecodeConfig(min_possession=1.0, min_dead=2.0)
    path = codes("H" * 100 + "D" * 30 + "H" * 100 + "A" * 10 + "H" * 50)
    assert np.array_equal(decoding.enforce_minimums(path, config), codes("H" * 290))


# -- the submission file -----------------------------------------------------------------------

def test_assemble_marks_ungraded_frames_and_the_second_half_run_in():
    first = np.full(competition.GRADED[1][1] - competition.KICKOFF[1] + 1, H, dtype=np.int8)
    second = np.full(competition.GRADED[2][1] - competition.KICKOFF[2] + 1, A, dtype=np.int8)
    labels = competition.assemble([(competition.KICKOFF[1], first),
                                   (competition.KICKOFF[2], second)])
    assert len(labels) == competition.N_FRAMES
    assert (labels[:competition.GRADED[1][0]] == X).all()
    assert (labels[competition.GRADED[1][1] + 1:competition.GRADED[2][0]] == X).all()
    assert (labels[competition.GRADED[2][0]:competition.KICKOFF[2]] == D).all()
    assert (labels[competition.KICKOFF[2]:competition.GRADED[2][1] + 1] == A).all()
    assert (labels[competition.GRADED[2][1] + 1:] == X).all()
    assert int((labels != X).sum()) == 158857          # the published number of graded frames


def test_assemble_refuses_a_graded_frame_without_a_label():
    short = np.full(10, H, dtype=np.int8)
    with pytest.raises(ValueError):
        competition.assemble([(competition.KICKOFF[1], short)])


def test_written_file_passes_the_official_validator(tmp_path):
    from vantage import _official_metrics as official

    labels = np.full(competition.N_FRAMES, X, dtype=np.int8)
    labels[1000:2000] = H
    labels[2000:2500] = D
    path = competition.write_submission(os.path.join(tmp_path, "team.csv"), labels)
    with open(path, "rb") as handle:
        content = handle.read()
    assert content.startswith(b"frame,label\n0,X\n1,X\n")
    assert b"\r" not in content
    assert content.count(b"\n") == competition.N_FRAMES + 1
    table = official.read_label_csv(path)
    assert official.validate_submission(table, competition.N_FRAMES) is None
    assert np.array_equal(official.label_codes(table), labels)


# -- EPTS reader -------------------------------------------------------------------------------

EPTS_META = """<?xml version="1.0"?>
<main>
 <Metadata>
  <GlobalConfig><ProviderGlobalParameters>
   <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
  </ProviderGlobalParameters></GlobalConfig>
  <Teams><Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
  <Players>
   <Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/>
  </Players>
  <PlayerChannels>
   <PlayerChannel id="p1_x" playerId="P1" channelId="x"/><PlayerChannel id="p1_y" playerId="P1" channelId="y"/>
   <PlayerChannel id="p2_x" playerId="P2" channelId="x"/><PlayerChannel id="p2_y" playerId="P2" channelId="y"/>
   <PlayerChannel id="p3_x" playerId="P3" channelId="x"/><PlayerChannel id="p3_y" playerId="P3" channelId="y"/>
  </PlayerChannels>
 </Metadata>
 <DataFormatSpecifications>
  <DataFormatSpecification startFrame="1" endFrame="2" separator=":">
   <StringRegister name="frameCount"/>
   <SplitRegister separator=";">
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/></SplitRegister>
   </SplitRegister>
  </DataFormatSpecification>
  <DataFormatSpecification startFrame="3" endFrame="3" separator=":">
   <StringRegister name="frameCount"/>
   <SplitRegister separator=";">
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/></SplitRegister>
    <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/></SplitRegister>
   </SplitRegister>
  </DataFormatSpecification>
 </DataFormatSpecifications>
</main>
"""
EPTS_RAW = "1:0.1,0.2;0.6,0.7:0.5,0.5\n2:NaN,NaN;0.61,0.71:NaN,NaN\n3:0.3,0.4;0.62,0.72:0.4,0.6\n"


def test_epts_reader_follows_the_layout_blocks(tmp_path):
    meta = os.path.join(tmp_path, "m.xml")
    raw = os.path.join(tmp_path, "t.txt")
    with open(meta, "w") as handle:
        handle.write(EPTS_META)
    with open(raw, "w") as handle:
        handle.write(EPTS_RAW)
    data = epts.read(meta, raw)
    assert data.team.tolist() == [0, 1, 0] and data.team_names == ("Home FC", "Away FC")
    assert data.frame.tolist() == [1, 2, 3]
    assert data.xy[0, 0].tolist() == pytest.approx([0.1, 0.2])
    assert np.isnan(data.xy[1, 0]).all()                        # NaN slot stays missing
    assert np.isnan(data.xy[2, 0]).all()                        # P1 is not in the second layout
    assert data.xy[2, 2].tolist() == pytest.approx([0.3, 0.4])  # the first slot is now P3
    assert data.xy[2, 1].tolist() == pytest.approx([0.62, 0.72])
    assert np.isnan(data.ball[1]).all() and data.ball[2].tolist() == pytest.approx([0.4, 0.6])
    assert epts.read_team_names(meta) == ("Home FC", "Away FC")


# -- Smart Tagging votes -----------------------------------------------------------------------

def instance(code, start, end, **tags):
    return {"code": code, "start": start, "end": end, "tags": tags}


def test_team_code_handles_the_spelling_difference_and_no_team():
    assert tagging.team_code("Barcelona", "Barcelona", "Athletic Club") == H
    assert tagging.team_code("Atletic Club", "Barcelona", "Athletic Club") == A
    assert tagging.team_code("N/A", "Barcelona", "Athletic Club") is None
    assert tagging.team_code(None, "Barcelona", "Athletic Club") is None


def test_one_tag_is_worth_one_vote_however_long_it_is():
    short = tagging.votes([instance("BUILD UP", 100, 150, Team="Home")], 1000, "Home", "Away")
    long = tagging.votes([instance("BUILD UP", 100, 600, Team="Home")], 1000, "Home", "Away")
    assert short[:, H].sum() == pytest.approx(tagging.VOTE)
    assert long[:, H].sum() == pytest.approx(tagging.VOTE)
    assert short[:, A].sum() == 0 and short[:, D].sum() == 0


def test_a_defending_phase_votes_for_the_other_team():
    vote = tagging.votes([instance("DEFENDING IN MIDDLE THIRD", 10, 60, Team="Home")],
                         200, "Home", "Away")
    assert vote[:, A].sum() == pytest.approx(tagging.VOTE) and vote[:, H].sum() == 0


def test_a_set_piece_votes_dead_before_and_live_after():
    vote = tagging.votes([instance("SET PIECES", 500, 600, Team="Away", Type="Corner Kick")],
                         1000, "Home", "Away")
    assert vote[425:500, D].sum() == pytest.approx(tagging.VOTE)
    assert vote[500:600].sum() == 0                       # inside the window: no claim
    assert vote[600:650, A].sum() == pytest.approx(tagging.VOTE)


def test_a_goal_is_dead_ball_until_the_kick_off():
    items = [instance("GOALS", 100, 200, Team="Home"),
             instance("SET PIECES", 1500, 1600, Team="Away", Type="Kick Off (After Goal)")]
    vote = tagging.votes(items, 3000, "Home", "Away")
    assert (vote[200:1425, D] == tagging.GOAL_VOTE_PER_FRAME).all()


def test_agreement_reads_a_defending_tag_as_the_other_team():
    labels = np.full(200, A, dtype=np.int8)
    items = [instance("DEFENDING IN MIDDLE THIRD", 0, 100, Team="Home"),
             instance("PROGRESSION", 100, 200, Team="Home")]
    result = tagging.agreement(items, labels, "Home", "Away")
    assert result["out_of_possession"] == 1.0 and result["in_possession"] == 0.0


# -- features and evaluation -------------------------------------------------------------------

def toy_half(seed=0, frames=400):
    rng = np.random.default_rng(seed)
    home = rng.normal([-15.0, 0.0], [12.0, 14.0], size=(frames, 11, 2)).astype(np.float32)
    away = rng.normal([15.0, 0.0], [12.0, 14.0], size=(frames, 11, 2)).astype(np.float32)
    ball = rng.normal([0.0, 0.0], [20.0, 15.0], size=(frames, 2)).astype(np.float32)
    ball[50:90] = np.nan
    home[10:20, 3] = np.nan
    return Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=home,
                away_xy=away, labels=rng.integers(0, 3, size=frames).astype(np.int8))


def test_features_are_mirror_images_under_a_team_swap():
    half = toy_half()
    names, table = build(half)
    swapped_names, swapped = build(half.swapped())
    assert names == swapped_names
    for own, other in (("h_cx", "a_cx"), ("bh_d1", "ba_d1"), ("h_speed", "a_speed"),
                       ("ctl_since_h", "ctl_since_a")):
        left = table[:, names.index(own)]
        right = swapped[:, names.index(other)]
        assert np.allclose(left, right, equal_nan=True, atol=1e-4), own
    assert np.allclose(table[:, names.index("b_ddiff")], -swapped[:, names.index("b_ddiff")],
                       equal_nan=True, atol=1e-4)


def test_features_do_not_depend_on_slot_order():
    half = toy_half(1)
    names, table = build(half)
    order = np.random.default_rng(5).permutation(half.home_xy.shape[1])
    shuffled = Half(match="toy", period=1, pitch=half.pitch, ball_xy=half.ball_xy,
                    home_xy=half.home_xy[:, order], away_xy=half.away_xy, labels=half.labels)
    _, other = build(shuffled)
    assert np.allclose(table, other, equal_nan=True, atol=1e-4)


def test_perfect_prediction_scores_one_and_halves_are_scored_apart():
    truth = [codes("H" * 200 + "D" * 50 + "A" * 200), codes("A" * 300 + "H" * 100)]
    result = evaluate.headline(evaluate.score(truth, [part.copy() for part in truth]))
    assert result["spellF1"] == pytest.approx(1.0) and result["n_true"] == 4
    assert len(segments(evaluate.stitch(truth))) == 6        # the half-time gap is its own run
