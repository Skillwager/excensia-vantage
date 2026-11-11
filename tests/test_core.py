"""Checks on generated data only. No match data, external or competition, is needed to run them."""

import os

import numpy as np
import pytest

from vantage import A, D, H, X, competition, epts, evaluate, tagging, viewer_export
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


def test_the_defending_copy_of_a_phase_and_other_codes_do_not_vote():
    items = [instance("DEFENDING IN MIDDLE THIRD", 10, 60, Team="Home"),
             instance("SET PIECES", 100, 200, Team="Away", Type="Corner Kick"),
             instance("PLAYERS IN THE BOX", 300, 400, Team="Home"),
             instance("GOALS", 500, 600, Team="Home")]
    assert not tagging.votes(items, 1000, "Home", "Away").any()


def test_a_tag_without_a_team_does_not_vote():
    assert not tagging.votes([instance("PROGRESSION", 10, 60, Team="N/A")], 200, "Home", "Away").any()


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
    for own, other in (("tch_since_h", "tch_since_a"), ("tch_h_past75", "tch_a_past75")):
        assert np.allclose(table[:, names.index(own)], swapped[:, names.index(other)],
                           equal_nan=True, atol=1e-4), own
    for signed in ("tch_last", "tch_run_next_signed"):
        assert np.allclose(table[:, names.index(signed)], -swapped[:, names.index(signed)],
                           equal_nan=True, atol=1e-4), signed


def test_tagged_restarts_read_the_same_from_either_side():
    half = toy_half()
    half.meta["restarts"] = [(100, H), (250, None), (330, A)]
    names, table = build(half)
    _, swapped = build(half.swapped())
    for same in ("sp_until", "sp_since"):
        assert np.array_equal(table[:, names.index(same)], swapped[:, names.index(same)]), same
    for signed in ("sp_next_team", "sp_last_team"):
        assert np.array_equal(table[:, names.index(signed)], -swapped[:, names.index(signed)]), signed
    until = table[:, names.index("sp_until")]
    assert until[100] == 0.0 and until[75] == 1.0 and until[331] == 30.0
    assert table[90, names.index("sp_next_team")] == 1.0
    assert table[240, names.index("sp_next_team")] == 0.0          # a tag without a team
    assert table[335, names.index("sp_last_team")] == -1.0


def test_a_half_without_tags_gets_the_far_values():
    names, table = build(toy_half())
    assert np.all(table[:, names.index("sp_until")] == 30.0)
    assert np.all(table[:, names.index("sp_next_team")] == 0.0)


def test_set_piece_tags_become_restarts_a_fixed_time_into_the_clip():
    items = [instance("SET PIECES", 100, 975, Team="Home", Type="Corner Kick"),
             instance("SET PIECES", 2000, 2875, Team="N/A", Type="Throw-in"),
             instance("PROGRESSION", 10, 60, Team="Home")]
    found = tagging.restarts(items, "Home", "Away")
    assert found == [(100 + int(round(tagging.RESTART_LEAD * 25)), H), (2000 + int(round(tagging.RESTART_LEAD * 25)), None)]


def test_touches_are_found_where_the_ball_changes_direction_next_to_a_player():
    frames = 300
    ball = np.zeros((frames, 2), dtype=np.float32)
    # Rolls right at 8 m/s, is turned back at frame 100 by a home player, and again at frame 200
    # by an away player.
    x = 0.0
    for index in range(frames):
        direction = 1.0 if index < 100 or index >= 200 else -1.0
        x += direction * 8.0 / 25.0
        ball[index] = (x, 0.0)
    home = np.full((frames, 2, 2), 40.0, dtype=np.float32)
    away = np.full((frames, 2, 2), -40.0, dtype=np.float32)
    home[:, 0] = ball[100] + np.array([0.5, 0.0])
    away[:, 0] = ball[200] + np.array([-0.5, 0.0])
    half = Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=home,
                away_xy=away)
    names, table = build(half)
    column = {name: table[:, index] for index, name in enumerate(names)}
    assert abs(int(np.flatnonzero(column["tch_since_h"] == 0)[0]) - 100) <= 2
    assert abs(int(np.flatnonzero(column["tch_since_a"] == 0)[0]) - 200) <= 2
    assert column["tch_last"][150] == 1 and column["tch_next"][150] == -1
    assert column["tch_last"][250] == -1 and np.isnan(column["tch_next"][250])
    assert column["tch_flight"][150] == pytest.approx(4.0, abs=0.2)


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


# -- boundary model targets and boundary-aware decoding ------------------------------------------

def test_change_distances_mark_both_frames_at_a_change():
    from vantage.labels import change_classes, change_distances

    labels = codes("H" * 10 + "A" * 10 + "D" * 10 + "H" * 10)
    turn, stop, start = change_distances(labels)
    assert turn[9] == 0 and turn[10] == 0 and turn[8] == 1 and turn[12] == 2
    assert stop[19] == 0 and stop[20] == 0 and stop[9] == 10
    assert start[29] == 0 and start[30] == 0
    classes = change_classes(change_distances(labels), 2)
    assert classes[9] == 1 and classes[20] == 2 and classes[30] == 3 and classes[5] == 0


def test_a_change_out_of_an_ungraded_stretch_is_not_a_restart():
    from vantage.labels import DISTANCE_CAP, change_distances

    labels = codes("X" * 10 + "H" * 10)
    assert (change_distances(labels) == DISTANCE_CAP).all()


def test_boundary_model_lets_a_short_possession_through_and_blocks_flicker():
    # Evidence for a 20-frame away spell that a fixed cost would smooth away.
    prob = np.full((400, 3), 0.1)
    prob[:, H] = 0.8
    prob[200:220] = [0.25, 0.65, 0.1]
    config = decoding.DecodeConfig(cost_team=14.0, min_possession=0.4, boundary_weight=0.0)
    assert (decoding.decode(prob, config) == H).all()

    boundary = np.full((400, 4), 0.001)
    boundary[:, 0] = 0.997
    for frame in (200, 220):
        boundary[frame - 2:frame + 3] = [0.05, 0.93, 0.01, 0.01]      # turnovers seen here
    weighted = decoding.DecodeConfig(cost_team=14.0, min_possession=0.4, boundary_weight=2.0)
    path = decoding.decode(prob, weighted, boundary)
    assert (path[203:217] == A).all() and (path[:195] == H).all() and (path[225:] == H).all()

    # The same evidence without a boundary signal stays smoothed, even with the weight on.
    quiet = np.full((400, 4), 0.001)
    quiet[:, 0] = 0.997
    assert (decoding.decode(prob, weighted, quiet) == H).all()


def test_extra_log_prob_is_added_before_decoding():
    prob = np.full((100, 3), 1.0 / 3.0)
    extra = np.zeros((100, 3))
    extra[:, A] = 0.5
    assert (decoding.decode(prob, decoding.DecodeConfig(), extra_log_prob=extra) == A).all()


# -- run files for the viewer --------------------------------------------------------------------

def test_a_window_run_starts_the_video_where_the_window_starts():
    frames = 200
    ball = np.zeros((frames, 2), dtype=np.float32)
    ball[:, 0] = np.arange(frames)
    ball[50:60] = np.nan
    home = np.full((frames, 2, 2), 10.0, dtype=np.float32)
    away = np.full((frames, 3, 2), -10.0, dtype=np.float32)
    half = Half(match="toy", period=1, pitch=(105.0, 68.0), ball_xy=ball, home_xy=home,
                away_xy=away, first_frame=1000, meta={"flipped": True})
    labels = np.full(1300, X, dtype=np.int8)
    labels[1040:1100] = H
    labels[1100:1150] = D
    run = viewer_export.window("Toy", "Reds", "Blues", labels, [half], 1030, 1160, every=5)
    assert run["video"] == {"offset_frames": 1030}
    assert run["match"]["frames"] == 130
    assert run["labels"] == [["X", 10], ["H", 60], ["D", 50], ["X", 10]]
    assert run["match"]["periods"] == [{"name": "Labelled", "start": 10, "end": 119}]
    samples = run["tracking"]["samples"]
    assert len(samples) == 26
    # run frame 0 is video frame 1030, row 30 of the half, and a turned half is turned back
    assert samples[0]["b"] == [-30.0, 0.0]
    assert samples[0]["h"] == [[-10.0, -10.0]] * 2 and samples[0]["a"] == [[10.0, 10.0]] * 3
    assert samples[4]["b"] is None and samples[5]["b"] is None and samples[6]["b"] == [-60.0, 0.0]
    # nothing in it pretends to come from a model
    assert "spells" not in run and "transitions" not in run
    with pytest.raises(ValueError):
        viewer_export.window("Toy", "Reds", "Blues", labels, [half], 1200, 1400)
