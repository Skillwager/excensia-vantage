"""How labels are made from SkillCorner's files, on made-up phases and possessions. No data needed."""

import numpy as np

from vantage import A, D, H, X
from vantage.skillcorner import labels_at_source_rate, restart_team


def codes(text):
    return np.array([{"H": H, "A": A, "D": D, "X": X}[ch] for ch in text], dtype=np.int8)


def test_frames_outside_every_phase_are_dead_ball():
    phases = [(10, 19, H)]
    possessions = [(10, 14, H, "")]
    labels = labels_at_source_rate(0, 29, phases, possessions)
    assert np.array_equal(labels, codes("D" * 10 + "H" * 10 + "D" * 10))


def test_the_ball_stays_with_a_team_until_an_opponent_has_it_under_control():
    phases = [(0, 39, H)]
    possessions = [(0, 5, H, ""), (12, 15, H, ""), (20, 22, A, ""), (30, 33, H, "")]
    labels = labels_at_source_rate(0, 39, phases, possessions)
    assert np.array_equal(labels, codes("H" * 20 + "A" * 10 + "H" * 10))


def test_the_start_of_a_stretch_belongs_to_whoever_took_the_restart():
    # the away team wins the home team's throw: the home team had the ball until then
    phases = [(100, 129, A)]
    possessions = [(108, 112, A, "throw_in_against")]
    labels = labels_at_source_rate(100, 129, phases, possessions)
    assert np.array_equal(labels, codes("H" * 8 + "A" * 22))
    # its own throw: the away team's from the first frame
    labels = labels_at_source_rate(100, 129, phases, [(108, 112, A, "throw_in_for")])
    assert np.array_equal(labels, codes("A" * 30))


def test_after_a_goal_the_team_that_conceded_kicks_off():
    assert restart_team(H, "goal_against") == H
    assert restart_team(H, "goal_for") == A
    assert restart_team(A, "corner_for") == A
    assert restart_team(A, "free_kick_against") == H
    assert restart_team(H, "") == H


def test_possession_does_not_carry_over_a_stoppage():
    phases = [(0, 9, H), (20, 29, A)]
    possessions = [(0, 4, H, ""), (24, 26, A, "goal_kick_for")]
    labels = labels_at_source_rate(0, 29, phases, possessions)
    assert np.array_equal(labels, codes("H" * 10 + "D" * 10 + "A" * 10))


def test_a_stretch_in_which_no_possession_is_named_is_not_graded():
    labels = labels_at_source_rate(0, 19, [(5, 14, H)], [])
    assert np.array_equal(labels, codes("D" * 5 + "X" * 10 + "D" * 5))


def test_phases_that_touch_make_one_stretch_of_play():
    phases = [(0, 10, H), (10, 20, A)]
    possessions = [(0, 3, H, ""), (10, 12, A, "")]
    labels = labels_at_source_rate(0, 20, phases, possessions)
    assert np.array_equal(labels, codes("H" * 10 + "A" * 11))
