import pytest

from courtvision.decision import DecisionConfig, evaluate, pass_value, shot_value

OFFENSE, DEFENSE = 0, 1


def _teams(positions):
    return {t: (OFFENSE if t < 10 else DEFENSE) for t in positions}


def test_contest_lowers_shot_value():
    cfg = DecisionConfig()
    open_shot = shot_value(1, (0.5, 1.0), [(3.5, 1.0)], cfg)
    contested = shot_value(1, (0.5, 1.0), [(0.9, 1.0)], cfg)
    assert open_shot.zone == "corner_three"
    assert open_shot.expected_points > contested.expected_points


def test_defender_in_lane_lowers_pass_success():
    cfg = DecisionConfig()
    clear = pass_value(1, (5.0, 6.0), 2, (12.0, 6.0), [(8.5, 9.0)], cfg)
    blocked = pass_value(1, (5.0, 6.0), 2, (12.0, 6.0), [(8.5, 6.1)], cfg)
    assert clear.success > 0.95
    assert blocked.success < 0.5


def test_closeout_during_long_pass():
    cfg = DecisionConfig()
    short = pass_value(1, (12.0, 4.0), 2, (14.4, 1.0), [(11.0, 1.0)], cfg)
    long = pass_value(1, (2.0, 8.0), 2, (14.4, 1.0), [(11.0, 1.0)], cfg)
    assert short.shot.defender_distance > long.shot.defender_distance


def test_contested_midrange_with_open_corner_should_pass():
    positions = {2: (10.4, 5.4), 5: (14.4, 1.0), 12: (10.0, 4.9), 15: (11.0, 2.6)}
    d = evaluate("shot", 4.0, 2, None, positions, _teams(positions))
    assert d.verdict == "should_pass"
    assert d.recommended_target == 5
    assert d.regret == pytest.approx(d.best_points - d.shot.expected_points)


def test_open_layup_is_a_good_shot():
    positions = {1: (7.5, 2.2), 2: (14.4, 1.0), 11: (9.5, 3.5), 12: (13.6, 1.4)}
    d = evaluate("shot", 1.0, 1, None, positions, _teams(positions))
    assert d.shot.zone == "restricted"
    assert d.verdict == "good"


def test_passing_up_an_open_layup_should_shoot():
    positions = {1: (7.5, 2.2), 2: (7.5, 9.0), 11: (9.5, 3.5), 12: (7.5, 8.2)}
    d = evaluate("pass", 1.0, 1, 2, positions, _teams(positions))
    assert d.verdict == "should_shoot"


def test_pass_to_covered_player_when_another_is_open():
    positions = {1: (7.5, 9.0), 2: (13.5, 5.6), 3: (1.0, 5.6), 11: (7.5, 8.3), 12: (13.0, 5.4), 13: (6.0, 3.0)}
    d = evaluate("pass", 1.0, 1, 2, positions, _teams(positions))
    assert d.verdict == "better_pass"
    assert d.recommended_target == 3
