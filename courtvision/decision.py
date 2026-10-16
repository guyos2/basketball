"""Shoot or pass? Compare the expected points of every option at the moment of release.

Expected points of a shot = make probability for the zone x points x contest factor.
A pass is worth the receiver's catch-and-shoot value after defenders close out during the pass,
times the chance the pass is not intercepted. The numbers are rough league-wide averages meant
for comparing options against each other, not for predicting a player's percentage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import court
from .models import Point

# Approximate field-goal percentage by zone (NBA league averages, rounded).
DEFAULT_ZONE_FG = {
    "restricted": 0.62,
    "paint": 0.42,
    "midrange": 0.40,
    "corner_three": 0.385,
    "above_break_three": 0.355,
    "deep_three": 0.30,
}

# Multiplier on the make probability by distance (m) to the nearest defender.
CONTEST_DISTANCES = (0.0, 0.6, 1.2, 1.8, 3.0)
CONTEST_FACTORS = (0.62, 0.75, 0.90, 1.00, 1.08)


def contest_label(defender_distance: float) -> str:
    if defender_distance < 0.6:
        return "צמוד"
    if defender_distance < 1.2:
        return "מכוסה"
    if defender_distance < 1.8:
        return "פנוי"
    return "פנוי לגמרי"


@dataclass
class DecisionConfig:
    zone_fg: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ZONE_FG))
    pass_speed: float = 8.0  # m/s, a crisp chest pass
    catch_time: float = 0.3  # s from catch to release
    closeout_speed: float = 3.0  # m/s a defender covers while the ball travels
    closeout_min_distance: float = 0.9  # a closing-out defender rarely ends up glued to the shooter
    lane_sigma: float = 0.7  # m, how quickly interception risk falls with distance from the passing lane
    lane_max_intercept: float = 0.6  # chance a defender standing right in the lane steals the pass
    margin: float = 0.12  # expected-points difference that is worth a comment


@dataclass
class ShotOption:
    player: int
    xy: Point
    zone: str
    defender_distance: float
    expected_points: float


@dataclass
class PassOption:
    passer: int
    receiver: int
    distance: float
    success: float  # probability the pass is not intercepted
    shot: ShotOption  # receiver's shot after the defense closes out
    expected_points: float


@dataclass
class Decision:
    event_kind: str
    time: float
    player: int
    target: int | None
    shot: ShotOption
    passes: list[PassOption]  # best first
    actual_points: float | None
    best_points: float
    recommended: str  # "shot" or "pass"
    recommended_target: int | None
    verdict: str  # "good", "should_pass", "should_shoot", "better_pass"

    @property
    def regret(self) -> float | None:
        return None if self.actual_points is None else self.best_points - self.actual_points


def shot_value(player: int, xy: Point, defenders: list[Point], cfg: DecisionConfig,
               defender_distance: float | None = None) -> ShotOption:
    if defender_distance is None:
        defender_distance = min((math.dist(xy, d) for d in defenders), default=math.inf)
    z = court.zone(xy)
    points = 3 if court.is_three(xy) else 2
    factor = float(np.interp(min(defender_distance, 10.0), CONTEST_DISTANCES, CONTEST_FACTORS))
    return ShotOption(player, xy, z, defender_distance, cfg.zone_fg[z] * points * factor)


def _distance_to_segment(p: Point, a: Point, b: Point) -> tuple[float, float]:
    """Distance from p to segment ab, and where along the segment (0..1) the closest point is."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.dist(p, (ax + t * dx, ay + t * dy)), t


def pass_value(passer: int, passer_xy: Point, receiver: int, receiver_xy: Point, defenders: list[Point],
               cfg: DecisionConfig) -> PassOption:
    distance = math.dist(passer_xy, receiver_xy)
    success = 1.0
    for d in defenders:
        lane_distance, t = _distance_to_segment(d, passer_xy, receiver_xy)
        if 0.1 < t < 0.95:  # defenders at the passer or on top of the receiver are counted as contest instead
            success *= 1 - cfg.lane_max_intercept * math.exp(-(lane_distance**2) / (2 * cfg.lane_sigma**2))
    travel = distance / cfg.pass_speed + cfg.catch_time
    nearest = min((math.dist(receiver_xy, d) for d in defenders), default=math.inf)
    after_closeout = max(nearest - cfg.closeout_speed * travel, min(nearest, cfg.closeout_min_distance))
    shot = shot_value(receiver, receiver_xy, defenders, cfg, defender_distance=after_closeout)
    return PassOption(passer, receiver, distance, success, shot, success * shot.expected_points)


def evaluate(kind: str, time: float, player: int, target: int | None, positions: dict[int, Point],
             teams: dict[int, int], cfg: DecisionConfig | None = None) -> Decision:
    """Evaluate the decision of `player` at release. `positions` are court coordinates of everyone on the map."""
    cfg = cfg or DecisionConfig()
    team = teams[player]
    handler_xy = positions[player]
    defenders = [xy for t, xy in positions.items() if teams.get(t) is not None and teams[t] != team]
    teammates = {t: xy for t, xy in positions.items() if t != player and teams.get(t) == team}

    shot = shot_value(player, handler_xy, defenders, cfg)
    passes = sorted((pass_value(player, handler_xy, t, xy, defenders, cfg) for t, xy in teammates.items()),
                    key=lambda p: p.expected_points, reverse=True)
    best_pass = passes[0] if passes else None

    if best_pass and best_pass.expected_points > shot.expected_points:
        recommended, recommended_target, best = "pass", best_pass.receiver, best_pass.expected_points
    else:
        recommended, recommended_target, best = "shot", None, shot.expected_points

    if kind == "shot":
        actual = shot.expected_points
        verdict = "should_pass" if best - actual > cfg.margin else "good"
    else:
        chosen = next((p for p in passes if p.receiver == target), None)
        actual = chosen.expected_points if chosen else None
        pass_points = actual if actual is not None else (best_pass.expected_points if best_pass else 0.0)
        if shot.expected_points - pass_points > cfg.margin:
            verdict = "should_shoot"
        elif chosen and best_pass and best_pass.receiver != target and best_pass.expected_points - actual > cfg.margin:
            verdict = "better_pass"
        else:
            verdict = "good"
    return Decision(kind, time, player, target, shot, passes, actual, best, recommended, recommended_target, verdict)
