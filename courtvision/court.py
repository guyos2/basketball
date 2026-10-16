"""FIBA half-court geometry.

Court coordinates are in meters, seen from above:
  x runs along the baseline, 0 = left sideline when you stand at midcourt facing the basket.
  y runs from the baseline (0) to the midcourt line (14).
"""
from __future__ import annotations

import math

WIDTH = 15.0
HALF_LENGTH = 14.0
BASKET = (7.5, 1.575)
THREE_RADIUS = 6.75
CORNER_THREE_X = 0.9  # distance of the straight corner-three line from the sideline
CORNER_THREE_Y = BASKET[1] + math.sqrt(THREE_RADIUS**2 - (BASKET[0] - CORNER_THREE_X) ** 2)
LANE_HALF_WIDTH = 2.45
LANE_LENGTH = 5.8
RESTRICTED_RADIUS = 1.25
CIRCLE_RADIUS = 1.8  # free-throw circle and center circle
DEEP_THREE_RADIUS = 8.5

# Points that can be clicked in the video to calibrate the camera (court coordinates).
LANDMARKS: dict[str, tuple[float, float]] = {
    "baseline_left_corner": (0.0, 0.0),
    "baseline_right_corner": (WIDTH, 0.0),
    "corner3_baseline_left": (CORNER_THREE_X, 0.0),
    "corner3_baseline_right": (WIDTH - CORNER_THREE_X, 0.0),
    "lane_baseline_left": (BASKET[0] - LANE_HALF_WIDTH, 0.0),
    "lane_baseline_right": (BASKET[0] + LANE_HALF_WIDTH, 0.0),
    "lane_ft_left": (BASKET[0] - LANE_HALF_WIDTH, LANE_LENGTH),
    "lane_ft_right": (BASKET[0] + LANE_HALF_WIDTH, LANE_LENGTH),
    "three_point_top": (BASKET[0], BASKET[1] + THREE_RADIUS),
    "midcourt_left": (0.0, HALF_LENGTH),
    "midcourt_right": (WIDTH, HALF_LENGTH),
    "midcourt_center": (BASKET[0], HALF_LENGTH),
}

ZONES = ("restricted", "paint", "midrange", "corner_three", "above_break_three", "deep_three")

ZONE_LABELS_HE = {
    "restricted": "מתחת לסל",
    "paint": "בתוך הצבע",
    "midrange": "חצי מרחק",
    "corner_three": "שלשה מהפינה",
    "above_break_three": "שלשה",
    "deep_three": "שלשה עמוקה",
}


def distance_to_basket(xy: tuple[float, float]) -> float:
    return math.hypot(xy[0] - BASKET[0], xy[1] - BASKET[1])


def is_three(xy: tuple[float, float]) -> bool:
    x, y = xy
    if y <= CORNER_THREE_Y:
        return x < CORNER_THREE_X or x > WIDTH - CORNER_THREE_X
    return distance_to_basket(xy) >= THREE_RADIUS


def zone(xy: tuple[float, float]) -> str:
    x, y = xy
    d = distance_to_basket(xy)
    if is_three(xy):
        if y <= CORNER_THREE_Y:
            return "corner_three"
        return "deep_three" if d >= DEEP_THREE_RADIUS else "above_break_three"
    if d <= RESTRICTED_RADIUS:
        return "restricted"
    if abs(x - BASKET[0]) <= LANE_HALF_WIDTH and y <= LANE_LENGTH:
        return "paint"
    return "midrange"


def in_bounds(xy: tuple[float, float], margin: float = 0.0) -> bool:
    x, y = xy
    return -margin <= x <= WIDTH + margin and -margin <= y <= HALF_LENGTH + margin
