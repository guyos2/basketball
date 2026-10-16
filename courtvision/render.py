"""Draw the 2D court map, players and decision overlays."""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import court
from .decision import Decision
from .models import FrameState, Point

DEFAULT_TEAM_COLORS = {0: (220, 120, 30), 1: (40, 70, 225)}  # BGR: blue, red
OTHER_COLOR = (150, 150, 150)
FLOOR = (130, 190, 225)
LINE = (255, 255, 255)
GOOD = (60, 200, 60)
BAD = (40, 40, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class CourtCanvas:
    def __init__(self, team_colors: dict[int, tuple] | None = None, scale: float = 40.0, margin: int = 30):
        self.team_colors = team_colors or DEFAULT_TEAM_COLORS
        self.scale, self.margin = scale, margin
        self.width = int(court.WIDTH * scale) + 2 * margin
        self.height = int(court.HALF_LENGTH * scale) + 2 * margin
        self._base = self._draw_base()

    def px(self, xy: Point) -> tuple[int, int]:
        return int(round(self.margin + xy[0] * self.scale)), int(round(self.margin + xy[1] * self.scale))

    def _r(self, meters: float) -> int:
        return int(round(meters * self.scale))

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), FLOOR, np.uint8)
        t = 2
        cv2.rectangle(img, self.px((0, 0)), self.px((court.WIDTH, court.HALF_LENGTH)), LINE, t)
        bx, by = court.BASKET
        lane = (self.px((bx - court.LANE_HALF_WIDTH, 0)), self.px((bx + court.LANE_HALF_WIDTH, court.LANE_LENGTH)))
        cv2.rectangle(img, *lane, (110, 160, 200), -1)
        cv2.rectangle(img, *lane, LINE, t)
        cv2.ellipse(img, self.px((bx, court.LANE_LENGTH)), (self._r(court.CIRCLE_RADIUS),) * 2, 0, 0, 180, LINE, t)
        cv2.ellipse(img, self.px((bx, court.HALF_LENGTH)), (self._r(court.CIRCLE_RADIUS),) * 2, 0, 180, 360, LINE, t)
        cv2.ellipse(img, self.px(court.BASKET), (self._r(court.RESTRICTED_RADIUS),) * 2, 0, 0, 180, LINE, t)
        # Three-point line: straight corners, then the arc.
        for x in (court.CORNER_THREE_X, court.WIDTH - court.CORNER_THREE_X):
            cv2.line(img, self.px((x, 0)), self.px((x, court.CORNER_THREE_Y)), LINE, t)
        start = math.degrees(math.atan2(court.CORNER_THREE_Y - by, court.WIDTH - court.CORNER_THREE_X - bx))
        cv2.ellipse(img, self.px(court.BASKET), (self._r(court.THREE_RADIUS),) * 2, 0, start, 180 - start, LINE, t)
        # Backboard and rim.
        cv2.line(img, self.px((bx - 0.9, 1.2)), self.px((bx + 0.9, 1.2)), (60, 60, 60), 3)
        cv2.circle(img, self.px(court.BASKET), self._r(0.23), (0, 90, 240), 2)
        return img

    def draw_state(self, state: FrameState, teams: dict[int, int]) -> np.ndarray:
        img = self._base.copy()
        for p in state.players.values():
            center = self.px(p.xy)
            color = self.team_colors.get(teams.get(p.track_id), OTHER_COLOR)
            cv2.circle(img, center, 11, color, -1)
            cv2.circle(img, center, 11, (30, 30, 30), 2)
            if p.track_id == state.holder:
                cv2.circle(img, center, 16, (0, 165, 255), 3)
            label = str(p.track_id)
            text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            (w, h), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            cv2.putText(img, label, (center[0] - w // 2, center[1] + h // 2), FONT, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(img, f"t={state.time:5.1f}s", (self.margin, self.height - 8), FONT, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        return img

    def draw_decision(self, state: FrameState, teams: dict[int, int], d: Decision) -> np.ndarray:
        img = self.draw_state(state, teams)
        src = self.px(d.shot.xy)
        actual_ok = d.verdict == "good"
        if d.event_kind == "shot":
            cv2.arrowedLine(img, src, self.px(court.BASKET), GOOD if actual_ok else BAD, 3, cv2.LINE_AA, tipLength=0.08)
        elif d.target is not None and d.target in state.players:
            cv2.arrowedLine(img, src, self.px(state.players[d.target].xy), GOOD if actual_ok else BAD, 3,
                            cv2.LINE_AA, tipLength=0.08)
        if not actual_ok:
            if d.recommended == "pass" and d.recommended_target in state.players:
                _dashed_arrow(img, src, self.px(state.players[d.recommended_target].xy), GOOD)
            elif d.recommended == "shot":
                _dashed_arrow(img, src, self.px(court.BASKET), GOOD)
        # Expected points of each option next to the player who would shoot.
        _tag(img, src, f"shot {d.shot.expected_points:.2f}", (255, 255, 255))
        for p in d.passes:
            if p.receiver in state.players:
                _tag(img, self.px(state.players[p.receiver].xy), f"{p.expected_points:.2f}", (255, 255, 255))
        title = {"good": "GOOD DECISION", "should_pass": f"SHOULD PASS TO #{d.recommended_target}",
                 "should_shoot": "SHOULD SHOOT", "better_pass": f"BETTER PASS: #{d.recommended_target}"}[d.verdict]
        cv2.rectangle(img, (0, 0), (self.width, 26), (40, 40, 40), -1)
        cv2.putText(img, f"{d.event_kind.upper()} #{d.player} @ {d.time:.1f}s - {title}", (8, 18), FONT, 0.55,
                    GOOD if actual_ok else (80, 80, 255), 1, cv2.LINE_AA)
        return img


def _tag(img: np.ndarray, at: tuple[int, int], text: str, color) -> None:
    org = (at[0] + 14, at[1] - 12)
    (w, h), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + w + 2, org[1] + 3), (30, 30, 30), -1)
    cv2.putText(img, text, org, FONT, 0.45, color, 1, cv2.LINE_AA)


def _dashed_arrow(img: np.ndarray, a: tuple[int, int], b: tuple[int, int], color, dash: int = 12) -> None:
    length = math.dist(a, b)
    steps = max(int(length // dash), 1)
    for i in range(0, steps - 1, 2):
        p = (int(a[0] + (b[0] - a[0]) * i / steps), int(a[1] + (b[1] - a[1]) * i / steps))
        q = (int(a[0] + (b[0] - a[0]) * (i + 1) / steps), int(a[1] + (b[1] - a[1]) * (i + 1) / steps))
        cv2.line(img, p, q, color, 3, cv2.LINE_AA)
    tail = (int(a[0] + (b[0] - a[0]) * (steps - 1) / steps), int(a[1] + (b[1] - a[1]) * (steps - 1) / steps))
    cv2.arrowedLine(img, tail, b, color, 3, cv2.LINE_AA, tipLength=0.6)


def annotate_frame(frame: np.ndarray, state: FrameState, teams: dict[int, int],
                   team_colors: dict[int, tuple] | None = None) -> np.ndarray:
    """Boxes, ids and the ball on the original video frame."""
    team_colors = team_colors or DEFAULT_TEAM_COLORS
    img = frame.copy()
    for p in state.players.values():
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        color = team_colors.get(teams.get(p.track_id), OTHER_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"#{p.track_id}", (x1, y1 - 5), FONT, 0.6, color, 2, cv2.LINE_AA)
    if state.ball is not None:
        cv2.circle(img, (int(state.ball[0]), int(state.ball[1])), 8, (0, 165, 255), 2)
    return img


def side_by_side(frame: np.ndarray, minimap: np.ndarray) -> np.ndarray:
    h = minimap.shape[0]
    scaled = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
    return np.hstack([scaled, minimap])


def _court_lines() -> list[np.ndarray]:
    """The court markings as polylines in court meters."""
    def arc(center, radius, a0, a1, n=40):
        a = np.radians(np.linspace(a0, a1, n))
        return np.stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)], axis=1)

    bx, by = court.BASKET
    w, half = court.WIDTH, court.HALF_LENGTH
    l, r = bx - court.LANE_HALF_WIDTH, bx + court.LANE_HALF_WIDTH
    start = math.degrees(math.atan2(court.CORNER_THREE_Y - by, w - court.CORNER_THREE_X - bx))
    three = np.vstack([[[w - court.CORNER_THREE_X, 0]], arc(court.BASKET, court.THREE_RADIUS, start, 180 - start),
                       [[court.CORNER_THREE_X, 0]]])
    return [
        np.array([[0, 0], [w, 0], [w, half], [0, half], [0, 0]], float),
        np.array([[l, 0], [l, court.LANE_LENGTH], [r, court.LANE_LENGTH], [r, 0]], float),
        arc((bx, court.LANE_LENGTH), court.CIRCLE_RADIUS, 0, 180),
        arc((bx, half), court.CIRCLE_RADIUS, 180, 360),
        three,
    ]


def draw_court_lines(img: np.ndarray, to_image, color=(0, 255, 255), thickness: int = 2) -> None:
    """Draw the court markings on a video frame, given a court -> pixel function."""
    for line in _court_lines():
        dense = np.vstack([np.linspace(a, b, 12) for a, b in zip(line[:-1], line[1:])])
        pts = to_image(dense)
        if np.all(np.isfinite(pts)):
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_court_overlay(frame: np.ndarray, calib) -> np.ndarray:
    """Project the court markings onto the video frame, to check a calibration by eye."""
    img = frame.copy()
    draw_court_lines(img, calib.to_image)
    for name, px in calib.points.items():
        cv2.circle(img, (int(px[0]), int(px[1])), 6, (0, 0, 255), -1)
    if calib.rim_px:
        cv2.circle(img, (int(calib.rim_px[0]), int(calib.rim_px[1])), int(calib.rim_radius_px), (0, 140, 255), 2)
    return img
