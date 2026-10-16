"""A synthetic half-court possession seen through a virtual camera.

It produces the same detector output a real video would, so the whole pipeline after YOLO
(calibration, 2D map, teams, possession, shoot-or-pass) can be tried without a video.

The play: #1 swings the ball to #2 on the right wing. #2 drives to the right elbow, #5's defender
leaves the corner to help, and #2 takes a contested mid-range shot while #5 is open in the corner.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import court
from .calibration import Calibration
from .models import FrameDetections, PlayerDetection

FPS = 25.0
DURATION = 5.6

# A camera high above midcourt looking at the basket (1280x720).
CAMERA = Calibration(
    points={
        "baseline_left_corner": (330.0, 170.0),
        "baseline_right_corner": (950.0, 170.0),
        "midcourt_left": (40.0, 700.0),
        "midcourt_right": (1240.0, 700.0),
    },
    rim_radius_px=25.0,
)

WHITE, RED = np.array([235.0, 128.0, 128.0]), np.array([120.0, 185.0, 165.0])  # Lab

# Keyframes: track id -> [(time, (x, y)), ...], positions in court meters.
PATHS = {
    # Offense (white)
    1: [(0, (7.5, 9.0))],
    2: [(0, (13.5, 5.6)), (1.7, (13.5, 5.6)), (3.8, (10.4, 5.4))],
    3: [(0, (0.6, 1.0))],
    4: [(0, (5.3, 2.5))],
    5: [(0, (14.4, 1.0))],
    # Defense (red)
    11: [(0, (7.5, 8.4)), (1.5, (7.8, 8.0))],
    12: [(0, (12.8, 5.0)), (1.7, (12.8, 5.0)), (3.8, (10.0, 4.9))],
    13: [(0, (1.3, 1.5))],
    14: [(0, (5.8, 2.2))],
    15: [(0, (13.5, 1.6)), (2.6, (13.5, 1.6)), (3.8, (11.0, 2.6))],
}
TEAM_COLOR = {t: (WHITE if t < 10 else RED) for t in PATHS}

# Ball: who holds it when, and the flights between holders.
HOLDS = [(1, 0.0, 1.3), (2, 1.7, 4.0)]
PASS_FLIGHT = (1, 1.3, 2, 1.7)
SHOT_FLIGHT = (2, 4.0, 4.9)


def position(track: int, t: float) -> tuple[float, float]:
    keys = PATHS[track]
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t <= t1:
            a = (t - t0) / (t1 - t0)
            return (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]))
    return keys[-1][1]


def _scale(xy) -> float:
    """Pixels per meter around a court point."""
    a, b = CAMERA.to_image([(xy[0] - 0.5, xy[1]), (xy[0] + 0.5, xy[1])])
    return float(np.linalg.norm(b - a))


def _bbox(xy):
    fx, fy = CAMERA.to_image([xy])[0]
    h = 1.95 * _scale(xy)
    w = 0.45 * h
    return (fx - w / 2, fy - h, fx + w / 2, fy)


def _hands(track: int, t: float):
    x1, y1, x2, y2 = _bbox(position(track, t))
    return ((x1 + x2) / 2, y1 + 0.45 * (y2 - y1))


def rim_pixel():
    bx, by = CAMERA.to_image([court.BASKET])[0]
    return (float(bx), float(by - 3.05 * _scale(court.BASKET)))


def _ball(t: float):
    for holder, t0, t1 in HOLDS:
        if t0 <= t < t1:
            return _hands(holder, t)
    passer, t0, receiver, t1 = PASS_FLIGHT
    if t0 <= t < t1:
        return _arc(_hands(passer, t0), _hands(receiver, t1), (t - t0) / (t1 - t0), 20)
    shooter, t0, t1 = SHOT_FLIGHT
    if t0 <= t <= t1:
        return _arc(_hands(shooter, t0), rim_pixel(), (t - t0) / (t1 - t0), 120)
    return None


def _arc(a, b, s: float, lift: float):
    return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]) - lift * 4 * s * (1 - s))


def camera_shake(t: float) -> np.ndarray:
    """Hand-held camera: homography from the first frame's pixels to frame t's pixels (pan, tilt, zoom, roll)."""
    w = 2 * np.pi * t / DURATION
    scale = 1 + 0.10 * np.sin(w)
    angle = np.radians(2.0 * np.sin(1.3 * w))
    tx, ty = 140 * np.sin(w), 35 * np.sin(2 * w)
    c, s = scale * np.cos(angle), scale * np.sin(angle)
    cx, cy = 640, 360
    return np.array([[c, -s, cx - c * cx + s * cy + tx], [s, c, cy - s * cx - c * cy + ty], [0, 0, 1]])


def _warp(points, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.float64(points).reshape(-1, 1, 2), H).reshape(-1, 2)


def detections(seed: int = 0, moving: bool = False) -> list[FrameDetections]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(DURATION * FPS)):
        t = i / FPS
        view = camera_shake(t) if moving else np.eye(3)
        players = []
        for track in PATHS:
            x1, y1, x2, y2 = _bbox(position(track, t))
            corners = _warp([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], view)
            box = (*corners.min(axis=0), *corners.max(axis=0))
            jitter = rng.normal(0, 1.5, 4)  # detector box noise in pixels
            bbox = tuple(float(v) for v in np.add(box, jitter))
            color = TEAM_COLOR[track] + rng.normal(0, 6, 3)
            players.append(PlayerDetection(track, bbox, 0.9, color))
        ball = _ball(t)
        if ball is not None:
            ball = tuple(float(v) for v in _warp([ball], view)[0])
        out.append(FrameDetections(i, players, ball=ball))
    return out


def calibration() -> Calibration:
    return Calibration(points=dict(CAMERA.points), rim_px=rim_pixel(), rim_radius_px=CAMERA.rim_radius_px)


def render_video(path: str | Path, moving: bool = False, seed: int = 0) -> Path:
    """Draw the demo play as a video: a textured gym (so camera motion can be tracked) with box players."""
    from .render import draw_court_lines

    rng = np.random.default_rng(seed)
    margin = 400  # the world image extends past the first frame so a panning camera still sees texture
    world_size = (1280 + 2 * margin, 720 + 2 * margin)
    noise = rng.normal(0, 1, (world_size[1] // 8, world_size[0] // 8, 3))
    world = cv2.resize(noise, world_size, interpolation=cv2.INTER_CUBIC)
    world = np.clip(70 + 35 * world, 0, 255).astype(np.uint8)
    shift = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], float)  # first-frame pixel -> world pixel
    floor = np.round(_warp(CAMERA.to_image([[-1, -1], [16, -1], [16, 15], [-1, 15]]), shift)).astype(np.int32)
    wood = world.copy()
    cv2.fillPoly(wood, [floor], (120, 175, 215))
    world = cv2.addWeighted(world, 0.25, wood, 0.75, 0)
    draw_court_lines(world, lambda pts: _warp(CAMERA.to_image(pts), shift), (255, 255, 255), 3)

    dets = detections(seed, moving)
    path = Path(path)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for i, fd in enumerate(dets):
        view = camera_shake(i / FPS) if moving else np.eye(3)
        frame = cv2.warpPerspective(world, view @ np.linalg.inv(shift), (1280, 720), borderMode=cv2.BORDER_REFLECT)
        for p in fd.players:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240) if p.track_id < 10 else (40, 40, 200), -1)
        if fd.ball:
            cv2.circle(frame, (int(fd.ball[0]), int(fd.ball[1])), 7, (0, 110, 230), -1)
        writer.write(frame)
    writer.release()
    return path
