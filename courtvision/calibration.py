"""Camera calibration: maps the floor in the video onto the 2D court map.

The camera must not move during the clip, so one homography covers every frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import court
from .models import Point


@dataclass
class Calibration:
    points: dict[str, Point]  # landmark name -> pixel
    rim_px: Point | None = None  # center of the rim in the image
    rim_radius_px: float = 30.0  # how close the ball must get to the rim to count as a shot
    H: np.ndarray = field(init=False, repr=False)  # image -> court
    reprojection_error: float = field(init=False)  # mean error in meters on the clicked points

    def __post_init__(self):
        unknown = set(self.points) - set(court.LANDMARKS)
        if unknown:
            raise ValueError(f"Unknown landmarks: {sorted(unknown)}. Valid: {sorted(court.LANDMARKS)}")
        if len(self.points) < 4:
            raise ValueError("At least 4 landmarks are needed (no three of them on one line).")
        names = list(self.points)
        src = np.array([self.points[n] for n in names], dtype=np.float64)
        dst = np.array([court.LANDMARKS[n] for n in names], dtype=np.float64)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(names) > 4 else 0, 0.5)
        if H is None:
            raise ValueError("Could not compute a homography; check that the points are not collinear.")
        self.H = H
        self.reprojection_error = float(np.linalg.norm(self.to_court(src) - dst, axis=1).mean())

    def to_court(self, pixels) -> np.ndarray:
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, court_points) -> np.ndarray:
        pts = np.asarray(court_points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(
            points={k: tuple(v) for k, v in data["points"].items()},
            rim_px=tuple(data["rim"]) if data.get("rim") else None,
            rim_radius_px=float(data.get("rim_radius_px", 30.0)),
        )

    def save(self, path: str | Path) -> None:
        data = {"points": {k: list(v) for k, v in self.points.items()}, "rim_radius_px": self.rim_radius_px}
        if self.rim_px:
            data["rim"] = list(self.rim_px)
        Path(path).write_text(json.dumps(data, indent=2))


def interactive_calibrate(frame: np.ndarray) -> Calibration:
    """Click landmarks on a frame in an OpenCV window (needs a desktop, not headless OpenCV).

    For each landmark: left-click to place it, or press any key to skip. Finally click the rim center.
    """
    window = "calibration"
    clicked: list[Point] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    canvas = frame.copy()

    def ask(prompt: str) -> Point | None:
        clicked.clear()
        while True:
            view = canvas.copy()
            cv2.putText(view, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(view, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(20)
            if clicked:
                cv2.circle(canvas, tuple(map(int, clicked[0])), 6, (0, 255, 255), -1)
                return clicked[0]
            if key != -1:
                return None

    points = {}
    for name in court.LANDMARKS:
        p = ask(f"Click {name} (any key = skip)")
        if p:
            points[name] = p
    rim = ask("Click the RIM center (any key = skip)")
    cv2.destroyWindow(window)
    return Calibration(points=points, rim_px=rim)
