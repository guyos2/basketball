"""Plain data types shared by the pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class PlayerDetection:
    """One person found by the detector in one video frame."""

    track_id: int
    bbox: BBox
    conf: float = 1.0
    color: np.ndarray | None = None  # median jersey color (Lab), used to split teams

    @property
    def foot(self) -> Point:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass
class FrameDetections:
    """Detector output for one frame, in image pixels."""

    index: int
    players: list[PlayerDetection]
    ball: Point | None = None
    rim: Point | None = None  # only when the model has a rim/hoop class


@dataclass
class Player:
    """A player on the 2D court map."""

    track_id: int
    team: int
    xy: Point  # court meters
    bbox: BBox  # image pixels, for drawing and ball possession


@dataclass
class FrameState:
    index: int
    time: float
    players: dict[int, Player] = field(default_factory=dict)
    ball: Point | None = None  # image pixels
    holder: int | None = None  # track id of the player holding the ball


@dataclass
class Event:
    kind: str  # "shot" or "pass"
    frame: int  # index into the list of frame states, at the moment of release
    time: float
    player: int
    target: int | None = None  # receiver of a pass, when known
    source: str = "auto"  # "auto" (detected from ball) or "manual" (events file)
