"""Follow a hand-held camera: how each frame maps onto the first frame.

Phone videos pan, tilt and zoom. Between two frames of a camera that turns in place, every
pixel moves by one homography, so matching background features (floor, lines, walls) between
frames gives the motion. The court then only has to be marked on one frame.

To limit drift, each frame is matched directly to an anchor frame instead of only to the
previous one; the anchor moves forward only when the view has changed too much to match.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

WORK_WIDTH = 640  # frames are matched at this width, which is plenty for a homography
MIN_INLIERS = 40
REFERENCE_RETRY = 10  # every N frames try to match the first frame again, which removes drift


@dataclass
class CameraMotion:
    to_first: np.ndarray  # (N, 3, 3): pixel in frame t -> pixel in frame 0
    tracked: np.ndarray  # (N,) bool: False where matching failed and the previous motion was reused

    def __len__(self) -> int:
        return len(self.to_first)

    def between(self, src: int, dst: int) -> np.ndarray:
        """Homography taking pixels of frame `src` to frame `dst`."""
        return np.linalg.inv(self.to_first[dst]) @ self.to_first[src]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, to_first=self.to_first, tracked=self.tracked)

    @classmethod
    def load(cls, path: str | Path) -> "CameraMotion":
        data = np.load(path)
        return cls(data["to_first"], data["tracked"])

    def max_shift(self, size: tuple[int, int]) -> float:
        """Largest distance (pixels) any frame corner moved relative to the first frame."""
        w, h = size
        corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        return max(float(np.abs(cv2.perspectiveTransform(corners, G) - corners).max()) for G in self.to_first)

    def is_static(self, size: tuple[int, int], tolerance_px: float = 2.0) -> bool:
        return self.max_shift(size) <= tolerance_px

    @classmethod
    def static(cls, frames: int) -> "CameraMotion":
        return cls(np.repeat(np.eye(3)[None], frames, axis=0), np.ones(frames, bool))


class _Features:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract(self, frame: np.ndarray, scale: float):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        kp, desc = self.sift.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return pts, desc, gray.shape[::-1]

    def homography(self, a, b) -> np.ndarray | None:
        """Homography from features `a` to features `b` (work resolution), or None."""
        (pa, da, size), (pb, db, _) = a, b
        if da is None or db is None or len(pa) < MIN_INLIERS or len(pb) < MIN_INLIERS:
            return None
        pairs = self.matcher.knnMatch(da, db, k=2)
        good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
        if len(good) < MIN_INLIERS:
            return None
        src = pa[[m.queryIdx for m in good]]
        dst = pb[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask.sum() < MIN_INLIERS or not _plausible(H, size):
            return None
        return H


def _plausible(H: np.ndarray, size: tuple[int, int]) -> bool:
    """Reject degenerate fits: the frame must stay a convex quad of similar area."""
    w, h = size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return False
    ratio = cv2.contourArea(quad) / float(w * h)
    return 0.4 < ratio < 2.5


def estimate_motion(video: str | Path, progress: Callable[[int, int], None] | None = None,
                    on_frame: Callable[[int, np.ndarray], None] | None = None) -> CameraMotion:
    """`on_frame(t, frame)` sees every decoded frame, e.g. to save thumbnails in the same pass."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features = _Features()
    to_first: list[np.ndarray] = []
    tracked: list[bool] = []
    first = anchor = prev = None
    anchor_index = 0
    S = Sinv = None
    t = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t += 1
        if on_frame:
            on_frame(t, frame)
        if S is None:
            scale = min(1.0, WORK_WIDTH / frame.shape[1])
            S = np.diag([scale, scale, 1.0])
            Sinv = np.linalg.inv(S)
        cur = features.extract(frame, S[0, 0])
        if t == 0:
            first = anchor = cur
            to_first.append(np.eye(3))
            tracked.append(True)
        else:
            G = None
            if anchor_index != 0 and t % REFERENCE_RETRY == 0:
                H = features.homography(cur, first)
                if H is not None:
                    G = Sinv @ H @ S
                    anchor, anchor_index = first, 0
            if G is None:
                H = features.homography(cur, anchor)
                if H is not None:
                    G = to_first[anchor_index] @ Sinv @ H @ S
            if G is None:
                H = features.homography(cur, prev)
                if H is not None:
                    G = to_first[t - 1] @ Sinv @ H @ S
                    anchor, anchor_index = cur, t  # the view moved on; match the next frames to this one
            tracked.append(G is not None)
            to_first.append(G / G[2, 2] if G is not None else to_first[t - 1])
        prev = cur
        if progress:
            progress(t + 1, total)
    cap.release()
    if not to_first:
        raise ValueError("The video has no frames")
    return CameraMotion(np.array(to_first), np.array(tracked))
