"""Find players and the ball in each video frame (YOLO + ByteTrack)."""
from __future__ import annotations

import cv2
import numpy as np

from .models import FrameDetections, PlayerDetection

PLAYER_NAMES = {"person", "player"}
RIM_NAMES = {"rim", "hoop", "basket"}


def jersey_color(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Median Lab color of the torso: the upper-middle part of the box, away from background and legs."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    w, h = x2 - x1, y2 - y1
    crop = frame[max(y1 + int(0.15 * h), 0): max(y1 + int(0.5 * h), 0), max(x1 + int(0.25 * w), 0): max(x2 - int(0.25 * w), 0)]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    return np.median(lab, axis=0)


class Detector:
    """Wraps a YOLO model. Works with the stock COCO model (person + sports ball) or a basketball
    model whose class names contain 'player', 'ball' and 'rim'/'hoop'."""

    def __init__(self, model_path: str = "yolo11n.pt", fps: float = 25.0, player_conf: float = 0.3,
                 ball_conf: float = 0.1, device: str | None = None, imgsz: int = 1280):
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        self.model = YOLO(model_path)
        names = self.model.names
        self.player_cls = {i for i, n in names.items() if n.lower() in PLAYER_NAMES}
        self.ball_cls = {i for i, n in names.items() if "ball" in n.lower()}
        self.rim_cls = {i for i, n in names.items() if n.lower() in RIM_NAMES}
        if not self.player_cls or not self.ball_cls:
            raise ValueError(f"Model {model_path} needs player/person and ball classes, has {list(names.values())}")
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        cfg.track_buffer = int(round(fps))  # keep a lost player alive for ~1 second
        self.tracker = BYTETracker(cfg)
        self.player_conf, self.ball_conf = player_conf, ball_conf
        self.device, self.imgsz = device, imgsz

    def __call__(self, frame: np.ndarray, index: int) -> FrameDetections:
        result = self.model.predict(frame, conf=min(self.ball_conf, 0.1), imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)
        conf = boxes.conf

        ball = rim = None
        ball_idx = [i for i in range(len(cls)) if cls[i] in self.ball_cls and conf[i] >= self.ball_conf]
        if ball_idx:
            x1, y1, x2, y2 = boxes.xyxy[max(ball_idx, key=lambda i: conf[i])]
            ball = ((x1 + x2) / 2, (y1 + y2) / 2)
        rim_idx = [i for i in range(len(cls)) if cls[i] in self.rim_cls and conf[i] >= 0.3]
        if rim_idx:
            x1, y1, x2, y2 = boxes.xyxy[max(rim_idx, key=lambda i: conf[i])]
            rim = ((x1 + x2) / 2, (y1 + y2) / 2)

        person_mask = np.isin(cls, list(self.player_cls))
        tracks = self.tracker.update(boxes[person_mask], frame)
        players = []
        for x1, y1, x2, y2, track_id, score, *_ in tracks:
            if score < self.player_conf:
                continue
            bbox = (float(x1), float(y1), float(x2), float(y2))
            players.append(PlayerDetection(int(track_id), bbox, float(score), jersey_color(frame, bbox)))
        return FrameDetections(index=index, players=players, ball=ball, rim=rim)
