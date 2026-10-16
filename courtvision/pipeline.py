"""Video -> 2D court map -> shoot-or-pass decisions -> report."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from . import court
from .calibration import Calibration
from .decision import Decision, DecisionConfig, evaluate
from .events import EventConfig, detect_events, load_manual_events
from .feedback import decision_text, report_markdown
from .models import Event, FrameDetections, FrameState, Player
from .render import CourtCanvas, annotate_frame, side_by_side
from .teams import assign_teams, team_display_colors

COURT_MARGIN_M = 1.0  # people standing further outside the half court are spectators or benches
MIN_TRACK_FRAMES = 8
SMOOTH_WINDOW = 5


@dataclass
class Analysis:
    fps: float
    states: list[FrameState]
    teams: dict[int, int]
    events: list[Event]
    decisions: list[Decision]
    decision_frames: list[int]  # state index of each decision
    team_colors: dict[int, tuple] = field(default_factory=dict)  # BGR jersey color per team


def build_states(detections: list[FrameDetections], calib: Calibration, fps: float
                 ) -> tuple[list[FrameState], dict[int, int], dict[int, tuple]]:
    """Project every detected player onto the court, smooth their paths and split them into teams."""
    paths: dict[int, dict[int, tuple]] = defaultdict(dict)  # track -> frame -> (xy, bbox)
    colors: dict[int, list[np.ndarray]] = defaultdict(list)
    for k, fd in enumerate(detections):
        if not fd.players:
            continue
        feet = calib.to_court([p.foot for p in fd.players])
        for p, xy in zip(fd.players, feet):
            if court.in_bounds(xy, COURT_MARGIN_M):
                paths[p.track_id][k] = (tuple(xy), p.bbox)
                if p.color is not None:
                    colors[p.track_id].append(p.color)
    paths = {t: f for t, f in paths.items() if len(f) >= MIN_TRACK_FRAMES}
    colors = {t: colors[t] for t in paths}
    teams = assign_teams(colors)

    states = [FrameState(index=fd.index, time=fd.index / fps, ball=fd.ball) for fd in detections]
    half = SMOOTH_WINDOW // 2
    for t, frames in paths.items():
        for k, (_, bbox) in frames.items():
            near = [frames[j][0] for j in range(k - half, k + half + 1) if j in frames]
            xy = tuple(float(v) for v in np.mean(near, axis=0))
            states[k].players[t] = Player(t, teams.get(t, -1), xy, bbox)
    return states, teams, team_display_colors(colors, teams)


def rim_position(calib: Calibration, detections: list[FrameDetections]):
    if calib.rim_px is not None:
        return calib.rim_px
    found = [fd.rim for fd in detections if fd.rim is not None]
    return tuple(np.median(found, axis=0)) if found else None


def analyze_detections(detections: list[FrameDetections], calib: Calibration, fps: float,
                       manual_events: str | Path | list[dict] | None = None, decision_cfg: DecisionConfig | None = None,
                       event_cfg: EventConfig | None = None) -> Analysis:
    states, teams, team_colors = build_states(detections, calib, fps)
    events = detect_events(states, teams, rim_position(calib, detections), calib.rim_radius_px, fps, event_cfg)
    if manual_events:
        events = load_manual_events(manual_events, states)
    decisions, frames = [], []
    for e in events:
        positions = {t: p.xy for t, p in states[e.frame].players.items()}
        if e.player not in positions or e.player not in teams:
            continue
        decisions.append(evaluate(e.kind, e.time, e.player, e.target, positions, teams, decision_cfg))
        frames.append(e.frame)
    return Analysis(fps, states, teams, events, decisions, frames, team_colors)


def detect_video(video: str | Path, model: str = "yolo11n.pt", stride: int = 1, max_frames: int | None = None,
                 progress: Callable[[int, int], None] | None = None) -> tuple[list[FrameDetections], float]:
    from .detection import Detector

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / stride
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = Detector(model, fps=fps)
    detections = []
    for i, frame in enumerate(_frames(cap)):
        if max_frames is not None and len(detections) >= max_frames:
            break
        if i % stride:
            continue
        detections.append(detector(frame, len(detections)))
        if progress:
            progress(i + 1, total)
    cap.release()
    return detections, fps


def _frames(cap: cv2.VideoCapture) -> Iterable[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def open_video_writer(stem: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Pick a codec that browsers can play: H.264 where OpenCV has it (macOS), otherwise VP8 WebM."""
    for fourcc, ext in (("avc1", ".mp4"), ("VP80", ".webm"), ("mp4v", ".mp4")):
        path = stem.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("OpenCV cannot write video on this machine")


def write_outputs(analysis: Analysis, out_dir: str | Path, video: str | Path | None = None, stride: int = 1,
                  progress: Callable[[int, int], None] | None = None) -> Path:
    """Write report.md, report.json, tracks.csv, a snapshot per decision and the 2D video."""
    out = Path(out_dir)
    (out / "decisions").mkdir(parents=True, exist_ok=True)
    for old in (out / "decisions").glob("*.png"):
        old.unlink()
    canvas = CourtCanvas(analysis.team_colors)
    teams = analysis.teams

    duration = analysis.states[-1].time if analysis.states else 0.0
    (out / "report.md").write_text(report_markdown(analysis.decisions, duration), encoding="utf-8")
    (out / "tracks.csv").write_text("frame,time,track_id,team,x_m,y_m,has_ball\n" + "".join(
        f"{s.index},{s.time:.3f},{p.track_id},{p.team},{p.xy[0]:.2f},{p.xy[1]:.2f},{int(p.track_id == s.holder)}\n"
        for s in analysis.states for p in s.players.values()
    ))

    decision_frames, images = {}, []
    for n, (k, d) in enumerate(zip(analysis.decision_frames, analysis.decisions), 1):
        img = canvas.draw_decision(analysis.states[k], teams, d)
        name = f"decisions/{n:02d}_{d.event_kind}_{d.time:05.1f}s.png"
        cv2.imwrite(str(out / name), img)
        images.append(name)
        decision_frames[k] = img

    cap = cv2.VideoCapture(str(video)) if video else None
    frames = (f for i, f in enumerate(_frames(cap)) if i % stride == 0) if cap else None
    writer, video_path = None, None
    source_writer, source_path = None, None  # the original frames with player ids, same timeline as the analysis
    hold = int(analysis.fps * 1.5)  # freeze on each decision for 1.5 s
    for k, state in enumerate(analysis.states):
        minimap = canvas.draw_state(state, teams)
        frame = next(frames, None) if frames else None
        annotated = annotate_frame(frame, state, teams, analysis.team_colors) if frame is not None else None
        img = side_by_side(annotated, minimap) if annotated is not None else minimap
        if annotated is not None:
            if source_writer is None:
                source_writer, source_path = open_video_writer(out / "source", analysis.fps,
                                                               (annotated.shape[1], annotated.shape[0]))
            source_writer.write(annotated)
        if writer is None:
            writer, video_path = open_video_writer(out / "court2d", analysis.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
        if k in decision_frames:
            still = decision_frames[k]
            still = side_by_side(annotated, still) if annotated is not None else still
            for _ in range(hold):
                writer.write(still)
        if progress:
            progress(k + 1, len(analysis.states))
    for w in (writer, source_writer):
        if w:
            w.release()
    if cap:
        cap.release()

    (out / "report.json").write_text(json.dumps({
        "fps": analysis.fps,
        "duration": duration,
        "video": video_path.name if video_path else None,
        "source_video": source_path.name if source_path else None,
        "teams": {str(k): v for k, v in teams.items()},
        "players": sorted(teams),
        "events": [asdict(e) for e in analysis.events],
        "decisions": [dict(asdict(d), text=decision_text(d), image=img) for d, img in zip(analysis.decisions, images)],
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return out
