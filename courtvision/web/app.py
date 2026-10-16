"""Web interface: upload a video, mark the court, run the analysis, read the results.

Run locally with  python -m courtvision serve  and open http://localhost:8000.
Environment variables (useful when it runs in the cloud):
  COURTVISION_PASSWORD       ask for this password before showing anything
  COURTVISION_MAX_UPLOAD_MB  largest accepted video (default 500)
  COURTVISION_KEEP_HOURS     delete uploaded videos and results older than this (default 48)
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import secrets
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import court, demo
from ..calibration import Calibration, CameraTrack
from ..camera_motion import CameraMotion, estimate_motion
from ..pipeline import analyze_detections, detect_video, write_outputs
from ..render import draw_court_lines

STATIC = Path(__file__).parent / "static"
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
THUMB_STEP = 5  # frames that can be picked for calibration: every 5th
THUMB_WIDTH = 1280  # stored frames are scaled down to this width; points are always in video pixels


@dataclass
class Job:
    id: str
    dir: Path
    video: Path | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    state: str = "uploaded"  # uploaded, queued, running, done, error
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    motion: CameraMotion | None = None
    motion_state: str = "pending"  # pending, running, done, error
    motion_progress: float = 0.0
    motion_error: str | None = None
    camera_moves: bool = False
    demo: bool = False
    motion_done: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def out(self) -> Path:
        return self.dir / "output"

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def thumb(self, frame: int) -> Path:
        return self.dir / "frames" / f"{frame:06d}.jpg"

    def public(self) -> dict:
        data = {
            "id": self.id, "state": self.state, "stage": self.stage, "progress": round(self.progress, 3),
            "error": self.error, "width": self.width, "height": self.height, "fps": self.fps,
            "frames": self.frames, "duration": self.duration, "thumb_step": THUMB_STEP,
            "has_video": self.video is not None, "demo": self.demo,
            "video_url": f"/files/{self.id}/{self.video.name}" if self.video else None,
            "motion": {
                "state": self.motion_state, "progress": round(self.motion_progress, 3), "error": self.motion_error,
                "moves": self.camera_moves,
                "lost_frames": int((~self.motion.tracked).sum()) if self.motion is not None else 0,
            },
        }
        report = self.out / "report.json"
        if self.state == "done" and report.exists():
            data["results"] = json.loads(report.read_text(encoding="utf-8"))
        return data


class Keyframe(BaseModel):
    frame: int = 0
    points: dict[str, tuple[float, float]]
    rim: tuple[float, float] | None = None


class CalibrationRequest(BaseModel):
    keyframes: list[Keyframe] | None = None
    points: dict[str, tuple[float, float]] | None = None  # shorthand for one keyframe on frame 0
    rim: tuple[float, float] | None = None
    rim_radius_px: float = 30.0

    def keyframe_list(self) -> list[Keyframe]:
        if self.keyframes:
            return self.keyframes
        if self.points:
            return [Keyframe(frame=0, points=self.points, rim=self.rim)]
        raise HTTPException(400, "כיול לא תקין: לא סומנו נקודות")


class PreviewRequest(CalibrationRequest):
    frame: int = 0


class AnalyzeRequest(CalibrationRequest):
    model: str = MODELS[0]
    stride: int = 1
    events: list[dict] | None = None


def _require_password(app: FastAPI, password: str) -> None:
    @app.middleware("http")
    async def check(request: Request, call_next):
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode().partition(":")
            except ValueError:
                given = ""
            if secrets.compare_digest(given, password):
                return await call_next(request)
        return Response("Password required", 401, {"WWW-Authenticate": 'Basic realm="courtvision"'})


def _cleanup(workspace: Path, keep_hours: float) -> None:
    cutoff = time.time() - keep_hours * 3600
    for d in workspace.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def create_app(workspace: str | Path = "workspace", password: str | None = None) -> FastAPI:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    password = password if password is not None else os.environ.get("COURTVISION_PASSWORD")
    max_upload = float(os.environ.get("COURTVISION_MAX_UPLOAD_MB", 500)) * 1024 * 1024
    keep_hours = float(os.environ.get("COURTVISION_KEEP_HOURS", 48))
    jobs: dict[str, Job] = {}
    heavy = threading.Lock()  # one analysis at a time; YOLO already uses every CPU core
    app = FastAPI(title="courtvision")
    if password:
        _require_password(app, password)

    def get_job(job_id: str) -> Job:
        if job_id not in jobs:
            raise HTTPException(404, "הסרטון לא נמצא. אולי השרת הופעל מחדש – העלו אותו שוב.")
        return jobs[job_id]

    def keyframes(req: CalibrationRequest) -> dict[int, Calibration]:
        out = {}
        for kf in req.keyframe_list():
            try:
                out[kf.frame] = Calibration(points=dict(kf.points), rim_px=kf.rim, rim_radius_px=req.rim_radius_px)
            except ValueError as e:
                raise HTTPException(400, f"כיול לא תקין בפריים {kf.frame}: {e}")
        return out

    def save_thumb(job: Job, frame_index: int, frame: np.ndarray) -> None:
        scale = min(1.0, THUMB_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        cv2.imwrite(str(job.thumb(frame_index)), small, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def follow_camera(job: Job) -> None:
        """Runs right after upload, while the user marks the court."""
        def on_frame(t, frame):
            if t % THUMB_STEP == 0 and t:
                save_thumb(job, t, frame)

        def progress(done, total):
            job.motion_progress = done / max(total, 1)

        try:
            job.motion_state = "running"
            motion = estimate_motion(job.video, progress, on_frame)
            motion.save(job.dir / "motion.npz")
            job.frames = len(motion)
            job.camera_moves = not motion.is_static((job.width, job.height))
            job.motion, job.motion_state, job.motion_progress = motion, "done", 1.0
        except Exception as e:
            traceback.print_exc()
            job.motion_state, job.motion_error = "error", f"{type(e).__name__}: {e}"
        finally:
            job.motion_done.set()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/landmarks")
    def landmarks():
        geometry = {k: getattr(court, k.upper()) for k in (
            "width", "half_length", "basket", "three_radius", "corner_three_x", "corner_three_y",
            "lane_half_width", "lane_length", "restricted_radius", "circle_radius")}
        return {"landmarks": court.LANDMARKS, "court": geometry, "models": MODELS}

    @app.post("/api/videos")
    def upload(file: UploadFile = File(...)):
        _cleanup(workspace, keep_hours)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_TYPES:
            raise HTTPException(400, f"סוג קובץ לא נתמך ({suffix or 'ללא סיומת'}). אפשר: {', '.join(sorted(VIDEO_TYPES))}")
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dir=workspace / job_id)
        (job.dir / "frames").mkdir(parents=True)
        job.video = job.dir / f"video{suffix}"
        size = 0
        with job.video.open("wb") as f:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > max_upload:
                    f.close()
                    shutil.rmtree(job.dir, ignore_errors=True)
                    raise HTTPException(413, f"הסרטון גדול מדי (מעל {max_upload / 2**20:.0f}MB). קצרו אותו לקטע של התקפה או שתיים.")
                f.write(chunk)
        cap = cv2.VideoCapture(str(job.video))
        ok, frame = cap.read()
        job.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        job.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not ok:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise HTTPException(400, "לא הצלחתי לקרוא את הסרטון. נסו לייצא אותו מחדש כ-MP4.")
        job.height, job.width = frame.shape[:2]
        save_thumb(job, 0, frame)
        jobs[job.id] = job
        threading.Thread(target=follow_camera, args=(job,), daemon=True).start()
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        return get_job(job_id).public()

    @app.get("/api/jobs/{job_id}/frames/{frame}.jpg")
    def frame_image(job_id: str, frame: int):
        job = get_job(job_id)
        path = job.thumb(frame)
        if not path.exists():
            raise HTTPException(404, "הפריים עוד לא מוכן")
        return FileResponse(path)

    @app.post("/api/jobs/{job_id}/preview")
    def preview(job_id: str, req: PreviewRequest):
        """Court lines drawn on any frame, to check the marking (and the camera tracking) by eye."""
        job = get_job(job_id)
        kfs = keyframes(req)
        if req.frame not in kfs and job.motion_state != "done":
            raise HTTPException(409, "עדיין מחשב את תנועת המצלמה – אפשר לבדוק בינתיים רק פריים שסומן")
        img = cv2.imread(str(job.thumb(req.frame)))
        if img is None:
            raise HTTPException(404, "הפריים לא נמצא")
        camera = CameraTrack(kfs, job.motion if job.camera_moves else None)
        scale = img.shape[1] / job.width
        draw_court_lines(img, lambda pts: camera.to_image(req.frame, pts) * scale)
        rim = camera.rim(req.frame)
        if rim is not None:
            cv2.circle(img, (int(rim[0] * scale), int(rim[1] * scale)), int(req.rim_radius_px * scale), (0, 140, 255), 2)
        if req.frame in kfs:
            for px in kfs[req.frame].points.values():
                cv2.circle(img, (int(px[0] * scale), int(px[1] * scale)), 5, (0, 0, 255), -1)
        name = f"preview_{req.frame}.jpg"
        cv2.imwrite(str(job.dir / name), img)
        errors = {f: c.reprojection_error for f, c in kfs.items()}
        return {"error_m": max(errors.values()), "image": f"/files/{job.id}/{name}"}

    @app.post("/api/jobs/{job_id}/analyze")
    def analyze(job_id: str, req: AnalyzeRequest):
        job = get_job(job_id)
        if req.model not in MODELS:
            raise HTTPException(400, f"מודל לא מוכר: {req.model}")
        kfs = {0: demo.calibration()} if job.demo else keyframes(req)
        with job.lock:
            if job.state in ("queued", "running"):
                raise HTTPException(409, "הניתוח כבר רץ")
            job.state, job.stage, job.progress, job.error = "queued", "ממתין לתורו", 0.0, None
        if not job.demo:
            (job.dir / "calibration.json").write_text(json.dumps(
                {"rim_radius_px": req.rim_radius_px, "keyframes": [kf.model_dump() for kf in req.keyframe_list()]}, indent=2))
        threading.Thread(target=run, args=(job, kfs, req), daemon=True).start()
        return job.public()

    @app.post("/api/demo")
    def start_demo():
        """The demo play filmed by a simulated hand-held camera, with known detections."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dir=workspace / f"demo-{job_id}", demo=True)
        (job.dir / "frames").mkdir(parents=True)
        job.state, job.stage = "queued", "מצלם את מהלך הדמו"
        job.fps, job.width, job.height = demo.FPS, 1280, 720
        jobs[job.id] = job

        def prepare_and_run():
            job.video = demo.render_video(job.dir / "video.mp4", moving=True)
            follow_camera(job)
            run(job, {0: demo.calibration()}, None)

        threading.Thread(target=prepare_and_run, daemon=True).start()
        return job.public()

    def run(job: Job, kfs: dict[int, Calibration], req: AnalyzeRequest | None) -> None:
        try:
            if not job.motion_done.is_set():
                job.stage = "ממתין לחישוב תנועת המצלמה"
                job.motion_done.wait()
            with heavy:
                job.state = "running"
                if job.demo:
                    detections, fps, stride = demo.detections(moving=True), demo.FPS, 1
                else:
                    detections, fps = cached_detections(job, req.model, req.stride)
                    stride = req.stride
                job.stage, job.progress = "מנתח החלטות", 0.0
                camera = CameraTrack(kfs, job.motion if job.camera_moves else None, stride)
                analysis = analyze_detections(detections, camera, fps, req.events if req and req.events else None)

                def on_video(done, total):
                    job.stage, job.progress = "יוצר סרטון דו-ממדי", done / max(total, 1)

                write_outputs(analysis, job.out, job.video, stride, on_video)
                job.state, job.stage, job.progress = "done", "", 1.0
        except ValueError as e:  # bad input, e.g. a manual event at a moment with no ball holder
            job.state, job.error = "error", str(e)
        except Exception as e:  # report any failure to the page instead of a silent dead thread
            traceback.print_exc()
            job.state, job.error = "error", f"{type(e).__name__}: {e}"

    def cached_detections(job: Job, model: str, stride: int):
        """Detection is the slow step and does not depend on calibration or events, so it runs once per model."""
        cache = job.dir / f"detections_{Path(model).stem}_s{stride}.pkl"
        if cache.exists():
            return pickle.loads(cache.read_bytes())

        def on_frame(done, total):
            job.stage, job.progress = "מזהה שחקנים וכדור", done / max(total, 1)

        job.stage = "טוען את מודל הזיהוי"
        result = detect_video(job.video, model, stride, progress=on_frame)
        cache.write_bytes(pickle.dumps(result))
        return result

    @app.get("/files/{job_id}/{path:path}")
    def files(job_id: str, path: str):
        job = get_job(job_id)
        target = (job.dir / path).resolve()
        if not target.is_relative_to(job.dir.resolve()) or not target.is_file() or target.suffix in (".pkl", ".npz"):
            raise HTTPException(404, "הקובץ לא נמצא")
        return FileResponse(target)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
