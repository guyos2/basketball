"""Local web interface: upload a video, mark the court, run the analysis, read the results.

Run with:  python -m courtvision serve   then open http://localhost:8000
"""
from __future__ import annotations

import json
import pickle
import shutil
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import court, demo
from ..calibration import Calibration
from ..pipeline import analyze_detections, detect_video, write_outputs
from ..render import draw_court_overlay

STATIC = Path(__file__).parent / "static"
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")


@dataclass
class Job:
    id: str
    dir: Path
    video: Path | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    state: str = "uploaded"  # uploaded, queued, running, done, error
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def out(self) -> Path:
        return self.dir / "output"

    def public(self) -> dict:
        data = {
            "id": self.id, "state": self.state, "stage": self.stage, "progress": round(self.progress, 3),
            "error": self.error, "width": self.width, "height": self.height, "fps": self.fps,
            "duration": self.duration, "has_video": self.video is not None,
            "video_url": f"/files/{self.id}/{self.video.name}" if self.video else None,
        }
        report = self.out / "report.json"
        if self.state == "done" and report.exists():
            data["results"] = json.loads(report.read_text(encoding="utf-8"))
        return data


class AnalyzeRequest(BaseModel):
    points: dict[str, tuple[float, float]]
    rim: tuple[float, float] | None = None
    rim_radius_px: float = 30.0
    model: str = MODELS[0]
    stride: int = 1
    events: list[dict] | None = None


class CalibrationRequest(BaseModel):
    points: dict[str, tuple[float, float]]
    rim: tuple[float, float] | None = None
    rim_radius_px: float = 30.0


def create_app(workspace: str | Path = "workspace") -> FastAPI:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, Job] = {}
    heavy = threading.Lock()  # one analysis at a time; YOLO already uses every CPU core
    app = FastAPI(title="courtvision")

    def get_job(job_id: str) -> Job:
        if job_id not in jobs:
            raise HTTPException(404, "הסרטון לא נמצא. אולי השרת הופעל מחדש – העלו אותו שוב.")
        return jobs[job_id]

    def calibration(req: CalibrationRequest) -> Calibration:
        try:
            return Calibration(points=dict(req.points), rim_px=req.rim, rim_radius_px=req.rim_radius_px)
        except ValueError as e:
            raise HTTPException(400, f"כיול לא תקין: {e}")

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
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_TYPES:
            raise HTTPException(400, f"סוג קובץ לא נתמך ({suffix or 'ללא סיומת'}). אפשר: {', '.join(sorted(VIDEO_TYPES))}")
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dir=workspace / job_id)
        job.dir.mkdir(parents=True)
        job.video = job.dir / f"video{suffix}"
        with job.video.open("wb") as f:
            shutil.copyfileobj(file.file, f, length=1 << 20)
        cap = cv2.VideoCapture(str(job.video))
        ok, frame = cap.read()
        job.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if not ok:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise HTTPException(400, "לא הצלחתי לקרוא את הסרטון. נסו לייצא אותו מחדש כ-MP4.")
        job.height, job.width = frame.shape[:2]
        job.duration = frames / job.fps if frames > 0 else 0.0
        cv2.imwrite(str(job.dir / "frame.jpg"), frame)
        jobs[job.id] = job
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        return get_job(job_id).public()

    @app.post("/api/jobs/{job_id}/preview")
    def preview(job_id: str, req: CalibrationRequest):
        job = get_job(job_id)
        calib = calibration(req)
        frame = cv2.imread(str(job.dir / "frame.jpg"))
        cv2.imwrite(str(job.dir / "preview.jpg"), draw_court_overlay(frame, calib))
        return {"error_m": calib.reprojection_error, "image": f"/files/{job.id}/preview.jpg"}

    @app.post("/api/jobs/{job_id}/analyze")
    def analyze(job_id: str, req: AnalyzeRequest):
        job = get_job(job_id)
        if req.model not in MODELS:
            raise HTTPException(400, f"מודל לא מוכר: {req.model}")
        calib = calibration(CalibrationRequest(points=req.points, rim=req.rim, rim_radius_px=req.rim_radius_px))
        with job.lock:
            if job.state in ("queued", "running"):
                raise HTTPException(409, "הניתוח כבר רץ")
            job.state, job.stage, job.progress, job.error = "queued", "ממתין לתורו", 0.0, None
        calib.save(job.dir / "calibration.json")
        threading.Thread(target=run, args=(job, calib, req), daemon=True).start()
        return job.public()

    @app.post("/api/demo")
    def start_demo():
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dir=workspace / f"demo-{job_id}")
        job.dir.mkdir(parents=True)
        job.state, job.fps, job.duration = "queued", demo.FPS, demo.DURATION
        jobs[job.id] = job
        threading.Thread(target=run, args=(job, demo.calibration(), None), daemon=True).start()
        return job.public()

    def run(job: Job, calib: Calibration, req: AnalyzeRequest | None) -> None:
        try:
            with heavy:
                job.state = "running"
                if req is None:
                    detections, fps, stride = demo.detections(), demo.FPS, 1
                else:
                    detections, fps = cached_detections(job, req.model, req.stride)
                    stride = req.stride
                job.stage, job.progress = "מנתח החלטות", 0.0
                analysis = analyze_detections(detections, calib, fps, req.events if req and req.events else None)

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
        if not target.is_relative_to(job.dir.resolve()) or not target.is_file() or target.suffix == ".pkl":
            raise HTTPException(404, "הקובץ לא נמצא")
        return FileResponse(target)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
