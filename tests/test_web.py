import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision import demo
from courtvision.camera_motion import CameraMotion
from courtvision.render import draw_court_overlay
from courtvision.web import app as web


def make_client(tmp_path, monkeypatch, real_motion: bool):
    calls = []

    def fake_detect(video, model, stride, progress=None):
        calls.append((video, model, stride))
        return c.fake_detections, demo.FPS

    def fake_motion(video, progress=None, on_frame=None):
        """Treat the camera as fixed without the (slow) feature matching; still emits every frame."""
        cap = cv2.VideoCapture(str(video))
        n = 0
        while (frame := cap.read()[1]) is not None:
            if on_frame:
                on_frame(n, frame)
            n += 1
        return CameraMotion.static(n)

    monkeypatch.setattr(web, "detect_video", fake_detect)
    if not real_motion:
        monkeypatch.setattr(web, "estimate_motion", fake_motion)
    c = TestClient(web.create_app(tmp_path / "ws", password=""))
    c.detect_calls = calls
    c.fake_detections = demo.detections()
    c.app_workspace = tmp_path / "ws"
    return c


@pytest.fixture
def client(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch, real_motion=False)


@pytest.fixture
def motion_client(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch, real_motion=True)


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    frame = draw_court_overlay(np.full((720, 1280, 3), 80, np.uint8), demo.calibration())
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), demo.FPS, (1280, 720))
    for _ in range(int(demo.DURATION * demo.FPS)):
        writer.write(frame)
    writer.release()
    return path


def wait(client, job_id):
    for _ in range(4000):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError(job)


def upload(client, path):
    with open(path, "rb") as f:
        return client.post("/api/videos", files={"file": (path.name, f, "video/mp4")})


def test_page_and_landmarks(client):
    assert "זריקה או מסירה" in client.get("/").text
    data = client.get("/api/landmarks").json()
    assert data["landmarks"]["midcourt_left"] == [0.0, 14.0]
    assert data["court"]["three_radius"] == 6.75


def test_rejects_non_video(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    assert upload(client, bad).status_code == 400
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"not a video")
    assert upload(client, fake).status_code == 400


def test_upload_calibrate_analyze_and_rerun(client, video_file):
    job = upload(client, video_file).json()
    assert (job["width"], job["height"]) == (1280, 720)
    assert client.get(f"/api/jobs/{job['id']}/frames/0.jpg").status_code == 200

    calib = demo.calibration()
    points = {k: list(v) for k, v in calib.points.items()}
    too_few = {"points": dict(list(points.items())[:3])}
    assert client.post(f"/api/jobs/{job['id']}/preview", json=too_few).status_code == 400

    body = {"points": points, "rim": list(calib.rim_px), "rim_radius_px": calib.rim_radius_px}
    preview = client.post(f"/api/jobs/{job['id']}/preview", json=body).json()
    assert preview["error_m"] < 0.01
    assert client.get(preview["image"]).status_code == 200

    assert client.post(f"/api/jobs/{job['id']}/analyze", json=body).status_code == 200
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    results = done["results"]
    assert [d["verdict"] for d in results["decisions"]] == ["good", "should_pass"]
    for d in results["decisions"]:
        assert client.get(f"/files/{job['id']}/output/{d['image']}").status_code == 200
    assert client.get(f"/files/{job['id']}/output/{results['video']}").status_code == 200
    source = cv2.VideoCapture(str(next(p for p in (client.app_workspace).rglob(results["source_video"]))))
    assert source.get(cv2.CAP_PROP_FRAME_COUNT) == int(demo.DURATION * demo.FPS)  # no freeze frames

    # Re-run with a hand-typed event: detection is cached, only the analysis runs again.
    rerun = dict(body, events=[{"time": 3.0, "type": "shot"}])
    client.post(f"/api/jobs/{job['id']}/analyze", json=rerun)
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    assert [(e["kind"], e["player"], e["source"]) for e in done["results"]["events"]] == [("shot", 2, "manual")]
    assert len(client.detect_calls) == 1

    # A manual event nobody can be found for is reported back, in Hebrew.
    client.post(f"/api/jobs/{job['id']}/analyze", json=dict(body, events=[{"time": 0.5, "type": "shot", "player": 99}]))
    failed = wait(client, job["id"])
    assert failed["state"] == "error" and "#99" in failed["error"]


def test_demo_job(motion_client):
    client = motion_client
    job = client.post("/api/demo").json()
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    assert done["motion"]["moves"] is True
    assert done["results"]["decisions"][1]["recommended_target"] == 5
    # Re-running the demo with a typed event uses the demo's own calibration and detections.
    client.post(f"/api/jobs/{job['id']}/analyze", json={"events": [{"time": 3.0, "type": "shot"}]})
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    assert len(client.detect_calls) == 0


def test_hand_held_video_with_a_later_marked_frame(motion_client, tmp_path):
    client = motion_client
    video = demo.render_video(tmp_path / "handheld.mp4", moving=True)
    client.fake_detections = demo.detections(moving=True)
    job = upload(client, video).json()
    for _ in range(4000):
        status = client.get(f"/api/jobs/{job['id']}").json()
        if status["motion"]["state"] == "done":
            break
        time.sleep(0.05)
    assert status["motion"]["moves"] and status["frames"] == int(demo.DURATION * demo.FPS)
    assert client.get(f"/api/jobs/{job['id']}/frames/70.jpg").status_code == 200

    base = demo.calibration()
    view = demo.camera_shake(70 / demo.FPS)
    keyframe = {"frame": 70, "points": {k: list(demo._warp([v], view)[0]) for k, v in base.points.items()},
                "rim": list(demo._warp([base.rim_px], view)[0])}
    body = {"keyframes": [keyframe], "rim_radius_px": base.rim_radius_px}
    # The court lines can be checked on a frame that was not marked: they follow the camera.
    preview = client.post(f"/api/jobs/{job['id']}/preview", json=dict(body, frame=20))
    assert preview.status_code == 200, preview.text

    client.post(f"/api/jobs/{job['id']}/analyze", json=body)
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    assert [d["verdict"] for d in done["results"]["decisions"]] == ["good", "should_pass"]


def test_password(tmp_path):
    c = TestClient(web.create_app(tmp_path / "ws", password="hoops"))
    assert c.get("/").status_code == 401
    assert c.get("/", auth=("coach", "wrong")).status_code == 401
    assert c.get("/", auth=("coach", "hoops")).status_code == 200


def test_files_cannot_escape_the_job(client, video_file):
    job = upload(client, video_file).json()
    assert client.get(f"/files/{job['id']}/../../etc/passwd").status_code == 404
    assert client.get(f"/files/{job['id']}/%2e%2e/%2e%2e/etc/passwd").status_code == 404
    assert client.get("/files/nope/frame.jpg").status_code == 404
