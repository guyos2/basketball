import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from courtvision import demo
from courtvision.render import draw_court_overlay
from courtvision.web import app as web


@pytest.fixture
def client(tmp_path, monkeypatch):
    calls = []

    def fake_detect(video, model, stride, progress=None):
        calls.append((video, model, stride))
        return demo.detections(), demo.FPS

    monkeypatch.setattr(web, "detect_video", fake_detect)
    c = TestClient(web.create_app(tmp_path / "ws"))
    c.detect_calls = calls
    c.app_workspace = tmp_path / "ws"
    return c


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
    for _ in range(300):
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
    assert client.get(f"/files/{job['id']}/frame.jpg").status_code == 200

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


def test_demo_job(client):
    job = client.post("/api/demo").json()
    done = wait(client, job["id"])
    assert done["state"] == "done"
    assert done["results"]["decisions"][1]["recommended_target"] == 5


def test_files_cannot_escape_the_job(client, video_file):
    job = upload(client, video_file).json()
    assert client.get(f"/files/{job['id']}/../../etc/passwd").status_code == 404
    assert client.get(f"/files/{job['id']}/%2e%2e/%2e%2e/etc/passwd").status_code == 404
    assert client.get("/files/nope/frame.jpg").status_code == 404
