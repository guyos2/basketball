import numpy as np
import pytest

from courtvision import demo
from courtvision.calibration import Calibration, CameraTrack
from courtvision.camera_motion import CameraMotion, estimate_motion
from courtvision.pipeline import analyze_detections

PROBES = [[100, 100], [1180, 100], [1180, 620], [100, 620], [640, 360]]


@pytest.fixture(scope="module")
def moving(tmp_path_factory):
    video = demo.render_video(tmp_path_factory.mktemp("moving") / "clip.mp4", moving=True)
    return video, estimate_motion(video)


def calibration_on(frame: int) -> Calibration:
    """The demo calibration as it would be clicked on a later frame of the moving video."""
    base = demo.calibration()
    view = demo.camera_shake(frame / demo.FPS)
    points = {k: tuple(demo._warp([v], view)[0]) for k, v in base.points.items()}
    return Calibration(points, tuple(demo._warp([base.rim_px], view)[0]), base.rim_radius_px)


def test_follows_a_hand_held_camera(moving):
    _, motion = moving
    assert len(motion) == int(demo.DURATION * demo.FPS)
    assert motion.tracked.all()
    assert not motion.is_static((1280, 720))
    for t in range(len(motion)):
        truth = np.linalg.inv(demo.camera_shake(t / demo.FPS))
        error = np.abs(demo._warp(PROBES, truth) - demo._warp(PROBES, motion.to_first[t])).max()
        assert error < 3.0, f"frame {t}: {error:.1f}px"


def test_static_camera_is_recognized(tmp_path):
    motion = estimate_motion(demo.render_video(tmp_path / "tripod.mp4", moving=False))
    assert motion.is_static((1280, 720))


@pytest.mark.parametrize("marked_frame", [0, 70])
def test_moving_camera_analysis_matches_the_play(moving, marked_frame):
    _, motion = moving
    camera = CameraTrack({marked_frame: calibration_on(marked_frame)}, motion)
    analysis = analyze_detections(demo.detections(moving=True), camera, demo.FPS)
    assert [(e.kind, e.player, e.target) for e in analysis.events] == [("pass", 1, 2), ("shot", 2, None)]
    assert analysis.decisions[1].verdict == "should_pass" and analysis.decisions[1].recommended_target == 5
    errors = [np.hypot(*np.subtract(s.players[t].xy, demo.position(t, s.time))) for s in analysis.states for t in s.players]
    assert np.mean(errors) < 0.1 and np.max(errors) < 0.3


def test_ignoring_the_motion_breaks_the_analysis():
    analysis = analyze_detections(demo.detections(moving=True), demo.calibration(), demo.FPS)
    assert [(e.kind, e.player) for e in analysis.events] != [("pass", 1), ("shot", 2)]


def test_save_and_load(tmp_path, moving):
    _, motion = moving
    motion.save(tmp_path / "m.npz")
    loaded = CameraMotion.load(tmp_path / "m.npz")
    np.testing.assert_allclose(loaded.to_first, motion.to_first)
    assert CameraMotion.static(5).is_static((100, 100))
