import json

import cv2

from courtvision import demo
from courtvision.events import EventConfig, possession_runs, smooth_possession
from courtvision.pipeline import analyze_detections, write_outputs


def test_smooth_possession_fills_gaps_and_drops_blips():
    raw = [1, 1, 1, None, None, 1, 1, 7, None, None, None, None, None, None, None, None, None, None, 2, 2, 2]
    held = smooth_possession(raw, EventConfig(max_gap_frames=3, min_hold_frames=3))
    assert held[:7] == [1] * 7
    assert held[7] is None  # one-frame blip
    assert possession_runs(held) == [(1, 0, 6), (2, 18, 20)]


def test_demo_play_end_to_end(tmp_path):
    analysis = analyze_detections(demo.detections(), demo.calibration(), demo.FPS)

    assert len(analysis.teams) == 10
    offense = {t for t, team in analysis.teams.items() if team == analysis.teams[1]}
    assert offense == {1, 2, 3, 4, 5}

    # Players land within a few centimeters of where they were placed.
    state = analysis.states[0]
    assert abs(state.players[3].xy[0] - 0.6) < 0.1 and abs(state.players[3].xy[1] - 1.0) < 0.1

    kinds = [(e.kind, e.player, e.target) for e in analysis.events]
    assert kinds == [("pass", 1, 2), ("shot", 2, None)]
    shot = analysis.decisions[1]
    assert shot.verdict == "should_pass" and shot.recommended_target == 5

    out = write_outputs(analysis, tmp_path)
    report = json.loads((out / "report.json").read_text())
    assert [d["verdict"] for d in report["decisions"]] == ["good", "should_pass"]
    assert "עדיף היה למסור ל-#5" in (out / "report.md").read_text()
    assert len(list((out / "decisions").glob("*.png"))) == 2
    video = cv2.VideoCapture(str(out / "court2d.mp4"))
    assert video.get(cv2.CAP_PROP_FRAME_COUNT) > len(analysis.states)


def test_manual_events_override_detection(tmp_path):
    events = tmp_path / "events.json"
    events.write_text(json.dumps([{"time": 3.0, "type": "shot"}]))
    analysis = analyze_detections(demo.detections(), demo.calibration(), demo.FPS, manual_events=events)
    assert [(e.kind, e.player, e.source) for e in analysis.events] == [("shot", 2, "manual")]
