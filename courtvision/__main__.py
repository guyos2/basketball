"""Command line: python -m courtvision {calibrate,analyze,demo} ..."""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2

from . import demo
from .calibration import CameraTrack, interactive_calibrate, load_keyframes
from .camera_motion import estimate_motion
from .feedback import decision_text
from .pipeline import analyze_detections, detect_video, write_outputs


def cmd_calibrate(args) -> None:
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Cannot read a frame from {args.video}")
    if args.frame_out:
        cv2.imwrite(args.frame_out, frame)
        print(f"Saved the first frame to {args.frame_out}. Fill the pixel coordinates in a calibration JSON "
              f"(see examples/calibration_example.json).")
        return
    calib = interactive_calibrate(frame)
    calib.save(args.out)
    print(f"Saved {args.out} ({len(calib.points)} points, mean error {calib.reprojection_error:.2f} m)")


def cmd_analyze(args) -> None:
    keyframes = load_keyframes(args.calib)
    for frame, calib in keyframes.items():
        print(f"Calibration on frame {frame}: {len(calib.points)} points, mean error {calib.reprojection_error:.2f} m")

    def progress(label):
        return lambda done, total: print(f"\r{label}: frame {done}/{total}", end="", flush=True)

    motion = None
    if not args.fixed_camera:
        motion = estimate_motion(args.video, progress("Following the camera"))
        print()
        cap = cv2.VideoCapture(args.video)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if motion.is_static(size):
            print("The camera does not move.")
            motion = None
        else:
            print(f"The camera moves (up to {motion.max_shift(size):.0f} px); the court follows it. "
                  f"Lost track in {int((~motion.tracked).sum())} frames.")
    elif len(keyframes) > 1:
        sys.exit("Several calibrated frames only make sense for a moving camera (drop --fixed-camera).")
    detections, fps = detect_video(args.video, args.model, args.stride, args.max_frames, progress("Detecting players and ball"))
    print()
    camera = CameraTrack(keyframes, motion, args.stride)
    analysis = analyze_detections(detections, camera, fps, args.events)
    out = write_outputs(analysis, args.out, args.video, args.stride)
    _summary(analysis, out)


def cmd_demo(args) -> None:
    if not args.moving:
        analysis = analyze_detections(demo.detections(), demo.calibration(), demo.FPS)
        out = write_outputs(analysis, args.out)
        return _summary(analysis, out)
    # A hand-held camera: render the play as a video, follow the camera, then analyze.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    video = demo.render_video(Path(args.out) / "demo_input.mp4", moving=True)
    motion = estimate_motion(video, lambda d, t: print(f"\rFollowing the camera: frame {d}/{t}", end="", flush=True))
    print()
    camera = CameraTrack({0: demo.calibration()}, motion)
    analysis = analyze_detections(demo.detections(moving=True), camera, demo.FPS)
    out = write_outputs(analysis, args.out, video)
    _summary(analysis, out)


def cmd_serve(args) -> None:
    import uvicorn

    from .web.app import create_app

    url = f"http://localhost:{args.port}"
    if args.open:
        threading.Thread(target=_open_when_ready, args=(url, args.port), daemon=True).start()
    print(f"\nThe interface runs at {url}\nKeep this window open while you use it; close it (or press Ctrl+C) to stop.\n")
    uvicorn.run(create_app(args.workspace), host=args.host, port=args.port, log_level="warning")


def _open_when_ready(url: str, port: int) -> None:
    """Open the browser once the server accepts connections."""
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.5)


def _summary(analysis, out: Path) -> None:
    print(f"Players on the map: {len(analysis.teams)}, events: {len(analysis.events)}")
    for d in analysis.decisions:
        print(decision_text(d))
    print(f"\nOutputs in {out}/: report.md, report.json, tracks.csv, court2d video, decisions/*.png")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="courtvision", description="Basketball video to 2D court + shoot-or-pass analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="mark court landmarks on the first frame")
    p.add_argument("video")
    p.add_argument("--out", default="calibration.json")
    p.add_argument("--frame-out", help="only save the first frame as an image, to fill the JSON by hand")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("analyze", help="analyze a video")
    p.add_argument("video")
    p.add_argument("--calib", required=True, help="calibration JSON")
    p.add_argument("--events", help="optional JSON of shots/passes typed by hand, replaces automatic detection")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (larger = more accurate, e.g. yolo11m.pt)")
    p.add_argument("--stride", type=int, default=1, help="analyze every Nth frame")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--fixed-camera", action="store_true", help="skip camera-motion tracking (tripod videos)")
    p.add_argument("--out", default="output")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("demo", help="run the analysis on a built-in synthetic play")
    p.add_argument("--out", default="demo_output")
    p.add_argument("--moving", action="store_true", help="film the play with a simulated hand-held camera")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="web interface: upload a video and analyze it in the browser")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workspace", default="workspace", help="where uploaded videos and results are kept")
    p.add_argument("--open", action="store_true", help="open the browser when the server is ready")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
