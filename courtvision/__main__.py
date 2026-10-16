"""Command line: python -m courtvision {calibrate,analyze,demo} ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import demo
from .calibration import Calibration, interactive_calibrate
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
    calib = Calibration.load(args.calib)
    print(f"Calibration: {len(calib.points)} points, mean error {calib.reprojection_error:.2f} m")

    def progress(done, total):
        print(f"\rDetecting players and ball: frame {done}/{total}", end="", flush=True)

    detections, fps = detect_video(args.video, args.model, args.stride, args.max_frames, progress)
    print()
    analysis = analyze_detections(detections, calib, fps, args.events)
    out = write_outputs(analysis, args.out, args.video, args.stride)
    _summary(analysis, out)


def cmd_demo(args) -> None:
    analysis = analyze_detections(demo.detections(), demo.calibration(), demo.FPS)
    out = write_outputs(analysis, args.out)
    _summary(analysis, out)


def cmd_serve(args) -> None:
    import uvicorn

    from .web.app import create_app

    print(f"Open http://localhost:{args.port} in your browser")
    uvicorn.run(create_app(args.workspace), host=args.host, port=args.port, log_level="warning")


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
    p.add_argument("--out", default="output")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("demo", help="run the analysis on a built-in synthetic play")
    p.add_argument("--out", default="demo_output")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="web interface: upload a video and analyze it in the browser")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workspace", default="workspace", help="where uploaded videos and results are kept")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
