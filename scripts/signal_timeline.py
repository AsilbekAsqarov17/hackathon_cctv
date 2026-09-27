"""Print the phase timeline of a configured traffic-light ROI over a video.

Decoding is sequential on purpose: seeking a 4K file per sample is far slower
than reading straight through.

Example::

    python scripts/signal_timeline.py data/data_video1.mp4 ^
        --scene configs/scenes/data_video1.json --light signal_eastbound
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.traffic_lights import TrafficLightPerception  # noqa: E402
from src.scene.config import load_scene_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--light", default="signal_eastbound")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    config = load_scene_config(args.scene, args.width, args.height)
    light = next((item for item in config.traffic_lights if item.light_id == args.light), None)
    if light is None:
        raise SystemExit(f"light {args.light!r} not in {args.scene}")
    reader = TrafficLightPerception(
        {"model": "weights/yolo11n.pt", "confidence": 0.10, "imgsz": 640, "device": "0"},
        min_pixels=8,
    )

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    samples = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.step == 0:
            color, _ = reader.classify_traffic_light_state(frame, light.roi)
            samples.append((index / fps, color))
        index += 1
    cap.release()

    phases = []
    current = None
    for timestamp, color in samples:
        if current is None or current[0] != color:
            if current is not None:
                phases.append((current[0], current[1], start, timestamp))
            current = (color, timestamp)
            start = timestamp
    if current is not None:
        phases.append((current[0], current[1], start, samples[-1][0]))

    print(f"{args.light} phase timeline ({len(samples)} samples, step={args.step} frames):")
    for color, _first, begin, end in phases:
        if end - begin > 0.3:
            print(f"  t={begin:7.2f} -> {end:7.2f}  ({end - begin:6.2f}s)  {color}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
