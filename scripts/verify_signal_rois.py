"""Quantitative check that configured signal ROIs sit on real, cycling lamps.

For each configured traffic-light ROI this samples the video and counts how often
a strongly saturated red / yellow / green lamp fills enough of the ROI to give a
definite reading. A correct ROI shows a definite colour on most frames; a ROI
placed on grass, asphalt or a static sign does not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import load_scene_config  # noqa: E402


def lit_counts(frame: np.ndarray, roi) -> tuple[int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in roi)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(int)
    sat = hsv[:, :, 1].astype(int)
    val = hsv[:, :, 2].astype(int)
    red = int((((hue <= 10) | (hue >= 170)) & (sat >= 80) & (val >= 80)).sum())
    green = int(((hue >= 40) & (hue <= 90) & (sat >= 80) & (val >= 80)).sum())
    yellow = int(((hue >= 18) & (hue <= 38) & (sat >= 80) & (val >= 80)).sum())
    return red, green, yellow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--step", type=int, default=17)
    args = parser.parse_args()

    config = load_scene_config(args.scene, args.width, args.height)
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stats = {
        light.light_id: {"red": 0, "green": 0, "yellow": 0, "dark": 0}
        for light in config.traffic_lights
    }
    sampled = 0
    for index in range(0, total, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        sampled += 1
        for light in config.traffic_lights:
            red, green, yellow = lit_counts(frame, light.roi)
            entry = stats[light.light_id]
            if red >= max(green, yellow) and red >= 8:
                entry["red"] += 1
            elif green >= max(red, yellow) and green >= 8:
                entry["green"] += 1
            elif yellow >= max(red, green) and yellow >= 8:
                entry["yellow"] += 1
            else:
                entry["dark"] += 1
    cap.release()

    print(f"sampled {sampled} frames of {args.video}")
    for light_id, entry in stats.items():
        seen = sum(entry.values())
        definite = seen - entry["dark"]
        pct = 100.0 * definite / seen if seen else 0.0
        print(
            f"  {light_id:18} red={entry['red']:4d} green={entry['green']:4d} "
            f"yellow={entry['yellow']:4d} dark={entry['dark']:4d} "
            f"-> definite colour on {pct:.1f}% of frames"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
