#!/usr/bin/env python3
"""
validate_signal_reader.py - check the traffic-light reader against known phases.

The `red_light` and `stop_line` rules are only as good as the signal reader, and
a wrong red costs precision on a class that is already rare. A fixed camera makes
this unusually cheap to verify honestly: read the phase by eye once, record the
windows, then let the code predict them.

This script does the second half. Supply the hand-read red windows and it reports
agreement, and specifically red precision and recall, over the whole clip.

    python scripts/validate_signal_reader.py VIDEO SCENE --light signal_centre \
        --red 0,36 --red 80,114 --red 158,194 --red 238,274
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.signals import TrafficLightReader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("scene")
    ap.add_argument("--light", default="signal_centre")
    ap.add_argument(
        "--red", action="append", default=[], metavar="A,B",
        help="a hand-read red window in seconds; repeat once per phase",
    )
    ap.add_argument("--hz", type=float, default=1.0, help="sampling rate (default 1 Hz)")
    args = ap.parse_args()

    windows = []
    for item in args.red:
        try:
            a, b = item.split(",")
            windows.append((float(a), float(b)))
        except ValueError:
            print(f"bad --red value: {item!r}", file=sys.stderr)
            return 2
    if not windows:
        print("supply at least one --red A,B window", file=sys.stderr)
        return 2

    def is_red(t: float) -> bool:
        return any(a <= t <= b for a, b in windows)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scene = SceneContext(load_scene_config(args.scene, width, height), width, height)
    reader = TrafficLightReader(scene)

    step = max(1, int(round(fps / max(0.1, args.hz))))
    samples: list[tuple[float, str]] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            state = reader.update(frame, [])
            light = state.get(args.light)
            samples.append((index / fps, light.color if light else "none"))
        index += 1
    cap.release()

    # The reader debounces over several samples, so the first ones are not a
    # fair test of it.
    body = samples[3:]
    tp = fp = tn = fn = unknown = 0
    for t, color in body:
        if color == "unknown":
            unknown += 1
            continue
        predicted_red = color == "red"
        if predicted_red and is_red(t):
            tp += 1
        elif predicted_red:
            fp += 1
        elif is_red(t):
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    if not total:
        print("no comparable samples", file=sys.stderr)
        return 1

    print(f"video        : {args.video}")
    print(f"light        : {args.light}   scene: {args.scene}")
    print(f"red windows  : {windows}  ({sum(b - a for a, b in windows):.0f} s of red)")
    print(f"samples      : {total} compared at {args.hz} Hz ({unknown} skipped as unknown)")
    print(f"agree RED    : {tp}")
    print(f"false RED    : {fp}")
    print(f"missed RED   : {fn}")
    print(f"agree GREEN  : {tn}")
    print(f"agreement    : {(tp + tn) / total:.1%}")
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    print(f"red precision: {precision:.1%}   red recall: {recall:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
