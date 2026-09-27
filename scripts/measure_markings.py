"""Measure painted road markings to propose geometry, for visual verification.

This reads the *physical paint* (bright low-saturation white, and saturated
yellow) rather than any object detector output, so it is a measurement aid and
not an inference of geometry from detections. Every element it proposes is
still confirmed by eye with the overlay produced by ``--draw``.

For each window it reports the oriented minimum-area rectangle of the paint
pixels, which is how zebra crossings and lane lines are fitted here.

Example::

    python scripts/measure_markings.py data/data_video1.mp4 --frame 4618 ^
        --window xwalk_near:2400,900,2800,1080 --draw data/authoring/markings.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def paint_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of white and yellow road paint.

    An erosion step removes broad bright areas (sidewalk paving, kerbs, lane
    islands) and leaves only markings thin enough to be paint, so the fit is
    not dragged onto the pavement.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    white = ((sat < 70) & (val > 150)).astype(np.uint8)
    yellow = (((hue >= 18) & (hue <= 40)) & (sat > 90) & (val > 150)).astype(np.uint8)
    mask = ((white | yellow) * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))


def measure_window(mask: np.ndarray, box: tuple[int, int, int, int], min_pixels: int) -> dict | None:
    x1, y1, x2, y2 = box
    sub = mask[y1:y2, x1:x2]
    ys, xs = np.nonzero(sub)
    if len(xs) < min_pixels:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect
    corners = cv2.boxPoints(rect)
    corners[:, 0] += x1
    corners[:, 1] += y1
    return {
        "window": list(box),
        "paint_pixels": int(len(xs)),
        "center": [round(float(cx + x1), 1), round(float(cy + y1), 1)],
        "size": [round(float(w), 1), round(float(h), 1)],
        "angle_deg": round(float(angle), 2),
        "polygon": [[round(float(px), 1), round(float(py), 1)] for px, py in corners],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--window", action="append", default=[], help="name:x1,y1,x2,y2")
    parser.add_argument("--min-pixels", type=int, default=300)
    parser.add_argument("--draw", default="")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.frame}")
    mask = paint_mask(frame)

    results = []
    canvas = frame.copy()
    overlay = np.zeros_like(frame)
    overlay[mask > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

    for spec in args.window:
        name, _, coords = spec.partition(":")
        box = tuple(int(v) for v in coords.split(","))
        item = measure_window(mask, box, args.min_pixels)
        if item is None:
            print(f"{name}: no paint found in {box}")
            continue
        item["name"] = name
        results.append(item)
        polygon = np.asarray(item["polygon"], dtype=np.int32)
        cv2.polylines(canvas, [polygon], True, (0, 255, 255), 5)
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
        cx, cy = item["center"]
        cv2.putText(canvas, name, (int(cx) - 40, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        print(
            f"{name}: px={item['paint_pixels']} center={item['center']} "
            f"size={item['size']} angle={item['angle_deg']}"
        )

    print(json.dumps(results, indent=2))
    if args.draw:
        Path(args.draw).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.draw, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {args.draw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
