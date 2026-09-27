"""Locate painted road lines from the video itself, not from an authored guess.

A stop line's job is to be crossed, so its exact extent decides whether a rule
can see anything. This finds bright paint on the asphalt inside a search region
and reports the dominant line's endpoints in video pixels, plus the equivalent
coordinates in the scene's authoring resolution.

Only frames where the region is not dominated by vehicles are used, so a white
car body cannot be mistaken for paint.

Example::

    python scripts/find_paint_lines.py data/data_video2.mp4 --times 30 40 60 90 120 \
        --search 980 300 1300 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--search", nargs=4, type=int, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--author-width", type=int, default=3840,
                        help="scene authoring width, for reporting both frames")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = args.search
    k = width / args.author_width

    wanted = {round(t * args.fps): t for t in args.times}
    accumulator = np.zeros((y1 - y0, x1 - x0), np.float32)
    used = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            index += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = cv2.GaussianBlur(gray, (0, 0), 25)
            paint = ((gray > 170) & (gray.astype(np.int16) - background.astype(np.int16) > 25))
            region = paint[y0:y1, x0:x1]
            accumulator += region.astype(np.float32)
            used += 1
            if args.out:
                vis = frame[y0:y1, x0:x1].copy()
                vis[region] = (0, 255, 255)
                path = Path(args.out) / f"paint_{wanted[index - 1]:07.1f}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            continue
        index += 1
    capture.release()
    if not used:
        raise SystemExit("no frames matched --times")
    mean = accumulator / used

    # A painted line spanning a carriageway is bright over most of its length.
    # Threshold at half the maximum so faint markings and noise drop out.
    mask = (mean > 0.45 * mean.max()).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 120:
            continue
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        components.append({
            "area": int(area),
            "x0": int(xs.min() + x0), "x1": int(xs.max() + x0),
            "y0": int(ys.min() + y0), "y1": int(ys.max() + y0),
            "cx": float(xs.mean() + x0), "cy": float(ys.mean() + y0),
            "width": int(xs.max() - xs.min() + 1),
            "height": int(ys.max() - ys.min() + 1),
            "elongation": max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            / max(1, min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
        })
    components.sort(key=lambda c: -c["area"])

    print(f"video {width}x{height}  search x{x0}..{x1} y{y0}..{y1}  frames used {used}")
    print(f"{'area':>7} {'x0':>6}{'x1':>6}{'y0':>6}{'y1':>6} {'w':>5}{'h':>5} {'elong':>6}  "
          f"{'4K x0':>7}{'4K y0':>7}{'4K x1':>7}{'4K y1':>7}")
    print("-" * 96)
    for comp in components[:12]:
        print(f"{comp['area']:7d} {comp['x0']:6d}{comp['x1']:6d}{comp['y0']:6d}{comp['y1']:6d} "
              f"{comp['width']:5d}{comp['height']:5d} {comp['elongation']:6.1f}  "
              f"{comp['x0'] / k:7.0f}{comp['y0'] / k:7.0f}{comp['x1'] / k:7.0f}{comp['y1'] / k:7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
