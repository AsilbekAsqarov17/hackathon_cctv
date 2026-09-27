"""Map where each directional traffic stream physically sits in the image.

Horizontal y-bands are misleading here because the road descends to the right,
so one band cuts across several carriageways at different x. This aggregates
moving vehicle windows into a 2D (x, y) grid and reports the eastbound /
westbound split per cell, which shows the real physical boundary between the
two streams.

Turning traffic is excluded: only windows whose net motion is close to
horizontal (|vx| > ratio*|vy|) and long enough to be a real traverse count.

Example::

    python scripts/flow_map_2d.py debug/perception/data_video1_tracks.jsonl ^
        --draw data/authoring/flow_map_2d.jpg
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import trailing_window  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--cell", type=int, default=240)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--min-travel", type=float, default=70.0)
    parser.add_argument("--ratio", type=float, default=0.8, help="|vx| must exceed ratio*|vy|")
    parser.add_argument("--min-cell", type=int, default=15)
    parser.add_argument("--draw", default="")
    args = parser.parse_args()

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            tid = track["track_id"]
            klass[tid] = track["class_name"]
            series[tid].append((record["timestamp"], track["bottom_center"][0], track["bottom_center"][1]))

    vehicles = {"car", "truck", "bus", "motorcycle", "vehicle", "bicycle"}
    right: dict[tuple[int, int], int] = defaultdict(int)
    left: dict[tuple[int, int], int] = defaultdict(int)
    for tid, points in series.items():
        if klass.get(tid) not in vehicles or len(points) < 30:
            continue
        for i in range(len(points)):
            t_end = points[i][0]
            recent = trailing_window(points, args.window, t_end)
            if len(recent) < 2:
                continue
            vx = recent[-1][1] - recent[0][1]
            vy = recent[-1][2] - recent[0][2]
            if math.hypot(vx, vy) < args.min_travel:
                continue
            if abs(vx) < args.ratio * abs(vy):
                continue  # turning or near-vertical: not a traverse
            x = (recent[0][1] + recent[-1][1]) / 2.0
            y = (recent[0][2] + recent[-1][2]) / 2.0
            if not (0 <= x < args.width and 0 <= y < args.height):
                continue
            key = (int(x // args.cell), int(y // args.cell))
            if vx > 0:
                right[key] += 1
            else:
                left[key] += 1

    cells = sorted(set(right) | set(left), key=lambda k: (k[1], k[0]))
    print(f"cell={args.cell}px  window={args.window}s  min_travel={args.min_travel}px  "
          f"|vx|>{args.ratio}|vy|")
    print(f"showing cells with >= {args.min_cell} moving windows\n")
    header = f"{'x0':>6} {'y0':>6} {'->right':>8} {'<-left':>7} {'total':>6} {'%right':>7}  verdict"
    print(header)
    print("-" * len(header))
    rows = []
    for key in cells:
        r, l = right[key], left[key]
        total = r + l
        if total < args.min_cell:
            continue
        pct = 100.0 * r / total
        verdict = "EASTBOUND" if pct > 80 else ("WESTBOUND" if pct < 20 else "mixed")
        rows.append((key, r, l, total, pct, verdict))
        print(f"{key[0] * args.cell:6d} {key[1] * args.cell:6d} {r:8d} {l:7d} {total:6d} "
              f"{pct:6.1f}%  {verdict}")

    if args.draw:
        import cv2
        import numpy as np

        img = np.full((args.height, args.width, 3), 22, dtype=np.uint8)
        for key, r, l, total, pct, verdict in rows:
            x0, y0 = key[0] * args.cell, key[1] * args.cell
            if verdict == "EASTBOUND":
                color = (0, 140, 255)
            elif verdict == "WESTBOUND":
                color = (255, 120, 0)
            else:
                color = (140, 140, 140)
            alpha = min(0.85, 0.25 + 0.6 * total / 200.0)
            patch = img[y0:y0 + args.cell, x0:x0 + args.cell]
            img[y0:y0 + args.cell, x0:x0 + args.cell] = (
                patch.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha
            ).astype(np.uint8)
            cv2.putText(img, f"{pct:.0f}%", (x0 + 6, y0 + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        for name, color in (("EASTBOUND", (0, 140, 255)), ("WESTBOUND", (255, 120, 0))):
            cv2.putText(img, name, (24, 40 + (0 if name == "EASTBOUND" else 34)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        from pathlib import Path as _P
        _P(args.draw).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.draw, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\nwrote {args.draw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
