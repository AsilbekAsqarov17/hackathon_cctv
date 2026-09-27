"""Flow direction restricted to one side of a vertical cut in the frame.

Used to check the direction of a specific approach independently of lane
polygons: for vehicles west or east of a chosen x, report the share moving
image-right versus image-left, banded by y.
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
    parser.add_argument("--cut-x", type=int, required=True)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--min-travel", type=float, default=60.0)
    parser.add_argument("--band", type=int, default=100)
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
    right: dict[int, int] = defaultdict(int)
    left: dict[int, int] = defaultdict(int)
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
            if math.hypot(vx, vy) < args.min_travel or abs(vx) < abs(vy) * 0.8:
                continue
            x = (recent[0][1] + recent[-1][1]) / 2.0
            if (x < args.cut_x) != (recent[0][1] < args.cut_x):
                continue  # crossed the cut inside the window
            band = int(((recent[0][2] + recent[-1][2]) / 2.0) // args.band) * args.band
            if vx > 0:
                right[band] += 1
            else:
                left[band] += 1

    side = "WEST of cut" if True else ""
    print(f"moving vehicles on the {side} x={args.cut_x} (window={args.window}s, travel>={args.min_travel}px)")
    print(f"{'y band':>8} {'->right':>9} {'<-left':>8} {'total':>7} {'%right':>8}  dominant")
    for band in sorted(set(right) | set(left)):
        r, l = right[band], left[band]
        total = r + l
        if total < 5:
            continue
        pct = 100.0 * r / total
        dom = "EASTBOUND" if pct > 65 else ("WESTBOUND" if pct < 35 else "MIXED")
        print(f"{band:8d} {r:9d} {l:8d} {total:7d} {pct:7.1f}%  {dom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
