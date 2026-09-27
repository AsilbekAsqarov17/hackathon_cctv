"""Measure the lateral position of each directional stream, column by column.

For a set of x stations this reports the y distribution of westbound and
eastbound moving vehicles, so the far edge, the centre boundary and the near
edge of a two-way carriageway can be read off as reproducible coordinates.

Example::

    python scripts/measure_stream_bands.py debug/perception/data_video1_tracks.jsonl
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
def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import trailing_window  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--stations", type=int, default=12)
    parser.add_argument("--x-start", type=int, default=300)
    parser.add_argument("--x-end", type=int, default=3300)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--min-travel", type=float, default=70.0)
    parser.add_argument("--ratio", type=float, default=0.8)
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
    west: dict[int, list[float]] = defaultdict(list)
    east: dict[int, list[float]] = defaultdict(list)
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
            if math.hypot(vx, vy) < args.min_travel or abs(vx) < args.ratio * abs(vy):
                continue
            x = (recent[0][1] + recent[-1][1]) / 2.0
            y = (recent[0][2] + recent[-1][2]) / 2.0
            if not (args.x_start <= x < args.x_end):
                continue
            index = int((x - args.x_start) * args.stations / (args.x_end - args.x_start))
            index = max(0, min(args.stations - 1, index))
            (east if vx > 0 else west)[index].append(y)

    width = (args.x_end - args.x_start) / args.stations
    print(f"stations={args.stations} over x={args.x_start}..{args.x_end} (bin {width:.0f}px)")
    print(f"{'x_mid':>7} {'n_W':>5} {'W_p05':>7} {'W_p50':>7} {'W_p95':>7} | "
          f"{'n_E':>5} {'E_p05':>7} {'E_p50':>7} {'E_p95':>7} | {'gap':>7}")
    print("-" * 92)
    rows = []
    for index in range(args.stations):
        w, e = west.get(index, []), east.get(index, [])
        if len(w) < 8 or len(e) < 8:
            continue
        x_mid = args.x_start + (index + 0.5) * width
        wp05, wp50, wp95 = (percentile(w, 0.05), percentile(w, 0.5), percentile(w, 0.95))
        ep05, ep50, ep95 = (percentile(e, 0.05), percentile(e, 0.5), percentile(e, 0.95))
        gap = ep05 - wp95
        rows.append((x_mid, wp05, wp50, wp95, ep05, ep50, ep95, gap))
        print(f"{x_mid:7.0f} {len(w):5d} {wp05:7.0f} {wp50:7.0f} {wp95:7.0f} | "
              f"{len(e):5d} {ep05:7.0f} {ep50:7.0f} {ep95:7.0f} | {gap:7.0f}")

    if len(rows) >= 2:
        def fit(index: int) -> tuple[float, float]:
            xs = [r[0] for r in rows]
            ys = [r[index] for r in rows]
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den = sum((x - mean_x) ** 2 for x in xs)
            slope = num / den if den else 0.0
            return slope, mean_y - slope * mean_x

        print()
        for name, index in (("W far edge (p05)", 1), ("W median (p50)", 2),
                            ("W near edge (p95)", 3), ("E far edge (p05)", 4),
                            ("E median (p50)", 5), ("E near edge (p95)", 6)):
            slope, intercept = fit(index)
            print(f"  {name:20} y = {intercept:7.1f} + {slope:.4f} * x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
