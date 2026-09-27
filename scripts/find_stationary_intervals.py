"""Report stationary intervals per track from a perception JSONL.

Used to validate the stopped_vehicle definition: an interval is stationary when
the trailing-window bottom-centre net displacement stays under a threshold, and
a candidate event is a stationary interval lasting at least a minimum duration.
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
def net_and_path(points, window: float) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    t_end = points[-1][0]
    recent = trailing_window(points, window, t_end)
    if len(recent) < 2:
        return 0.0, 0.0
    net = math.hypot(recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2])
    path = 0.0
    for i in range(1, len(recent)):
        path += math.hypot(recent[i][1] - recent[i - 1][1], recent[i][2] - recent[i - 1][2])
    return net, path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import trailing_window  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--net-threshold", type=float, default=25.0)
    parser.add_argument("--path-threshold", type=float, default=250.0)
    parser.add_argument("--min-duration", type=float, default=10.0)
    parser.add_argument("--focus", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--top", type=int, default=25)
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
    results = []
    for tid, points in series.items():
        if klass.get(tid) not in vehicles or len(points) < 30:
            continue
        flags = []
        for i in range(len(points)):
            net, path = net_and_path(points[: i + 1], args.window)
            flags.append(net <= args.net_threshold and path <= args.path_threshold)
        intervals = []
        start = None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = points[i][0]
            elif not flag and start is not None:
                if points[i - 1][0] - start >= args.min_duration:
                    intervals.append((start, points[i - 1][0]))
                start = None
        if start is not None and points[-1][0] - start >= args.min_duration:
            intervals.append((start, points[-1][0]))
        for a, b in intervals:
            results.append((b - a, tid, klass[tid], a, b))

    results.sort(reverse=True)
    print(
        f"window={args.window}s net<={args.net_threshold} path<={args.path_threshold} "
        f"min_duration={args.min_duration}s"
    )
    print(f"tracks analysed: {sum(1 for t in series if klass.get(t) in vehicles)}")
    print(f"stationary intervals >= {args.min_duration}s: {len(results)}\n")
    print(f"{'dur_s':>7} {'track':>6} {'class':<10} {'start':>8} {'end':>8}")
    for dur, tid, name, a, b in results[: args.top]:
        print(f"{dur:7.1f} {tid:6d} {name:<10} {a:8.2f} {b:8.2f}")

    if args.focus:
        print(f"\n=== track #{args.focus} ({klass.get(args.focus)}) stationary intervals (any length) ===")
        points = series[args.focus]
        flags = [net_and_path(points[: i + 1], args.window) for i in range(len(points))]
        start = None
        for i, (net, path) in enumerate(flags):
            ok = net <= args.net_threshold and path <= args.path_threshold
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                print(f"  t={points[start][0]:7.2f} -> t={points[i-1][0]:7.2f}  dur={points[i-1][0]-points[start][0]:6.2f}s")
                start = None
        if start is not None:
            print(f"  t={points[start][0]:7.2f} -> t={points[-1][0]:7.2f}  dur={points[-1][0]-points[start][0]:6.2f}s (to end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
