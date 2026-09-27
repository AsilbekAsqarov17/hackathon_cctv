"""Where are the long-stationary vehicles, and are they queued or parked?

Replays a perception JSONL and reports every vehicle track with a long
stationary interval, together with its position, lane and distance to the
calibrated stop line, so a candidate can be judged as a signal queue, a parked
car, or a genuine stopped vehicle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import is_stationary, is_upstream_of_line  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.geometry import point_segment_distance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--net", type=float, default=25.0)
    parser.add_argument("--path", type=float, default=250.0)
    parser.add_argument("--min-duration", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    config = load_scene_config(args.scene, args.width, args.height)
    scene = SceneContext(config, args.width, args.height)
    lines = config.stop_lines

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            tid = track["track_id"]
            klass[tid] = track["class_name"]
            series[tid].append((record["timestamp"], track["bbox"], track["bottom_center"]))

    vehicles = {"car", "truck", "bus", "motorcycle", "vehicle", "bicycle"}
    rows = []
    for tid, points in series.items():
        if klass.get(tid) not in vehicles or len(points) < 30:
            continue
        history: list[tuple[float, float, float]] = []
        flags = []
        for timestamp, _bbox, bottom in points:
            history.append((timestamp, bottom[0], bottom[1]))
            flags.append(
                is_stationary(history, args.window, args.net, args.path, timestamp, min_samples=2)
            )
        start = None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                dur = points[i - 1][0] - points[start][0]
                if dur >= args.min_duration:
                    rows.append((dur, tid, points[start][0], points[i - 1][0], points[start][2], points[i - 1][2]))
                start = None
        if start is not None and points[-1][0] - points[start][0] >= args.min_duration:
            rows.append((points[-1][0] - points[start][0], tid, points[start][0], points[-1][0],
                         points[start][2], points[-1][2]))

    rows.sort(reverse=True)
    print(f"stationary intervals >= {args.min_duration}s: {len(rows)}")
    print(f"{'dur':>6} {'trk':>5} {'start':>7} {'end':>7} {'bottom_center':>16} {'lane':>5} {'d_line':>7} {'upstr':>6}")
    for dur, tid, t0, t1, p0, p1 in rows[: args.top]:
        point = (p1[0], p1[1])
        lane = scene.lane_for_point(point)
        lane_id = lane.lane_id if lane else None
        best, best_line = None, None
        for line in lines:
            d = min(point_segment_distance(point, line.segment[0], line.segment[1]),
                    point_segment_distance((p0[0], p0[1]), line.segment[0], line.segment[1]))
            if best is None or d < best:
                best, best_line = d, line
        upstream = is_upstream_of_line(point, best_line) if best_line is not None else False
        print(f"{dur:6.1f} {tid:5d} {t0:7.1f} {t1:7.1f} {str([int(point[0]), int(point[1])]):>16} "
              f"{str(lane_id):>5} {best:7.0f} {str(upstream):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
