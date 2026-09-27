"""Inventory every long stop in a track dump, with where the vehicle came to rest.

Used to decide which stops are genuine signal queues (so they are suppressed)
and which are real ``stopped_vehicle`` events, and to sanity-check a freshly
placed stop line against the queue it is supposed to govern.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import is_stationary  # noqa: E402
from src.scene.config import load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--min-sec", type=float, default=5.0)
    parser.add_argument("--lanes", default="2,3")
    args = parser.parse_args()

    lane_ids = {int(v) for v in args.lanes.split(",") if v.strip()}
    config = load_scene_config(args.scene, args.width, args.height)
    line = config.stop_lines[0]
    (ax, ay), _ = line.segment
    travel = normalized_vector(line.direction) or (1.0, 0.0)

    def along(point):
        return (point[0] - ax) * travel[0] + (point[1] - ay) * travel[1]

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for raw in open(args.jsonl, encoding="utf-8"):
        record = json.loads(raw)
        for track in record["tracks"]:
            tid = track["track_id"]
            series[tid].append((record["timestamp"], track["bottom_center"], track["lane_id"], track["class_name"]))
            klass[tid] = track["class_name"]

    vehicles = {"car", "bus", "truck", "motorcycle", "bicycle", "vehicle"}
    print(f"video {args.width}x{args.height}   line anchor ({ax:.0f},{ay:.0f})   "
          f"lanes {sorted(lane_ids)}   min stop {args.min_sec}s")
    print(f"\n{'trk':>6} {'class':>9} {'lane':>5} {'stop_from':>10} {'stop_to':>9} "
          f"{'dur':>6} {'rest_along':>11} {'x':>6} {'y':>6}")
    print("-" * 84)
    rows = []
    for tid, points in series.items():
        if klass.get(tid) not in vehicles:
            continue
        history = [(p[0], p[1][0], p[1][1]) for p in points]
        start = None
        for timestamp, bottom, lane_id, _ in points:
            if lane_id in lane_ids and is_stationary(history, 2.5, 25.0, 250.0, timestamp):
                if start is None:
                    start = timestamp
            else:
                if start is not None and timestamp - start >= args.min_sec:
                    rows.append((start, timestamp, tid, klass[tid], lane_id,
                                 along(bottom), bottom[0], bottom[1]))
                start = None
        if start is not None and points[-1][0] - start >= args.min_sec:
            bottom = points[-1][1]
            rows.append((start, points[-1][0], tid, klass[tid], points[-1][2],
                         along(bottom), bottom[0], bottom[1]))
    rows.sort()
    for t0, t1, tid, name, lane, a, x, y in rows:
        print(f"{tid:6d} {name:>9} {str(lane):>5} {t0:10.2f} {t1:9.2f} {t1 - t0:6.1f} "
              f"{a:11.0f} {x:6.0f} {y:6.0f}")
    print(f"\n{len(rows)} stops >= {args.min_sec}s in lanes {sorted(lane_ids)}")
    if rows:
        upstream = [r for r in rows if r[5] < 0]
        print(f"{len(upstream)} are upstream of the stop line (candidate signal queues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
