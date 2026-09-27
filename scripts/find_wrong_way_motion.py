"""Find sustained westbound vehicle motion and report where it happens.

Used to decide whether a ``wrong_way`` hit is genuine or an artefact of a lane
polygon that covers a road carrying traffic the other way.
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

from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.rules.motion import trailing_window  # noqa: E402
from src.scene.geometry import normalized_vector, vector_dot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--dot", type=float, default=-0.45)
    parser.add_argument("--min-travel", type=float, default=45.0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    config = load_scene_config(args.scene, args.width, args.height)
    scene = SceneContext(config, args.width, args.height)

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        if args.max_frames and record["frame"] > args.max_frames:
            break
        for track in record["tracks"]:
            tid = track["track_id"]
            klass[tid] = track["class_name"]
            series[tid].append((record["timestamp"], track["bottom_center"][0], track["bottom_center"][1]))

    vehicles = {"car", "truck", "bus", "motorcycle", "vehicle", "bicycle"}
    print(f"{'track':>6} {'class':<10} {'opp_windows':>12} {'lanes':<12} {'x_range':<16} {'y_range':<14} {'dot':>7}")
    rows = []
    for tid, points in series.items():
        if klass.get(tid) not in vehicles or len(points) < 30:
            continue
        hits, lanes, xs, ys, dots = 0, set(), [], [], []
        for i in range(len(points)):
            t_end = points[i][0]
            recent = trailing_window(points, args.window, t_end)
            if len(recent) < 2:
                continue
            vx = recent[-1][1] - recent[0][1]
            vy = recent[-1][2] - recent[0][2]
            travel = math.hypot(vx, vy)
            if travel < args.min_travel:
                continue
            lane = scene.lane_for_point((points[i][1], points[i][2]))
            if lane is None:
                continue
            unit = normalized_vector((vx, vy))
            expected = normalized_vector(lane.direction)
            if unit is None or expected is None:
                continue
            dot = vector_dot(expected, unit)
            if dot < args.dot:
                hits += 1
                lanes.add(lane.lane_id)
                xs.append(points[i][1])
                ys.append(points[i][2])
                dots.append(dot)
        if hits >= 10:
            rows.append((hits, tid, klass[tid], sorted(lanes), xs, ys, sum(dots) / len(dots)))
    rows.sort(reverse=True)
    for hits, tid, name, lanes, xs, ys, dot in rows:
        print(f"{tid:6d} {name:<10} {hits:12d} {str(lanes):<12} "
              f"{f'{int(min(xs))}-{int(max(xs))}':<16} {f'{int(min(ys))}-{int(max(ys))}':<14} {dot:7.3f}")
    print(f"\ntracks with sustained opposing motion: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
