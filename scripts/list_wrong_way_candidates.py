"""List the tracks that oppose their lane direction, for manual inspection.

For every track that shows sustained motion against its assigned lane this
reports the lane, the dot product, the travel distance, the window, the lane
change history and the track's lane tenure, so a candidate wrong_way event can
be judged as a real violation or as a turning / boundary artefact.

Example::

    python scripts/list_wrong_way_candidates.py debug/perception/data_video1_tracks.jsonl
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
    parser.add_argument("--min-hits", type=int, default=10)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    config = load_scene_config(args.scene, args.width, args.height)
    scene = SceneContext(config, args.width, args.height)

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            tid = track["track_id"]
            klass[tid] = track["class_name"]
            series[tid].append((record["timestamp"], track["bottom_center"][0], track["bottom_center"][1]))

    vehicles = {"car", "truck", "bus", "motorcycle", "vehicle", "bicycle"}
    rows = []
    for tid, points in series.items():
        if klass.get(tid) not in vehicles or len(points) < 30:
            continue
        hits: dict[int, int] = defaultdict(int)
        worst: dict[int, tuple] = {}
        lanes_seen: dict[int, set] = defaultdict(set)
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
            lanes_seen[lane.lane_id].add(tid)
            unit = normalized_vector((vx, vy))
            expected = normalized_vector(lane.direction)
            if unit is None or expected is None:
                continue
            dot = vector_dot(expected, unit)
            if dot < args.dot:
                hits[lane.lane_id] += 1
                # turning indicator: how horizontal the motion is
                horiz = abs(vx) / max(1e-9, abs(vy)) if abs(vy) > 1e-9 else 99.0
                if lane.lane_id not in worst or dot < worst[lane.lane_id][0]:
                    worst[lane.lane_id] = (dot, travel, round(t_end, 2), round(horiz, 2),
                                           int(points[i][1]), int(points[i][2]))
        for lane_id, count in hits.items():
            if count >= args.min_hits:
                dot, travel, t_end, horiz, x, y = worst[lane_id]
                rows.append((count, tid, klass[tid], lane_id, dot, travel, t_end, horiz, x, y,
                             len(lanes_seen)))
    rows.sort(reverse=True)
    print(f"tracks with >= {args.min_hits} opposing windows: {len(rows)}")
    print(f"{'hits':>5} {'trk':>5} {'class':<9} {'lane':>4} {'dot':>7} {'trav':>7} {'t_end':>8} "
          f"{'|vx|/|vy|':>9} {'x':>6} {'y':>6} {'lanes':>5}")
    print("-" * 92)
    for (count, tid, name, lane_id, dot, travel, t_end, horiz, x, y, nlanes) in rows[: args.top]:
        print(f"{count:5d} {tid:5d} {name:<9} {lane_id:4d} {dot:7.3f} {travel:7.0f} {t_end:8.2f} "
              f"{horiz:9.2f} {x:6d} {y:6d} {nlanes:5d}")
    turning = sum(1 for r in rows if r[7] < 2.0)
    print(f"\n{turning}/{len(rows)} have |vx|/|vy| < 2.0 (consistent with turning, not a traverse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
