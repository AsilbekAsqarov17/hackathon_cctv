"""Per-lane breakdown of opposing traffic, to decide where wrong_way is safe.

For each calibrated lane this counts distinct vehicle tracks that show sustained
motion against the lane's declared direction, and reports how large that
opposing flow is. A lane carrying a steady opposing flow cannot have a
single-direction wrong_way rule enabled on it.
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
    opposing: dict[int, set] = defaultdict(set)
    with_traffic: dict[int, set] = defaultdict(set)
    for tid, points in series.items():
        if klass.get(tid) not in vehicles or len(points) < 30:
            continue
        hits: dict[int, int] = defaultdict(int)
        for i in range(len(points)):
            t_end = points[i][0]
            recent = trailing_window(points, args.window, t_end)
            if len(recent) < 2:
                continue
            vx = recent[-1][1] - recent[0][1]
            vy = recent[-1][2] - recent[0][2]
            if math.hypot(vx, vy) < args.min_travel:
                continue
            lane = scene.lane_for_point((points[i][1], points[i][2]))
            if lane is None:
                continue
            unit = normalized_vector((vx, vy))
            expected = normalized_vector(lane.direction)
            if unit is None or expected is None:
                continue
            hits[lane.lane_id] += 1 if vector_dot(expected, unit) < args.dot else 0
        for lane_id, count in hits.items():
            with_traffic[lane_id].add(tid)
            if count >= args.min_hits:
                opposing[lane_id].add(tid)

    print(f"{'lane':>5} {'dir':>14} {'tracks_moving':>14} {'opposing':>10} {'opposing_pct':>13} {'verdict'}")
    for lane in config.lanes:
        lid = lane.lane_id
        total = len(with_traffic[lid])
        opp = len(opposing[lid])
        pct = 100.0 * opp / total if total else 0.0
        verdict = "SAFE" if pct < 2.0 else ("UNUSABLE" if pct > 15 else "SUSPECT")
        print(f"{lid:5d} {str(tuple(round(v,3) for v in lane.direction)):>14} "
              f"{total:14d} {opp:10d} {pct:12.1f}% {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
