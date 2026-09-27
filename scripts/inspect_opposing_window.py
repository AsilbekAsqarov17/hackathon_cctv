"""Print every motion window of one track that a directional rule would flag.

Used to tell a genuine wrong-way traverse from a turn, a frame-boundary
artefact, a track-birth two-point window or detector jitter.
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
    parser.add_argument("--ids", required=True)
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--dot", type=float, default=-0.45)
    parser.add_argument("--min-travel", type=float, default=45.0)
    args = parser.parse_args()
    wanted = {int(v) for v in args.ids.split(",") if v.strip()}

    config = load_scene_config(args.scene, args.width, args.height)
    scene = SceneContext(config, args.width, args.height)

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            tid = track["track_id"]
            if tid in wanted:
                series[tid].append((record["timestamp"], track["bottom_center"][0],
                                    track["bottom_center"][1], track.get("speed", 0.0)))
                klass[tid] = track["class_name"]

    for tid in sorted(series):
        points = series[tid]
        print(f"=== track #{tid} ({klass.get(tid)}) n={len(points)} "
              f"t={points[0][0]:.2f}..{points[-1][0]:.2f} ===")
        print(f"{'t_end':>8} {'nsamp':>5} {'span':>6} {'trav':>7} {'speed':>7} {'dot':>7} "
              f"{'x':>6} {'y':>6}  lane  flag")
        flagged = 0
        for i in range(len(points)):
            t_end = points[i][0]
            recent = trailing_window(points, args.window, t_end)
            if len(recent) < 2:
                continue
            lane = scene.lane_for_point((points[i][1], points[i][2]))
            if lane is None:
                continue
            span = recent[-1][0] - recent[0][0]
            vx = recent[-1][1] - recent[0][1]
            vy = recent[-1][2] - recent[0][2]
            travel = math.hypot(vx, vy)
            unit = normalized_vector((vx, vy))
            expected = normalized_vector(lane.direction)
            if unit is None or expected is None:
                continue
            dot = vector_dot(expected, unit)
            flag = ""
            if dot < args.dot and travel >= args.min_travel and points[i][3] >= 6.0:
                flag = "OPPOSING"
                flagged += 1
            if not flag and dot >= args.dot and flagged:
                break
            if flag or (flagged and flagged <= 40):
                print(f"{t_end:8.2f} {len(recent):5d} {span:6.2f} {travel:7.0f} "
                      f"{points[i][3]:7.1f} {dot:7.3f} {points[i][1]:6.0f} {points[i][2]:6.0f}  "
                      f"{lane.lane_id}:{lane.name}  {flag}")
        print(f"  -> {flagged} fully-guarded opposing windows "
              f"(dot<{args.dot}, travel>={args.min_travel}px, speed>=6.0)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
