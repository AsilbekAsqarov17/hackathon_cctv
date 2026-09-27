"""Trace one track's bottom-centre path with its assigned lane at each sample."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import SceneContext, load_scene_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--every", type=int, default=45)
    args = parser.parse_args()
    wanted = {int(v) for v in args.ids.split(",") if v.strip()}

    config = load_scene_config(args.scene, args.width, args.height)
    scene = SceneContext(config, args.width, args.height)
    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            if track["track_id"] in wanted:
                series[track["track_id"]].append(
                    (record["timestamp"], track["bottom_center"][0], track["bottom_center"][1])
                )
                klass[track["track_id"]] = track["class_name"]

    for tid in sorted(series):
        points = series[tid]
        print(f"=== track #{tid} ({klass.get(tid)}) frames={len(points)} "
              f"t={points[0][0]:.1f}..{points[-1][0]:.1f} ===")
        print(f"{'t':>8} {'x':>7} {'y':>7}  lane")
        for i in range(0, len(points), args.every):
            t, x, y = points[i]
            lane = scene.lane_for_point((x, y))
            name = f"{lane.lane_id}:{lane.name}" if lane else "none"
            print(f"{t:8.1f} {x:7.0f} {y:7.0f}  {name}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
