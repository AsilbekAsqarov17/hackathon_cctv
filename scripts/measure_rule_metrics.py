"""Report the distribution of rule metrics over real tracked traffic.

The accident / near_miss / jaywalking thresholds cannot be chosen by eye: in
dense traffic almost every metric is "large" at some point. This samples real
frames and prints the distribution of each quantity, so a threshold can be set
at a measured percentile instead of a guess, and the classes that cannot be
separated on this camera become visible rather than silently noisy.
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

from src.scene.geometry import bbox_iou, point_in_polygon  # noqa: E402
from src.config import load_config  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.rules.kinematics import (  # noqa: E402
    closing_speed,
    deceleration,
    swept_gap,
    time_to_collision,
)
from src.tracking.track_manager import TrackManager  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    scene_config = load_scene_config(config["scene"]["path"], args.width, args.height)
    scene = SceneContext(scene_config, args.width, args.height)

    gaps: list[float] = []
    ious: list[float] = []
    ttcs: list[float] = []
    closings: list[float] = []
    decels: list[float] = []
    person_road: list[tuple[float, float, float, int]] = []
    frame_index = 0
    for raw in open(args.jsonl, encoding="utf-8"):
        frame_index += 1
        if frame_index % args.step:
            continue
        record = json.loads(raw)
        tracks = record["tracks"]
        road_users = [
            t for t in tracks
            if TrackManager.is_vehicle_from_name(t["class_name"])
            or TrackManager.is_person_from_name(t["class_name"])
        ]
        for i, first in enumerate(road_users):
            for second in road_users[i + 1:]:
                first_ru = (TrackManager.is_vehicle_from_name(first["class_name"])
                            or TrackManager.is_person_from_name(first["class_name"]))
                second_ru = (TrackManager.is_vehicle_from_name(second["class_name"])
                             or TrackManager.is_person_from_name(second["class_name"]))
                if not (first_ru and second_ru):
                    continue
                a = _as_state(first)
                b = _as_state(second)
                gaps.append(swept_gap(a, b))
                ious.append(bbox_iou(a.bbox, b.bbox))
                ttc = time_to_collision(a, b)
                if ttc is not None and ttc < 10.0:
                    ttcs.append(ttc)
                closing = closing_speed(a, b)
                if math.isfinite(closing) and closing > 0:
                    closings.append(closing)
                decels.append(max(deceleration(a), deceleration(b)))
        for t in tracks:
            if TrackManager.is_person_from_name(t["class_name"]):
                bx, by = t["bottom_center"]
                person_road.append((
                    record["timestamp"], bx, by,
                    1 if scene.is_road_point((bx, by)) else 0,
                ))

    print(f"frames sampled: {frame_index // args.step} of {frame_index}")
    print(f"pairs: {len(gaps)}")
    for name, values in (("gap_px", gaps), ("iou", ious), ("ttc_s", ttcs),
                         ("closing_px_s", closings), ("decel", decels)):
        if not values:
            print(f"{name:>14}: no samples")
            continue
        print(f"{name:>14}: n={len(values):7d} "
              f"p50={percentile(values,0.5):9.2f} p90={percentile(values,0.9):9.2f} "
              f"p99={percentile(values,0.99):9.2f} max={max(values):9.2f}")
    print()
    print("Smallest gaps / largest IoUs (the pairs most likely to be false positives):")
    for value in sorted(gaps)[:10]:
        print(f"  gap={value:8.2f}")
    print()
    for value in sorted(ious, reverse=True)[:10]:
        print(f"  iou={value:8.3f}")
    on_road = [p for p in person_road if p[3]]
    print()
    print(f"pedestrian bottom-centre samples: {len(person_road)}; "
          f"on a road polygon: {len(on_road)} ({100*len(on_road)/max(1,len(person_road)):.1f}%)")
    return 0


def _as_state(item: dict):
    from src.contracts import TrackState
    from collections import deque

    return TrackState(
        track_id=item["track_id"], class_id=item["class_id"], class_name=item["class_name"],
        bbox=tuple(item["bbox"]), score=item.get("confidence", 0.9), first_seen=0.0, last_seen=0.0,
        center=tuple(item["center"]), velocity=tuple(item.get("velocity", (0.0, 0.0))),
        acceleration=tuple(item.get("acceleration", (0.0, 0.0))),
    )


if __name__ == "__main__":
    raise SystemExit(main())
