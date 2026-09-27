"""Measure the real distribution of the interaction metrics on tracked traffic.

The earlier percentile pass fed zeroed velocities, so its `decel` and `swerve`
numbers were meaningless. This replays the track dump through the real
``TrackManager`` so velocity and acceleration are the ones the pipeline would
actually see, then reports the distribution of every quantity the accident and
near-miss rules test.

The point is to place each threshold above the noise: a rule whose threshold
sits inside the bulk of real traffic will fire on every queue.
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

from src.config import load_config  # noqa: E402
from src.contracts import TrackState  # noqa: E402
from src.rules.kinematics import (  # noqa: E402
    closing_speed,
    deceleration,
    swept_gap,
    time_to_collision,
)
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.geometry import bbox_iou  # noqa: E402
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
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    scene = SceneContext(load_scene_config(config["scene"]["path"], args.width, args.height),
                         args.width, args.height)
    manager = TrackManager(max_age_seconds=2.0, history_seconds=8.0)

    gaps: list[float] = []
    ious: list[float] = []
    ttcs: list[float] = []
    decels: list[float] = []
    shared_gaps: list[float] = []
    shared_decels: list[float] = []
    close_decels: list[float] = []
    close_ttcs: list[float] = []
    index = 0
    for raw in open(args.jsonl, encoding="utf-8"):
        record = json.loads(raw)
        index += 1
        if index % args.step:
            # Still feed the manager so velocity stays continuous.
            _feed(manager, record)
            continue
        tracks = _feed(manager, record)
        road_users = [t for t in tracks if TrackManager.is_road_user(t)]
        for i, first in enumerate(road_users):
            for second in road_users[i + 1:]:
                if not (TrackManager.is_vehicle(first) or TrackManager.is_person(first)):
                    continue
                if not (TrackManager.is_vehicle(second) or TrackManager.is_person(second)):
                    continue
                gap = swept_gap(first, second)
                gaps.append(gap)
                ious.append(bbox_iou(first.bbox, second.bbox))
                ttc = time_to_collision(first, second)
                if ttc is not None and ttc < 10.0:
                    ttcs.append(ttc)
                decels.append(max(deceleration(first), deceleration(second)))
                lane_a = scene.lane_for_point(first.bottom_center)
                lane_b = scene.lane_for_point(second.bottom_center)
                if lane_a is not None and lane_b is not None and lane_a.lane_id == lane_b.lane_id:
                    shared_gaps.append(gap)
                    shared_decels.append(max(deceleration(first), deceleration(second)))
                # The population that actually matters for the rules: pairs
                # close enough, and with a finite TTC, to be a candidate.
                if gap <= 140.0 and ttc is not None and ttc <= 2.5:
                    close_decels.append(max(deceleration(first), deceleration(second)))
                    close_ttcs.append(ttc)
    print(f"frames: {index}  sampled every {args.step}")
    for name, values in (("gap_px", gaps), ("iou", ious), ("ttc_s", ttcs), ("decel", decels)):
        if not values:
            continue
        print(f"{name:>10}: n={len(values):8d} p50={percentile(values,0.5):9.2f} "
              f"p90={percentile(values,0.9):9.2f} p99={percentile(values,0.99):9.2f} "
              f"p999={percentile(values,0.999):9.2f} max={max(values):9.2f}")
    if shared_gaps:
        print(f"\nsame-lane pairs only ({len(shared_gaps)}):")
        print(f"{'gap_px':>10}: p1={percentile(shared_gaps,0.01):8.1f} "
              f"p5={percentile(shared_gaps,0.05):8.1f} p50={percentile(shared_gaps,0.5):8.1f}")
        print(f"{'decel':>10}: p50={percentile(shared_decels,0.5):8.1f} "
              f"p90={percentile(shared_decels,0.9):8.1f} p99={percentile(shared_decels,0.99):8.1f} "
              f"p999={percentile(shared_decels,0.999):8.1f} max={max(shared_decels):8.1f}")
    if close_decels:
        print(f"\nCANDIDATE population (gap<=140px AND ttc<=2.5s): n={len(close_decels)}")
        print("  this is what a near_miss threshold is actually applied to:")
        for q in (0.5, 0.9, 0.95, 0.99, 0.999):
            print(f"    decel p{q*100:g} = {percentile(close_decels, q):9.1f}")
        print(f"    decel max  = {max(close_decels):9.1f}")
        print(f"    ttc   max  = {max(close_ttcs):9.2f}")
    return 0


def _feed(manager: TrackManager, record: dict) -> list[TrackState]:
    from src.contracts import TrackObservation

    observations = [
        TrackObservation(
            track_id=t["track_id"], bbox=tuple(t["bbox"]), score=t.get("confidence", 0.9),
            class_id=t["class_id"], class_name=t["class_name"],
        )
        for t in record["tracks"]
    ]
    return manager.update(observations, record["timestamp"], record["frame"], None)


if __name__ == "__main__":
    raise SystemExit(main())
