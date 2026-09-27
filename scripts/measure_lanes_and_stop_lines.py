#!/usr/bin/env python3
"""
measure_lanes_and_stop_lines.py - derive lane centres and stop lines from
observed traffic, rather than drawing them.

Five rules (wrong_way, illegal_turn, illegal_u_turn, solid_line_crossing,
stop_line) and red_light are inert in the shipped scene because `lanes`,
`stop_lines` and `solid_lines` are all empty. They are inert for want of
geometry, not because their thresholds are strict. Hand-drawing that geometry
on a 4K view of a real junction is guesswork, and a lane in the wrong place is
worse than no lane: it manufactures wrong_way violations out of correct
manoeuvres.

Two things here are directly measurable, so they are measured:

  lanes      A lane is where vehicles actually drive. Moving vehicle ground
             contacts are grouped by heading, and the distribution of their
             lateral offset within each group is a set of peaks. One lane per
             peak.

  stop lines A stop line is where a queue actually halts. Within each heading
             group, the furthest-forward ground contact belonging to a vehicle
             that decelerated to a stand still and stayed there marks the head
             of the queue. Cross-checked against the traffic-light reader, so a
             breakdown in a live lane is not mistaken for a signal.

Nothing is written to configs/. This reports measurements and an overlay so
the numbers can be checked against the footage before anything is adopted.

    python scripts/measure_lanes_and_stop_lines.py data/samples_small
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import apply_environment_overrides, load_config  # noqa: E402
from src.perception.detector import build_detector  # noqa: E402
from src.perception.tracker import build_tracker  # noqa: E402
from src.scene.signals import TrafficLightReader  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402

MOVING_MPS = 2.0      # above this a vehicle is genuinely travelling, not queued
STOPPED_MPS = 0.6     # below this it counts as halted
STOP_HOLD_S = 1.5     # and it must stay halted this long to be a queue, not a pause
LANE_WIDTH_M = 3.5    # used only to reject peaks too close together to be lanes
MIN_LANE_OBS = 40     # observations needed before a peak is called a lane
MIN_STOP_OBS = 3      # distinct vehicles halted at a point before it is a line


def collect(video: str, detector, tracker_config: dict, scene_cfg, stride: int,
            max_frames: int) -> list[dict]:
    """Per-vehicle observations: ground contact, heading, speed, light state."""
    tracker = build_tracker({**tracker_config, "backend": "simple_bytetrack"})
    manager = TrackManager(max_age_seconds=2.0)
    reader = TrafficLightReader(scene_cfg)
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    observations: list[dict] = []
    track_first_stop: dict[int, float] = {}
    frame_id = 0
    try:
        while frame_id < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % stride:
                frame_id += 1
                continue
            timestamp = frame_id / fps
            detections = detector.predict(frame)
            seen = tracker.update(detections, frame_id, timestamp, frame)
            lights = reader.update(frame, detections)
            red = any(getattr(s, "color", "unknown") == "red" for s in lights.values())
            for state in manager.update(seen, timestamp, frame_id, None):
                if not TrackManager.is_vehicle(state):
                    continue
                speed = state.speed_mps
                if speed is None:
                    continue
                vx, vy = state.velocity
                norm = math.hypot(vx, vy)
                if norm < 1e-3:
                    continue
                if speed <= STOPPED_MPS:
                    track_first_stop.setdefault(state.track_id, timestamp)
                else:
                    track_first_stop.pop(state.track_id, None)
                held = None
                if state.track_id in track_first_stop:
                    held = timestamp - track_first_stop[state.track_id]
                observations.append(
                    {
                        "t": timestamp,
                        "frame": frame_id,
                        "track": state.track_id,
                        "x": state.bottom_center[0] / width,
                        "y": state.bottom_center[1] / height,
                        "ux": vx / norm,
                        "uy": vy / norm,
                        "speed": speed,
                        "scale": state.metres_per_pixel or 0.0,
                        "red": red,
                        "held": held,
                    }
                )
            frame_id += 1
    finally:
        cap.release()
    return observations


def group_by_heading(observations: list[dict], tolerance_deg: float = 30.0) -> list[list[dict]]:
    """Cluster observations by direction of travel.

    Heading alone is not enough at a junction -- a vehicle turning and one going
    straight share a bearing for a few frames -- so the clustering is on the
    unit velocity vector, which for a tracked vehicle is stable over its life.
    """
    groups: list[list[dict]] = []
    for obs in observations:
        angle = math.degrees(math.atan2(obs["uy"], obs["ux"]))
        placed = False
        for group in groups:
            other = group[0]
            delta = abs((angle - other["_angle"] + 180.0) % 360.0 - 180.0)
            if delta <= tolerance_deg:
                group.append(obs)
                placed = True
                break
        if not placed:
            obs = dict(obs)
            obs["_angle"] = angle
            groups.append([obs])
    return groups


def find_lane_peaks(group: list[dict], scale_m_per_px: float) -> dict:
    """Lateral-offset peaks within one heading group are the lane centres."""
    moving = [o for o in group if o["speed"] >= MOVING_MPS]
    if len(moving) < MIN_LANE_OBS:
        return {"moving": len(moving), "peaks": []}
    ux = float(np.mean([o["ux"] for o in moving]))
    uy = float(np.mean([o["uy"] for o in moving]))
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    # Perpendicular axis: signed lateral offset from the group's centre line.
    offsets = [-o["ux"] * 0 + (-uy * o["x"] + ux * o["y"]) for o in moving]
    along = [ux * o["x"] + uy * o["y"] for o in moving]
    lo, hi = min(offsets), max(offsets)
    if hi - lo < 1e-6:
        return {"moving": len(moving), "peaks": []}
    bins = np.linspace(lo, hi, 61)
    hist, edges = np.histogram(offsets, bins=bins)
    min_gap_px = LANE_WIDTH_M / max(scale_m_per_px, 1e-6) if scale_m_per_px else 0.0
    peaks: list[float] = []
    for i, count in enumerate(hist):
        if count < max(3, MIN_LANE_OBS // 12):
            continue
        centre = 0.5 * (edges[i] + edges[i + 1])
        if all(abs(centre - p) > min_gap_px for p in peaks):
            peaks.append(centre)
    lanes = []
    for peak in peaks:
        near = [o for o, off in zip(moving, offsets) if abs(off - peak) <= min_gap_px / 2]
        if len(near) < MIN_LANE_OBS:
            continue
        lanes.append(
            {
                "offset": peak,
                "n": len(near),
                "mean_speed": float(np.mean([o["speed"] for o in near])),
                "along_min": min(ux * o["x"] + uy * o["y"] for o in near),
                "along_max": max(ux * o["x"] + uy * o["y"] for o in near),
            }
        )
    return {"moving": len(moving), "direction": (ux, uy), "peaks": lanes}


def find_stop_lines(group: list[dict], scale_m_per_px: float) -> list[dict]:
    """The head of a halted queue marks the stop line for that approach."""
    halted = [
        o
        for o in group
        if o["held"] is not None and o["held"] >= STOP_HOLD_S and o["red"]
    ]
    if not halted:
        return []
    ux = float(np.mean([o["ux"] for o in group]))
    uy = float(np.mean([o["uy"] for o in group]))
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    projections = [(ux * o["x"] + uy * o["y"], o) for o in halted]
    front = max(p for p, _ in projections)
    tolerance = (LANE_WIDTH_M / max(scale_m_per_px, 1e-6)) * 0.75 if scale_m_per_px else 0.02
    at_front = [o for p, o in projections if p >= front - tolerance]
    if len({o["track"] for o in at_front}) < MIN_STOP_OBS:
        return []
    return [
        {
            "along": front,
            "vehicles": sorted({o["track"] for o in at_front}),
            "n_observations": len(at_front),
            "all_red": all(o["red"] for o in at_front),
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--frames", type=int, default=2400)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--out", default="data/inspection/measured_geometry.json")
    args = parser.parse_args()

    config = apply_environment_overrides(load_config(str(ROOT / "configs" / "default.json")))
    detector = build_detector(config["detector"])
    scene_cfg = json.loads((ROOT / "configs" / "scenes" / "tashkent_intersection.json").read_text())

    report: dict = {"videos": {}, "settings": {
        "moving_mps": MOVING_MPS, "stopped_mps": STOPPED_MPS,
        "stop_hold_s": STOP_HOLD_S, "lane_width_m": LANE_WIDTH_M,
    }}
    for video in args.videos:
        path = video if Path(video).exists() else str(ROOT / "data" / "samples_small" / video)
        if not Path(path).exists():
            print(f"skip {video}: not found", file=sys.stderr)
            continue
        observations = collect(path, detector, config["tracker"], scene_cfg,
                               args.stride, args.frames)
        if not observations:
            print(f"{Path(path).name}: no usable observations", file=sys.stderr)
            continue
        scales = [o["scale"] for o in observations if o["scale"]]
        scale = float(np.median(scales)) if scales else 0.0
        groups = group_by_heading(observations)
        lanes: list[dict] = []
        stops: list[dict] = []
        for group in groups:
            found = find_lane_peaks(group, scale)
            if found["peaks"]:
                ux, uy = found["direction"]
                for lane in found["peaks"]:
                    lanes.append({"heading_deg": round(math.degrees(math.atan2(uy, ux)), 1),
                                  "direction": (round(ux, 4), round(uy, 4)), **lane})
            stops.extend(find_stop_lines(group, scale))
        report["videos"][Path(path).name] = {
            "observations": len(observations),
            "m_per_px_median": round(scale, 6),
            "heading_groups": len(groups),
            "group_sizes": sorted((len(g) for g in groups), reverse=True),
            "lane_candidates": lanes,
            "stop_line_candidates": stops,
        }
        name = Path(path).name
        print(f"\n=== {name}: {len(observations)} vehicle observations, "
              f"{len(groups)} heading groups, {scale:.5f} m/px ===")
        for group in sorted(groups, key=len, reverse=True)[:4]:
            found = find_lane_peaks(group, scale)
            heading = math.degrees(math.atan2(found.get("direction", (0, 1))[1],
                                              found.get("direction", (1, 0))[0]))
            print(f"  group n={len(group):5d} moving={found['moving']:5d} "
                  f"heading={heading:6.1f}deg lanes={len(found['peaks'])}")
            for lane in found["peaks"]:
                print(f"      lane offset={lane['offset']:+.4f} n={lane['n']:4d} "
                      f"mean={lane['mean_speed']:.1f} m/s")
        for stop in stops:
            print(f"  STOP LINE along={stop['along']:.4f} vehicles={stop['vehicles']} "
                  f"n_obs={stop['n_observations']} all_red={stop['all_red']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
