"""Render what a rule decided on a specific frame, for manual checking.

Draws the tracks, the scene geometry, and the rule's own view of each track
(lane, signal colour, road-frame separation, stationarity), so a surprising
event can be traced to the quantity that caused it rather than guessed at.

Example::

    python scripts/inspect_rule_frame.py data/data_video2.mp4 --time 161.0 ^
        --jsonl debug/perception/data_video2_tracks.jsonl --rule stop_line
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.contracts import (  # noqa: E402
    FrameState,
    SceneState,
    TrackObservation,
    TrackState,
    TrafficLightState,
)
from src.rules.engine import RuleEngine  # noqa: E402
from src.rules.kinematics import road_frame_gap, swept_gap  # noqa: E402
from src.rules.motion import is_stationary  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.signals import classify_light_color  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402

VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--rule", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    scene = SceneContext(load_scene_config(config["scene"]["path"], width, height), width, height)
    engine = RuleEngine(config["rules"])
    manager = TrackManager(max_age_seconds=2.0, history_seconds=8.0)

    target = int(round(args.time * fps))
    # Replay to the target frame so velocity and history are correct.
    frame = None
    records: list[dict] = []
    with open(args.jsonl, encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            if record["frame"] <= target:
                records.append(record)
            else:
                break
    if not records:
        raise SystemExit("no track records at or before the requested time")

    for record in records:
        obs = [
            TrackObservation(track_id=t["track_id"], bbox=tuple(t["bbox"]),
                             score=t.get("confidence", 0.9), class_id=t["class_id"],
                             class_name=t["class_name"])
            for t in record["tracks"]
        ]
        tracks = manager.update(obs, record["timestamp"], record["frame"], scene)
        if record["frame"] == target:
            break

    light = scene.config.traffic_lights[0] if scene.config.traffic_lights else None
    color = "unknown"
    if light is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok and frame is not None:
            x0, y0, x1, y1 = (int(v) for v in light.roi)
            color, _ = classify_light_color(frame[y0:y1, x0:x1])
    cap.release()
    if frame is None:
        raise SystemExit("could not read the frame")

    if light is not None:
        scene.lights = {light.light_id: TrafficLightState(light.light_id, color, 1.0, light.roi)}
    state = FrameState(target, records[-1]["timestamp"], tracks,
                       SceneState(scene.scene_id, width, height, scene.lights, scene))
    signals = engine.collect_signals(state)

    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        lane = scene.lane_for_point(track.bottom_center)
        vehicle = track.class_name.lower() in VEHICLE_NAMES
        tint = (0, 200, 255) if vehicle else (255, 0, 255)
        if lane is None:
            tint = (140, 140, 140)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), tint, 2)
        tag = f"#{track.track_id} {track.class_name}"
        if lane is not None:
            tag += f" L{lane.lane_id}"
        else:
            tag += " noLane"
        tag += f" v{track.speed:.0f}"
        cv2.putText(canvas, tag, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(canvas, tag, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, tint, 1)
        cv2.circle(canvas, (int(track.bottom_center[0]), int(track.bottom_center[1])), 3,
                   (0, 255, 0), -1)
    for line in scene.config.stop_lines:
        (ax, ay), (bx, by) = line.segment
        cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (0, 0, 255), 4)
    for poly in scene.config.crossings:
        cv2.polylines(canvas, [__import__("numpy").array(poly, __import__("numpy").int32)],
                      True, (255, 255, 255), 2)
    header = f"t={args.time:.2f}s f={target} signal={color}"
    cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    print(f"t={args.time:.2f}s frame={target} signal={color} tracks={len(tracks)}")
    line = scene.config.stop_lines[0]
    print(f"\n{'track':>6} {'class':>11} {'lane':>5} {'speed':>7} {'front_along':>11} "
          f"{'in_xing':>8} {'stationary':>10}  flags")
    print("-" * 88)
    flagged = {
        int(s.evidence.get("track_id", -1))
        for items in signals.values() for s in items
        if isinstance(s.evidence.get("track_id"), int)
    } if not args.rule else {
        int(s.evidence.get("track_id", -1)) | int(s.evidence.get("vehicle_id", -1))
        | int(s.evidence.get("person_id", -1)) | int(s.evidence.get("first_id", -1))
        | int(s.evidence.get("second_id", -1))
        for s in signals.get(args.rule, [])
    }
    for track in sorted(tracks, key=lambda t: t.track_id):
        lane = scene.lane_for_point(track.bottom_center)
        from src.rules.kinematics import front_path
        fp = front_path(track, line.direction, 0.2, track.last_seen)
        along = float("nan")
        if fp:
            a = line.segment[0]
            along = (fp[-1][1] - a[0]) * line.direction[0] + (fp[-1][2] - a[1]) * line.direction[1]
        still = is_stationary(track.bottom_history, 2.5, 25.0, 250.0, state.timestamp)
        in_xing = scene.crossing_for_point(track.center) is not None
        mark = "  <== FLAGGED" if track.track_id in flagged else ""
        print(f"{track.track_id:6d} {track.class_name:>11} "
              f"{str(lane.lane_id) if lane else '-':>5} {track.speed:7.1f} {along:11.1f} "
              f"{str(in_xing):>8} {str(still):>10}{mark}")
    if args.rule:
        print(f"\n{args.rule} signals: {len(signals.get(args.rule, []))}")
        for s in signals.get(args.rule, []):
            print("  ", json.dumps(s.evidence, default=str))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        small = cv2.resize(canvas, (1600, int(1600 * height / width)))
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
