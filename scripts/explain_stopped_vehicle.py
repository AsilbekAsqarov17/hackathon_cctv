"""Explain why a specific track was (or was not) flagged as a signal queue.

Prints the track's position, lane, distance to and signed position relative to
each stop line, and whether the governing signal was red/yellow, at the moment
the stopped_vehicle rule fires for it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.contracts import FrameState, SceneState, TrackState, TrafficLightState  # noqa: E402
from src.rules.engine import RuleEngine  # noqa: E402
from src.rules.motion import is_stationary  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--ids", required=True, help="comma separated track ids")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--phases", default="", help="comma list of red start times")
    args = parser.parse_args()

    config = load_config(args.config)
    width, height = 3840, 2160
    scene_config = load_scene_config(config["scene"]["path"], width, height)
    scene = SceneContext(scene_config, width, height)
    engine = RuleEngine(config["rules"])
    rules = config["rules"]["stopped_vehicle"]
    wanted = {int(v) for v in args.ids.split(",") if v.strip()}
    # coarse signal phases: red windows as (start, end) pairs
    phases = []
    values = [float(v) for v in args.phases.split(",") if v.strip()]
    for i in range(0, len(values) - 1, 2):
        phases.append((values[i], values[i + 1]))

    def color_at(t: float) -> str:
        for start, end in phases:
            if start <= t <= end:
                return "red"
        return "green"

    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
    state: dict[int, TrackState] = {}
    reported: set[tuple[int, int]] = set()
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        timestamp = record["timestamp"]
        color = color_at(timestamp)
        lights = {"sig": TrafficLightState("sig", color, 1.0, (0, 0, 1, 1))}
        scene.lights = lights
        tracks = []
        for item in record["tracks"]:
            tid = item["track_id"]
            x1, y1, x2, y2 = item["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            bottom = tuple(item["bottom_center"])
            history[tid].append((timestamp, bottom[0], bottom[1]))
            track = state.get(tid)
            if track is None:
                track = TrackState(
                    track_id=tid, class_id=item["class_id"], class_name=item["class_name"],
                    bbox=item["bbox"], score=item["confidence"], first_seen=timestamp,
                    last_seen=timestamp, center=center, velocity=tuple(item["velocity"]),
                    history=deque(maxlen=300), last_seen_frame=record["frame"],
                )
                track.bottom_history = deque(history[tid], maxlen=300)
                state[tid] = track
            else:
                track.center = center
                track.bottom_history = deque(history[tid], maxlen=300)
                track.last_seen = timestamp
            tracks.append(track)

        frame_state = FrameState(record["frame"], timestamp, tracks,
                                 SceneState(scene.scene_id, width, height, lights, scene))
        for signal in engine._stopped_vehicle_candidates(frame_state):
            tid = int(signal.evidence["track_id"])
            if tid not in wanted:
                continue
            marker = (tid, int(timestamp))
            if marker in reported:
                continue
            reported.add(marker)
            track = next(t for t in tracks if t.track_id == tid)
            point = track.bottom_center
            lane = scene.lane_for_point(point)
            line = scene.line_for_track(track)
            info = {}
            if line is not None:
                info["line"] = line.line_id
                info["upstream_dist"] = round(engine._upstream_distance(point, line), 1)
                info["queued"] = engine._queued_at_signal(track, frame_state)
            info["on_road"] = engine._road_track(track, scene)
            print(
                f"t={timestamp:7.2f} #{tid:<5} {track.class_name:<9} "
                f"bottom=({point[0]:7.1f},{point[1]:7.1f}) lane={lane.lane_id if lane else None} "
                f"light={color:7} held={signal.evidence['held_sec']:6.2f} {info}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
