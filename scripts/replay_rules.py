"""Offline replay of the rule layer over a perception track dump.

Reconstructs TrackStates from a perception run and pushes them through the real
RuleEngine, so rule behaviour can be validated without re-running detection and
tracking. Signal state comes from the calibrated ROI, read by decoding the video
once at a coarse step.

The video's real resolution is read from the file, so the same scene file works
for 4K and 1080p sample videos.

Example::

    python scripts/replay_rules.py --jsonl debug/perception/data_video2_tracks.jsonl ^
        --video data/data_video2.mp4 --config configs/data_video1_rules_dev.json ^
        --out debug/rules/replay_video2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.contracts import FrameState, SceneState, TrackState, TrafficLightState  # noqa: E402
from src.perception.traffic_lights import TrafficLightPerception  # noqa: E402
from src.rules.engine import RuleEngine  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.signals import classify_light_color  # noqa: E402
from src.temporal.segmenter import TemporalSegmenter  # noqa: E402


def video_geometry(path: str) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frames


def signal_timeline(video: str, roi, step: int, fps: float) -> list[tuple[float, str]]:
    """Coarse (time, colour) samples of one configured signal ROI.

    The ROI is scaled into the video's own resolution by the caller, so this
    reads the lamp rather than an empty patch when the video is not 4K.
    """
    cap = cv2.VideoCapture(video)
    samples: list[tuple[float, str]] = []
    index = 0
    x0, y0, x1, y1 = (int(v) for v in roi)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            patch = frame[max(0, y0):y1, max(0, x0):x1]
            color, _ = classify_light_color(patch)
            samples.append((index / fps, color))
        index += 1
    cap.release()
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--video", default="data/data_video1.mp4")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--signal-step", type=int, default=20)
    parser.add_argument("--no-signal", action="store_true",
                        help="skip the video decode; every signal reads unknown")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    width, height, fps, frames = video_geometry(args.video)
    duration = frames / fps if fps else 0.0
    scene_config = load_scene_config(config["scene"]["path"], width, height)
    scene = SceneContext(scene_config, width, height)
    engine = RuleEngine(config["rules"])
    segmenter = TemporalSegmenter(config["temporal"], duration=duration)
    print(f"video {width}x{height} fps={fps:.3f} frames={frames} duration={duration:.1f}s")
    print(f"scene={scene_config.scene_id} has_geometry={scene.has_geometry} has_road={scene.has_road}")

    light = scene_config.traffic_lights[0] if scene_config.traffic_lights else None
    timeline: list[tuple[float, str]] = []
    if light is not None and not args.no_signal:
        print(f"reading signal '{light.light_id}' from calibrated ROI "
              f"{tuple(round(v) for v in light.roi)} (step={args.signal_step}) ...", flush=True)
        timeline = signal_timeline(args.video, light.roi, args.signal_step, fps)
        print(f"  {len(timeline)} signal samples", flush=True)

    def color_at(timestamp: float) -> str:
        best = "unknown"
        for t, color in timeline:
            if t <= timestamp + 1e-6:
                best = color
            else:
                break
        return best

    history: dict[int, dict[str, deque]] = defaultdict(
        lambda: {"center": deque(maxlen=300), "bottom": deque(maxlen=300), "bbox": deque(maxlen=300)}
    )
    state: dict[int, TrackState] = {}
    per_frame: dict[str, int] = defaultdict(int)
    first_seen: dict[tuple[str, int], float] = {}
    last_seen: dict[tuple[str, int], float] = {}
    sample_count = 0

    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        timestamp = record["timestamp"]
        if light is not None:
            color = color_at(timestamp)
            scene.lights = {light.light_id: TrafficLightState(light.light_id, color, 1.0, light.roi)}
        tracks: list[TrackState] = []
        for item in record["tracks"]:
            tid = item["track_id"]
            x1, y1, x2, y2 = item["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            bottom = tuple(item["bottom_center"])
            hist = history[tid]
            hist["center"].append((timestamp, center[0], center[1]))
            hist["bottom"].append((timestamp, bottom[0], bottom[1]))
            hist["bbox"].append((timestamp, x1, y1, x2, y2))
            track = state.get(tid)
            if track is None:
                track = TrackState(
                    track_id=tid, class_id=item["class_id"], class_name=item["class_name"],
                    bbox=item["bbox"], score=item["confidence"], first_seen=timestamp,
                    last_seen=timestamp, center=center, velocity=tuple(item.get("velocity", (0.0, 0.0))),
                    last_seen_frame=record["frame"], lane_id=item.get("lane_id"),
                )
                state[tid] = track
            else:
                track.bbox = item["bbox"]
                track.score = item["confidence"]
                track.class_id = item["class_id"]
                track.class_name = item["class_name"]
                track.center = center
                if "velocity" in item:
                    track.velocity = tuple(item["velocity"])
                track.last_seen = timestamp
                track.last_seen_frame = record["frame"]
                track.missed_frames = 0
                if item.get("lane_id") is not None and item["lane_id"] != track.lane_id:
                    track.lane_history.append(item["lane_id"])
                    track.lane_history = track.lane_history[-20:]
                track.lane_id = item.get("lane_id")
            track.history = deque(hist["center"], maxlen=300)
            track.bottom_history = deque(hist["bottom"], maxlen=300)
            track.bbox_history = deque(hist["bbox"], maxlen=300)
            tracks.append(track)

        frame_state = FrameState(
            record["frame"], timestamp, tracks,
            SceneState(scene.scene_id, width, height, scene.lights, scene),
        )
        signals = engine.collect_signals(frame_state)
        segmenter.add(timestamp, signals)
        for label, items in signals.items():
            for signal in items:
                per_frame[label] += 1
        for label, items in signals.items():
            for signal in items:
                tid = signal.evidence.get("track_id")
                if isinstance(tid, int):
                    k = (label, tid)
                    first_seen.setdefault(k, timestamp)
                    last_seen[k] = timestamp
        sample_count += 1

    events = segmenter.finalize(duration)
    print(f"sampled {sample_count} frames")
    print(f"per-frame candidate counts: {dict(sorted(per_frame.items()))}")
    print(f"final segments: {len(events)}")
    for event in events:
        print(f"  {str(event[2]):<20} {float(event[0]):8.2f} -> {float(event[1]):8.2f} "
              f"({float(event[1]) - float(event[0]):6.2f}s)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(events, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
