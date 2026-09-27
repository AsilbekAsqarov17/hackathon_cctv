"""Run hybrid detection + ByteTrack over a video and export tracks.

This is the perception/tracking verification entry point. It produces:

- an annotated MP4 (box, track id, class, confidence, bottom-center trail,
  timestamp) for visual inspection of id stability and trajectory shape;
- a JSONL file with one record per sampled frame, holding every field the
  milestone requires per track: id, class, box, confidence, timestamp,
  bottom-center and trajectory history length;
- a JSON summary with track counts, id-persistence statistics and timings.

Event rules are intentionally not evaluated here; scene geometry has to be
confirmed first (see data/scene_calibration.md).

Example::

    python scripts/run_perception.py --video data/data_video1.mp4 ^
        --config configs/data_video1_hybrid.json ^
        --out-video debug/perception/tracks.mp4 ^
        --out-jsonl debug/perception/tracks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import Detection
from src.perception.detector import build_detector
from src.perception.tracker import build_tracker
from src.perception.traffic_lights import is_traffic_light
from src.scene.config import SceneContext, load_scene_config
from src.tracking.track_manager import TrackManager

# Distinct colours so overlapping classes stay readable in the overlay.
CLASS_COLORS = {
    "car": (80, 200, 255),
    "truck": (60, 140, 255),
    "bus": (40, 90, 255),
    "motorcycle": (255, 160, 60),
    "bicycle": (255, 200, 60),
    "person": (120, 255, 120),
}
DEFAULT_COLOR = (200, 200, 200)
LIGHT_COLOR = (60, 60, 255)
TRAIL_LENGTH = 40


def _class_color(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name.lower(), DEFAULT_COLOR)


def _draw_tracks(
    canvas: Any,
    states: list[Any],
    lights: list[Any],
    timestamp: float,
    scale: float,
    fps: float,
) -> None:
    """Draw the overlay.

    ``states``/``lights`` carry source-resolution (4K) pixel coordinates while
    ``canvas`` may be a downscaled copy, so every coordinate is multiplied by
    ``scale`` before drawing.
    """
    def sx(value: float) -> int:
        return int(round(value * scale))

    def sy(value: float) -> int:
        return int(round(value * scale))

    thickness = max(1, int(round(3 * scale)))
    font = max(0.35, 0.9 * scale)
    for state in states:
        x1, y1, x2, y2 = [sx(v) for v in state.bbox]
        color = _class_color(state.class_name)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        # Bottom-center is the primary road position: mark it explicitly.
        bx, by = sx(state.bottom_center[0]), sy(state.bottom_center[1])
        radius = max(3, int(round(7 * scale)))
        cv2.circle(canvas, (bx, by), radius, (0, 255, 255), -1)
        cv2.circle(canvas, (bx, by), radius, (0, 0, 0), 2)
        trail = list(state.bottom_history)[-TRAIL_LENGTH:]
        for i in range(1, len(trail)):
            p0 = (sx(trail[i - 1][1]), sy(trail[i - 1][2]))
            p1 = (sx(trail[i][1]), sy(trail[i][2]))
            fade = 0.35 + 0.65 * (i / max(1, len(trail) - 1))
            cv2.line(canvas, p0, p1, tuple(int(c * fade) for c in color), thickness)
        label = f"#{state.track_id} {state.class_name} {state.score:.2f}"
        ty = max(int(round(24 * scale)), y1 - int(round(10 * scale)))
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thickness + 2)
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font, color, max(1, thickness - 1))
    for light in lights:
        x1, y1, x2, y2 = [sx(v) for v in light.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), LIGHT_COLOR, max(1, thickness - 1))
        text = f"{light.light_id}:{light.color} {light.color_confidence:.2f}"
        ly = max(int(round(20 * scale)), y1 - int(round(8 * scale)))
        cv2.putText(canvas, text, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, (0, 0, 0), thickness + 1)
        cv2.putText(canvas, text, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, LIGHT_COLOR, max(1, thickness - 1))
    header = f"t={timestamp:7.2f}s  frame={int(timestamp * fps)}  tracks={len(states)}"
    hfont = max(0.5, 1.2 * scale)
    cv2.putText(canvas, header, (sx(24), sy(48)), cv2.FONT_HERSHEY_SIMPLEX, hfont, (0, 0, 0), thickness + 3)
    cv2.putText(canvas, header, (sx(24), sy(48)), cv2.FONT_HERSHEY_SIMPLEX, hfont, (255, 255, 255), thickness)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/data_video1.mp4")
    parser.add_argument("--config", default="configs/data_video1_hybrid.json")
    parser.add_argument("--out-video", default="")
    parser.add_argument("--out-jsonl", default="debug/perception/tracks.jsonl")
    parser.add_argument("--out-summary", default="debug/perception/summary.json")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out-width", type=int, default=1920, help="0 keeps source width")
    parser.add_argument("--scene", default="", help="override scene config path")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    detector = build_detector(config)
    tracker = build_tracker(config.get("tracker", {}))
    manager = TrackManager(
        max_age_seconds=float(config.get("track_manager", {}).get("max_age_seconds", 2.0)),
        history_seconds=float(config.get("track_manager", {}).get("history_seconds", 8.0)),
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    scene_path = args.scene or config.get("scene", {}).get("path", "")
    scene_config = load_scene_config(scene_path, width, height) if scene_path else None
    scene = SceneContext(scene_config, width, height)
    print(f"video={args.video} {width}x{height} fps={fps:.3f} frames={total_frames}")
    print(f"detector={type(detector).__name__} tracker={type(tracker).__name__} scene={scene.scene_id}")

    out_scale = 1.0
    writer = None
    if args.out_video and not args.no_video:
        Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
        out_w = args.out_width if args.out_width > 0 else width
        out_h = int(round(height * out_w / width))
        out_scale = out_w / width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, fps / max(1, args.stride), (out_w, out_h))
        print(f"writing annotated video -> {args.out_video} ({out_w}x{out_h})")

    jsonl_path = Path(args.out_jsonl) if args.out_jsonl else None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")
    else:
        jsonl_file = None

    seen_frames = Counter()
    track_first: dict[int, float] = {}
    track_last: dict[int, float] = {}
    track_class: dict[int, str] = {}
    light_totals: Counter = Counter()
    light_ids: set[str] = set()
    processed = 0
    started = time.perf_counter()
    analyze = getattr(detector, "analyze", None)

    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.stride:
            continue
        timestamp = frame_index / fps
        if args.max_frames and processed >= args.max_frames:
            break
        processed += 1

        if analyze is not None:
            detections, light_observations = analyze(frame, frame_id=frame_index)
        else:
            detections = detector.predict(frame)
            reader = getattr(detector, "read_traffic_lights", None)
            light_observations = reader(frame, frame_index) if reader else []

        # Traffic lights are static scene elements, not road users, so they are
        # deliberately excluded from tracking and kept as their own channel.
        trackable: list[Detection] = [d for d in detections if not is_traffic_light(d.class_name)]
        observations = tracker.update(trackable, frame_index, timestamp, frame=frame)
        states = manager.update(observations, timestamp, frame_index, scene=scene)

        for state in states:
            seen_frames[state.track_id] += 1
            track_first.setdefault(state.track_id, timestamp)
            track_last[state.track_id] = timestamp
            track_class[state.track_id] = state.class_name
        for light in light_observations:
            light_ids.add(light.light_id)
            light_totals[light.color] += 1

        if writer is not None:
            canvas = frame
            if out_scale != 1.0:
                canvas = cv2.resize(frame, (int(width * out_scale), int(height * out_scale)))
            _draw_tracks(canvas, states, light_observations, timestamp, out_scale, fps)
            writer.write(canvas)

        if jsonl_file is not None:
            record = {
                "frame": frame_index,
                "timestamp": round(timestamp, 4),
                "tracks": [
                    {
                        "track_id": state.track_id,
                        "class_id": state.class_id,
                        "class_name": state.class_name,
                        "bbox": [round(float(v), 1) for v in state.bbox],
                        "confidence": round(float(state.score), 4),
                        "bottom_center": [round(float(v), 1) for v in state.bottom_center],
                        "center": [round(float(v), 1) for v in state.center],
                        "velocity": [round(float(v), 2) for v in state.velocity],
                        "speed": round(state.speed, 2),
                        "history_len": len(state.history),
                        "bottom_history_len": len(state.bottom_history),
                        "lane_id": state.lane_id,
                    }
                    for state in states
                ],
                "traffic_lights": [
                    {
                        "light_id": light.light_id,
                        "bbox": [round(float(v), 1) for v in light.bbox],
                        "score": round(float(light.score), 4),
                        "color": light.color,
                        "color_confidence": round(float(light.color_confidence), 4),
                        "color_history": [[int(f), c] for f, c in light.history],
                    }
                    for light in light_observations
                ],
            }
            jsonl_file.write(json.dumps(record) + "\n")

    cap.release()
    if writer is not None:
        writer.release()
    if jsonl_file is not None:
        jsonl_file.close()
    elapsed = time.perf_counter() - started

    # ------------------------------------------------------------------
    # id persistence: a long track seen in most sampled frames is stable;
    # many very short tracks indicate fragmentation.
    # ------------------------------------------------------------------
    spans = {
        track_id: round(track_last[track_id] - track_first[track_id], 3)
        for track_id in track_first
    }
    lengths = sorted(seen_frames.values(), reverse=True)
    total_track_frames = sum(lengths)
    by_class = Counter(track_class.values())
    long_tracks = sum(1 for track_id, count in seen_frames.items() if count >= 10)
    summary = {
        "video": args.video,
        "config": args.config,
        "detector": type(detector).__name__,
        "tracker": type(tracker).__name__,
        "scene": scene.scene_id,
        "has_geometry": scene.has_geometry,
        "fps": round(fps, 3),
        "frames_processed": processed,
        "stride": args.stride,
        "wall_seconds": round(elapsed, 2),
        "processing_fps": round(processed / elapsed, 2) if elapsed > 0 else None,
        "video_fps_equivalent": round(fps / args.stride, 2),
        "unique_track_ids": len(seen_frames),
        "tracks_by_class": dict(by_class.most_common()),
        "longest_track_frames": lengths[0] if lengths else 0,
        "median_track_frames": lengths[len(lengths) // 2] if lengths else 0,
        "tracks_seen_ge_10_frames": long_tracks,
        "track_frame_occupancy": round(total_track_frames / processed, 2) if processed else 0,
        "mean_track_span_sec": round(sum(spans.values()) / len(spans), 2) if spans else 0,
        "max_track_span_sec": round(max(spans.values()), 2) if spans else 0,
        "traffic_light_ids": sorted(light_ids),
        "traffic_light_frame_counts": dict(light_totals),
    }
    summary_path = Path(args.out_summary) if args.out_summary else None
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
