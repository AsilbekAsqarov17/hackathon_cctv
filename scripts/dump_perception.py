"""Cache detector+tracker output for a video so rule work needs no GPU.

Runs the same perception stack as ``PartAPipeline`` (detector, ByteTrack,
TrackManager, scene context) and writes one JSONL record per sampled frame:

    {"frame", "timestamp", "tracks": [{track_id, class_name, class_id, bbox,
     bottom_center, center, velocity, speed, lane_id}]}

Rules are then developed and re-validated against this file with
``scripts/replay_rules.py``, which makes a full rule iteration take seconds
instead of a quarter of an hour.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import apply_environment_overrides, load_config  # noqa: E402
from src.perception.detector import build_detector  # noqa: E402
from src.perception.tracker import build_tracker  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402
from src.video import VideoReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=0,
                        help="override detector stride (0 = use the config value)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    config = apply_environment_overrides(load_config(args.config))
    video = Path(args.video)
    reader = VideoReader(str(video))
    info = reader.info
    scene_path = ROOT / config.get("scene", {}).get("path", "configs/scenes/data_video1.json")
    scene = SceneContext(load_scene_config(scene_path, info.width, info.height), info.width, info.height)
    detector = build_detector(config.get("detector", {}))
    tracker_config = dict(config.get("tracker", {}))
    tracker_config.setdefault("frame_rate", info.fps)
    tracker = build_tracker(tracker_config)
    manager = TrackManager(
        max_age_seconds=float(config.get("track_manager", {}).get("max_age_seconds", 2.0)),
        history_seconds=float(config.get("track_manager", {}).get("history_seconds", 8.0)),
    )
    stride = args.stride or max(1, int(config.get("detector", {}).get("stride", 3)))
    if args.max_frames:
        stride = max(stride, -(-info.n_frames // args.max_frames))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for frame_id, timestamp, frame in reader.frames(stride=stride):
            detections = detector.predict(frame)
            observations = tracker.update(detections, frame_id, timestamp, frame)
            tracks = manager.update(observations, timestamp, frame_id, scene)
            handle.write(json.dumps({
                "frame": frame_id,
                "timestamp": round(timestamp, 4),
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "class_id": t.class_id,
                        "class_name": t.class_name,
                        "bbox": [round(v, 2) for v in t.bbox],
                        "center": [round(v, 2) for v in t.center],
                        "bottom_center": [round(v, 2) for v in t.bottom_center],
                        "velocity": [round(v, 3) for v in t.velocity],
                        "acceleration": [round(v, 3) for v in t.acceleration],
                        "speed": round(t.speed, 3),
                        "confidence": round(t.score, 4),
                        "lane_id": t.lane_id,
                    }
                    for t in tracks
                ],
            }) + "\n")
            written += 1
    reader.close()
    elapsed = time.perf_counter() - started
    print(
        f"{video.name}: {info.width}x{info.height} {written} frames (stride {stride}) "
        f"in {elapsed:.1f}s -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
