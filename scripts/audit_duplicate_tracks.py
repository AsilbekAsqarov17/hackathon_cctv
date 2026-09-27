"""Find pairs of tracks that are persistently the same object.

ByteTrack can hold two ids for one vehicle for a while, and a distant view makes
two adjacent vehicles' boxes overlap heavily. Either way, a "collision" between
a track and its own duplicate is a pure artefact, and it is the single largest
source of accident false positives on this camera.

A duplicate is identified by tracks that stay within a small distance of each
other for a long time while keeping a near-constant offset: real vehicles
separate, overtake or turn, but two ids on one object never do.

Example::

    python scripts/audit_duplicate_tracks.py debug/perception/data_video2_tracks.jsonl
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

from src.config import load_config  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402
from src.contracts import TrackObservation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-distance", type=float, default=60.0)
    parser.add_argument("--min-overlap", type=float, default=0.35)
    parser.add_argument("--min-frames", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    scene = SceneContext(load_scene_config(config["scene"]["path"], args.width, args.height),
                         args.width, args.height)
    manager = TrackManager(max_age_seconds=2.0, history_seconds=8.0)

    # co[tid] -> {other_tid: (frames_together, offset_consistency)}
    together: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    def iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ba = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ba - inter)

    index = 0
    for raw in open(args.jsonl, encoding="utf-8"):
        record = json.loads(raw)
        index += 1
        obs = [
            TrackObservation(track_id=t["track_id"], bbox=tuple(t["bbox"]),
                             score=t.get("confidence", 0.9), class_id=t["class_id"],
                             class_name=t["class_name"])
            for t in record["tracks"]
        ]
        tracks = manager.update(obs, record["timestamp"], record["frame"], scene)
        if index % 3:
            continue
        for i, first in enumerate(tracks):
            for second in tracks[i + 1:]:
                if first.class_name != second.class_name:
                    continue
                overlap = iou(first.bbox, second.bbox)
                if overlap < args.min_overlap:
                    continue
                distance = math.hypot(first.center[0] - second.center[0],
                                      first.center[1] - second.center[1])
                if distance > args.max_distance:
                    continue
                together[first.track_id][second.track_id].append(
                    (record["timestamp"], first.center[0] - second.center[0],
                     first.center[1] - second.center[1])
                )
                together[second.track_id][first.track_id].append(
                    (record["timestamp"], -first.center[0] + second.center[0],
                     -first.center[1] + second.center[1])
                )

    rows = []
    for tid, partners in together.items():
        for other, samples in partners.items():
            if tid > other or len(samples) < args.min_frames:
                continue
            dxs = [s[1] for s in samples]
            dys = [s[2] for s in samples]
            spread = math.hypot(max(dxs) - min(dxs), max(dys) - min(dys))
            duration = samples[-1][0] - samples[0][0]
            rows.append((len(samples), duration, spread, tid, other,
                         (sum(dxs) / len(dxs), sum(dys) / len(dys))))
    rows.sort(reverse=True)
    print(f"pairs co-existing with IoU>={args.min_overlap} and centre distance "
          f"<={args.max_distance}px, for >= {args.min_frames} samples:")
    print(f"{'frames':>7} {'dur':>7} {'offset_spread':>14} {'a':>6} {'b':>6} {'mean_offset':>18}")
    print("-" * 70)
    for count, duration, spread, a, b, offset in rows[:30]:
        print(f"{count:7d} {duration:7.1f} {spread:14.1f} {a:6d} {b:6d} "
              f"({offset[0]:7.1f},{offset[1]:7.1f})")
    print(f"\n{len(rows)} candidate duplicate pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
