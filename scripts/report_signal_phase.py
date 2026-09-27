"""Report the calibrated signal phase timeline and check the queue against it.

Two things have to agree for the signal-based rules to work: the phase timeline
read from the calibrated ROI, and the queue that actually forms. This prints
both on one axis so a mismatch is obvious, e.g. a queue that forms while the
light is green means the ROI or the queue is being read wrongly.
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

from src.rules.motion import is_stationary  # noqa: E402
from src.scene.config import load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector  # noqa: E402

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "vehicle"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline", help="JSON/JSONL phase timeline from signal_timeline.py")
    parser.add_argument("--jsonl", default=None, help="track dump for the queue check")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--lanes", default="2,3")
    args = parser.parse_args()

    raw = Path(args.timeline).read_text(encoding="utf-8").strip()
    samples: list[tuple[float, str]] = []
    if raw.startswith("["):
        samples = [(float(t), str(c)) for t, c in json.loads(raw)]
    else:
        for line in raw.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append((float(record.get("timestamp", record.get("t", 0.0))),
                            str(record.get("color", record.get("state", "unknown")))))

    runs: list[tuple[float, float, str]] = []
    for timestamp, color in samples:
        if runs and runs[-1][2] == color:
            runs[-1] = (runs[-1][0], timestamp, color)
        else:
            runs.append((timestamp, timestamp, color))
    print(f"{args.timeline}: {len(samples)} samples, {len(runs)} phase runs")
    print(f"{'from':>8} {'to':>8} {'dur':>7}  colour")
    for start, end, color in runs:
        print(f"{start:8.2f} {end:8.2f} {end - start:7.2f}  {color}")

    if not args.jsonl:
        return 0

    config = load_scene_config(args.scene, args.width, args.height)
    lane_ids = {int(v) for v in args.lanes.split(",") if v.strip()}
    (ax, ay), _ = config.stop_lines[0].segment
    travel = normalized_vector(config.stop_lines[0].direction) or (1.0, 0.0)

    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            tid = track["track_id"]
            series[tid].append((record["timestamp"], track["bottom_center"], track["lane_id"]))
            klass[tid] = track["class_name"]

    # Vehicles stationary in the governed lanes, per second.
    occupancy: dict[int, int] = defaultdict(int)
    for tid, points in series.items():
        if klass.get(tid) not in VEHICLES:
            continue
        history: list[tuple[float, float, float]] = []
        for timestamp, bottom, lane_id in points:
            history.append((timestamp, bottom[0], bottom[1]))
            if lane_id in lane_ids and is_stationary(history, 2.5, 25.0, 250.0, timestamp):
                d = (bottom[0] - ax) * travel[0] + (bottom[1] - ay) * travel[1]
                if d < 260.0:  # upstream of the line: queueing
                    occupancy[int(timestamp)] += 1

    def color_at(t: float) -> str:
        best = "unknown"
        for timestamp, color in samples:
            if timestamp <= t:
                best = color
            else:
                break
        return best

    print(f"\nstationary vehicles upstream of the line, per phase run:")
    print(f"{'from':>8} {'to':>8} {'dur':>7}  {'colour':<8} {'mean stopped':>13} {'peak':>5}")
    for start, end, color in runs:
        values = [occupancy[s] for s in range(int(start), max(int(start) + 1, int(end)))]
        if not values:
            continue
        print(f"{start:8.2f} {end:8.2f} {end - start:7.2f}  {color:<8} "
              f"{sum(values) / len(values):13.1f} {max(values):5d}")
    print(f"\ncolours seen: {sorted({c for _, c in samples})}")
    print(f"color_at(5s)={color_at(5.0)}  color_at(70s)={color_at(70.0)}  color_at(150s)={color_at(150.0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
