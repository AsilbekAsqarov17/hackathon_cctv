"""Validate the development event-label JSON shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OFFICIAL = {
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    args = parser.parse_args()
    data = json.loads(Path(args.labels).read_text(encoding="utf-8-sig"))
    errors = []
    for video, item in data.items():
        duration = float(item.get("duration", 0.0))
        for event in item.get("events", []):
            if not isinstance(event, list) or len(event) != 3:
                errors.append(f"{video}: event must be [start,end,label]")
                continue
            start, end, label = event
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= start < end:
                errors.append(f"{video}: invalid interval {event}")
            if label not in OFFICIAL:
                errors.append(f"{video}: unknown label {label}")
            if duration and end > duration + 0.5:
                errors.append(f"{video}: event ends after video ({event})")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"valid labels for {len(data)} video(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
