"""Print replayed events and check them against the known signal-queue windows."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("events")
    parser.add_argument("--window", action="append", default=[],
                        help="label:start:end of a window that must stay clean")
    parser.add_argument("--x-range", default="",
                        help="only consider events whose bottom_center x is in LO,HI")
    args = parser.parse_args()
    events = json.load(open(args.events, encoding="utf-8"))
    if args.x_range:
        lo, hi = (float(v) for v in args.x_range.split(","))
        kept = [e for e in events
                if "bottom_center" in e and lo <= e["bottom_center"][0] <= hi]
        print(f"x filter [{lo},{hi}]: {len(kept)}/{len(events)} events retained")
        events = kept
    print(f"{len(events)} events")
    for event in events:
        print(f"  {event['label']:16} track#{event['track_id']:<5} lane={event['lane_id']} "
              f"light={event.get('light_at_start')} "
              f"{event['start_sec']:7.2f} -> {event['end_sec']:7.2f} ({event['duration_sec']:6.2f}s)")
    print()
    for spec in args.window:
        name, _, span = spec.partition(":")
        start, end = (float(v) for v in span.split(","))
        hits = [e for e in events if e["start_sec"] < end and e["end_sec"] > start]
        status = "CLEAN" if not hits else f"COLLISION x{len(hits)}"
        print(f"  window {name:28} [{start:6.1f},{end:6.1f}] -> {status}")
        for hit in hits:
            print(f"      {hit['label']} track#{hit['track_id']} {hit['start_sec']}->{hit['end_sec']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
