"""Compare two event files field by field (used for before/after regression)."""
from __future__ import annotations

import json
import sys


def load(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("events", "segments", "items"):
            if key in data:
                return data[key]
        return [data]
    return data


def key(event: dict) -> tuple:
    return (
        event.get("label"),
        event.get("track_id"),
        round(float(event.get("start_sec", -1)), 2),
        round(float(event.get("end_sec", -1)), 2),
    )


def main() -> int:
    before = load(sys.argv[1])
    after = load(sys.argv[2])
    print(f"before: {len(before)} events   after: {len(after)} events\n")
    bkeys = [key(e) for e in before]
    akeys = [key(e) for e in after]
    print(f"{'label':<16}{'trk':>6}{'start':>10}{'end':>10}   status")
    print("-" * 62)
    for k in bkeys:
        status = "same" if k in akeys else ("MOVED" if any(x[:3] == k[:3] for x in akeys) else "REMOVED")
        print(f"{str(k[0]):<16}{k[1]:>6}{k[2]:>10.2f}{k[3]:>10.2f}   {status}")
    for k in akeys:
        if k not in bkeys:
            status = "MOVED" if any(x[:3] == k[:3] for x in bkeys) else "ADDED"
            print(f"{str(k[0]):<16}{k[1]:>6}{k[2]:>10.2f}{k[3]:>10.2f}   {status}")
    added = [k for k in akeys if k not in bkeys and not any(x[:3] == k[:3] for x in bkeys)]
    removed = [k for k in bkeys if k not in akeys and not any(x[:3] == k[:3] for x in akeys)]
    print(f"\nidentical={bkeys == akeys}  added={len(added)}  removed={len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
