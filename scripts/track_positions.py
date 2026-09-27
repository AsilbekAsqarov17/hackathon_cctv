"""Positions of specific tracks in a perception or rules JSONL at given times."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--at", default="", help="comma separated timestamps")
    args = parser.parse_args()
    wanted = {int(v) for v in args.ids.split(",") if v.strip()}
    targets = [float(v) for v in args.at.split(",") if v.strip()]

    best: dict[int, tuple] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        ts = record["timestamp"]
        for track in record["tracks"]:
            tid = track["track_id"]
            if tid not in wanted:
                continue
            if tid in best and targets:
                continue
            best[tid] = (ts, track["bottom_center"], track.get("lane_id"), track["class_name"])
    print(f"{'track':>6} {'t':>8} {'bottom_center':>18} {'lane':>6}  class")
    for tid, (ts, bc, lane, name) in sorted(best.items()):
        print(f"{tid:6d} {ts:8.2f} {str(bc):>18} {str(lane):>6}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
