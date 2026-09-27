"""Summarise rule debug JSONL: group frames into per-track runs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    if args.label:
        records = [r for r in records if r["label"] == args.label]
    by_track: dict[tuple, list] = defaultdict(list)
    for record in records:
        by_track[(record["label"], record["track_id"])].append(record)

    print(f"{len(records)} frames, {len(by_track)} track-runs")
    for (label, tid), group in sorted(by_track.items()):
        group.sort(key=lambda r: r["frame"])
        ev = group[len(group) // 2]["evidence"]
        lanes = sorted({r["lane_id"] for r in group}, key=lambda v: (v is None, v))
        print(
            f"  {label:16} track#{tid:<5} lane={lanes} "
            f"frames {group[0]['frame']:5d}-{group[-1]['frame']:<5d} "
            f"t={group[0]['timestamp']:7.2f}-{group[-1]['timestamp']:7.2f}"
        )
        if label == "wrong_way":
            print(f"      dot={ev.get('dot')} expected={ev.get('expected')} observed={ev.get('observed')} travelled={ev.get('travelled_px')}")
        else:
            print(f"      stopped_since={ev.get('stopped_since')} held={ev.get('held_sec')} class={ev.get('class_name')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
