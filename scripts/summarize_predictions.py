"""Summarise a predictions file: per-class counts, risk curve, alarms.

A quick sanity read of what a run actually produced, without having to open the
JSON. Also reports the alarm statistics the official Part B metric depends on,
so a saturated risk curve is visible immediately.

Example::

    python scripts/summarize_predictions.py predictions_samples.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# The official alarm threshold and merge distance (evaluate.py).
ALARM = 0.5
MERGE_GAP = 2.0


def alarm_runs(curve: list[list[float]]) -> list[tuple[float, float]]:
    """Maximal runs of score >= ALARM, with runs closer than MERGE_GAP merged."""
    runs: list[tuple[float, float]] = []
    for timestamp, score in curve:
        if score < ALARM:
            continue
        if runs and timestamp - runs[-1][1] <= MERGE_GAP:
            runs[-1] = (runs[-1][0], timestamp)
        else:
            runs.append((timestamp, timestamp))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred", nargs="?", default="predictions_samples.json")
    args = parser.parse_args()

    data = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    print(f"{args.pred}")
    print(f"team: {data.get('team', '(not set)')}")

    overall: collections.Counter = collections.Counter()
    for name in sorted(data.get("videos", {})):
        record = data["videos"][name]
        events = record.get("events", [])
        risk = record.get("risk", [])
        counts = collections.Counter(str(e[2]) for e in events)
        overall.update(counts)
        print(f"\n{name}")
        if record.get("duration"):
            print(f"  duration {record['duration']}s @ {record['fps']} fps")
        print(f"  {len(events)} events, {len(risk)} risk samples")
        for label in sorted(counts):
            print(f"    {label:<22} {counts[label]}")
        if risk:
            scores = [float(s) for _, s in risk]
            runs = alarm_runs(risk)
            span = float(risk[-1][0]) - float(risk[0][0])
            print(f"  risk: min {min(scores):.3f}  max {max(scores):.3f}  "
                  f"mean {sum(scores) / len(scores):.3f}")
            print(f"  frames >= {ALARM}: {sum(1 for s in scores if s >= ALARM)}/{len(scores)} "
                  f"({100.0 * sum(1 for s in scores if s >= ALARM) / len(scores):.1f}%)")
            print(f"  alarm runs: {len(runs)}"
                  + (f"  (longest {max(b - a for a, b in runs):.1f}s)" if runs else ""))
            print(f"  curve spans {span:.1f}s")
        # Overlap check: the harness rejects same-class overlaps, so a file that
        # reached disk should have none.
        by_label: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        for start, end, label in events:
            by_label[str(label)].append((float(start), float(end)))
        clashes = 0
        for label, spans in by_label.items():
            spans.sort()
            for i in range(1, len(spans)):
                if spans[i][0] < spans[i - 1][1]:
                    clashes += 1
                    print(f"  *** overlapping {label} segments: "
                          f"{spans[i - 1]} and {spans[i]}")
        if not clashes:
            print("  same-class overlaps: none")

    print(f"\ntotal {sum(overall.values())} events")
    for label in sorted(overall):
        print(f"  {label:<22} {overall[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
