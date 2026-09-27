"""Calibrate motion thresholds from an existing perception JSONL.

Reports, over trailing windows, the bottom-centre net displacement and path
length for stationary (queued) versus moving tracks, so rule thresholds are set
from measurements instead of guesses.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import trailing_window  # noqa: E402
def window_stats(points: list[tuple[float, float, float]], window: float) -> tuple[float, float]:
    """Return (net displacement px, path length px) over a trailing window."""
    if len(points) < 2:
        return 0.0, 0.0
    t_end = points[-1][0]
    recent = trailing_window(points, window, t_end)
    if len(recent) < 2:
        return 0.0, 0.0
    net = math.hypot(recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2])
    path = 0.0
    for i in range(1, len(recent)):
        path += math.hypot(recent[i][1] - recent[i - 1][1], recent[i][2] - recent[i - 1][2])
    return net, path



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import trailing_window  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--focus", type=int, default=0, help="track id to profile")
    args = parser.parse_args()

    series: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    klass: dict[int, str] = {}
    for line in open(args.jsonl, encoding="utf-8"):
        record = json.loads(line)
        for track in record["tracks"]:
            tid = track["track_id"]
            klass[tid] = track["class_name"]
            series[tid].append((record["timestamp"], track["bottom_center"][0], track["bottom_center"][1]))

    nets, paths = [], []
    still_samples, moving_samples = [], []
    for tid, points in series.items():
        if len(points) < 10:
            continue
        for i in range(0, len(points), 5):
            net, path = window_stats(points[: i + 1], args.window)
            if net == 0.0 and path == 0.0:
                continue
            nets.append(net)
            paths.append(path)
            (still_samples if net < 20 else moving_samples).append((net, path))

    def pct(values, q):
        values = sorted(values)
        return values[min(len(values) - 1, int(len(values) * q))]

    print(f"window={args.window}s  windows sampled={len(nets)}")
    print(f"net  displacement px: p50={pct(nets,0.5):.1f} p90={pct(nets,0.9):.1f} p99={pct(nets,0.99):.1f}")
    print(f"path length     px: p50={pct(paths,0.5):.1f} p90={pct(paths,0.9):.1f} p99={pct(paths,0.99):.1f}")
    if still_samples:
        sn = [s[0] for s in still_samples]
        sp = [s[1] for s in still_samples]
        print(f"\nwindows with net<20px (n={len(still_samples)}):")
        print(f"  net  p50={pct(sn,0.5):.1f} p95={pct(sn,0.95):.1f} max={max(sn):.1f}")
        print(f"  path p50={pct(sp,0.5):.1f} p95={pct(sp,0.95):.1f} max={max(sp):.1f}")
    if moving_samples:
        mn = [s[0] for s in moving_samples]
        mp = [s[1] for s in moving_samples]
        print(f"\nwindows with net>=20px (n={len(moving_samples)}):")
        print(f"  net  p5={pct(mn,0.05):.1f} p50={pct(mn,0.5):.1f}")
        print(f"  path p5={pct(mp,0.05):.1f} p50={pct(mp,0.5):.1f}")

    if args.focus:
        points = series.get(args.focus, [])
        print(f"\ntrack #{args.focus} ({klass.get(args.focus)}) frames={len(points)}")
        print("   t      x        y      net   path")
        for i in range(0, len(points), max(1, len(points) // 45)):
            net, path = window_stats(points[: i + 1], args.window)
            t, x, y = points[i]
            print(f"{t:7.1f} {x:7.0f} {y:7.0f} {net:7.1f} {path:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
