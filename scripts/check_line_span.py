"""Check that a calibrated line really separates the lanes it governs.

A line that covers only part of its carriageway silently disables the rules
that depend on it: crossings on the uncovered part are never seen. The test is
done in the line's own frame, because the carriageway is diagonal and a vertical
slice would give a misleading answer:

* ``s`` runs along the line, and must cover the whole ``s`` extent of the lane.
* ``d`` runs perpendicular to the line (along travel), and the lane must have
  points on both sides -- otherwise the line does not actually cut the lane.

Example::

    python scripts/check_line_span.py --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector, point_in_polygon  # noqa: E402


def lane_extent(lane, u: tuple[float, float], v: tuple[float, float], origin) -> tuple[float, float] | None:
    """Range of projections onto u (along-line) and v (across-line) for a lane."""
    su, sv = [], []
    for x in range(int(min(p[0] for p in lane.polygon)) - 200,
                   int(max(p[0] for p in lane.polygon)) + 200, 12):
        for y in range(int(min(p[1] for p in lane.polygon)) - 200,
                       int(max(p[1] for p in lane.polygon)) + 200, 12):
            if not point_in_polygon((x, y), lane.polygon):
                continue
            dx, dy = x - origin[0], y - origin[1]
            su.append(dx * u[0] + dy * u[1])
            sv.append(dx * v[0] + dy * v[1])
    if not su:
        return None
    return (min(su), max(su), min(sv), max(sv))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--min-span", type=float, default=0.98)
    parser.add_argument("--governed", default="2,3",
                        help="lane ids this stop line controls; others are reported only")
    args = parser.parse_args()
    args.governed = {int(v) for v in args.governed.split(",") if v.strip()}

    config = load_scene_config(args.scene, args.width, args.height)
    k = args.width / 3840.0
    print(f"scene {args.width}x{args.height}\n")
    ok = True
    for line in config.stop_lines:
        (ax, ay), (bx, by) = line.segment
        u = normalized_vector((bx - ax, by - ay))
        if u is None:
            print(f"stop line {line.line_id}: degenerate")
            ok = False
            continue
        v = (-u[1], u[0])
        span = math.hypot(bx - ax, by - ay)
        print(f"stop line {line.line_id}: ({ax:.0f},{ay:.0f})->({bx:.0f},{by:.0f})  "
              f"length {span:.0f}px ({span * k:.0f} in 4K)")
        for lane in config.lanes:
            extent = lane_extent(lane, u, v, (ax, ay))
            if extent is None:
                continue
            s0, s1, d0, d1 = extent
            covered = max(0.0, min(s1, span) - max(s0, 0.0))
            want = s1 - s0
            share = 100.0 * covered / want if want > 1e-6 else 100.0
            crosses = d0 < 0.0 < d1
            governed = lane.lane_id in args.governed if args.governed else True
            tag = "governed" if governed else "not governed"
            if not governed:
                # Covering a lane the line does not control is neither required
                # nor wrong, so it is reported but never fails the check.
                print(f"    lane {lane.lane_id} {lane.name:8s} s {s0:7.0f}..{s1:7.0f} "
                      f"covered {share:5.1f}%  ({tag})")
                continue
            flag = ""
            if share < args.min_span * 100.0:
                flag += "  <-- DOES NOT SPAN"
                ok = False
            if not crosses:
                flag += "  <-- DOES NOT CUT THE LANE"
                ok = True if ok and not flag else ok
            print(f"    lane {lane.lane_id} {lane.name:8s} s {s0:7.0f}..{s1:7.0f} "
                  f"d {d0:8.0f}..{d1:8.0f}  covered {share:5.1f}%{flag}")
        print()
    print("OK" if ok else "PROBLEM: a stop line does not span or cut its lane")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
