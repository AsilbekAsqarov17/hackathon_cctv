"""Sweep candidate stop-line positions and score each against the real queue.

A stop line has to sit where drivers actually stop, and this camera offers only
one signal-controlled approach, so the queue itself is the calibration target.
For each candidate position the script reports how far the vehicles holding the
queue are from the line, and selects the position that minimises that distance.

Only leads that stopped *upstream of the governed crossing* count. A vehicle
that stopped past the crossing was blocked by the junction, not by the signal,
so including it would drag the line into the middle of the intersection, which is
where a previous calibration attempt wrongly placed it.

Example::

    python scripts/fit_stop_line_sweep.py debug/perception/data_video2_tracks.jsonl \
        --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector, point_in_polygon  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from fit_stop_line import collect_leads  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--author-width", type=int, default=3840)
    parser.add_argument("--lanes", default="2,3")
    parser.add_argument("--setbacks", type=float, nargs="+",
                        default=[0, 40, 80, 120, 160, 200, 240, 300, 380, 460])
    parser.add_argument("--hold-sec", type=float, default=1.5)
    parser.add_argument("--episode-gap", type=float, default=8.0)
    args = parser.parse_args()

    lane_ids = {int(v) for v in args.lanes.split(",") if v.strip()}
    config = load_scene_config(args.scene, args.width, args.height)
    scene = SceneContext(config, args.width, args.height)
    k = args.author_width / args.width
    lanes = [l for l in config.lanes if l.lane_id in lane_ids]
    travel = normalized_vector(lanes[0].direction)
    assert travel is not None
    line_dir = (-travel[1], travel[0])
    line = config.stop_lines[0]

    # The crossing the governing signal faces.
    light_id = scene.signal_for_line(line)
    roi = next((l.roi for l in config.traffic_lights if l.light_id == light_id), None)
    if roi is None:
        raise SystemExit("no governing signal found")
    lx = (roi[0] + roi[2]) / 2.0
    ly = (roi[1] + roi[3]) / 2.0
    ranked = []
    for index, poly in enumerate(config.crossings):
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        ranked.append((math.hypot(cx - lx, cy - ly), index, cx, cy))
    ranked.sort()
    _, crossing_index, cx, cy = ranked[0]
    print(f"governing signal {light_id} at ({lx:.0f},{ly:.0f}); "
          f"crossing[{crossing_index}] centre ({cx:.0f},{cy:.0f}) is nearest")
    print(f"crossing centres: " + ", ".join(
        f"[{i}]=({px:.0f},{py:.0f})@{d:.0f}px" for d, i, px, py in ranked))

    leads = collect_leads(args.jsonl, lane_ids, travel, args.hold_sec, args.episode_gap)
    crossing_along = cx * travel[0] + cy * travel[1]
    holding = [a for a, _, _, _ in leads if a < crossing_along]
    print(f"\n{len(leads)} queue episodes; {len(holding)} held the queue upstream of "
          f"the crossing and are the calibration target")

    def governed(point) -> bool:
        return any(point_in_polygon(point, lane.polygon) for lane in lanes)

    def probe_for(cx_: float, cy_: float) -> tuple[float, float]:
        if governed((cx_, cy_)):
            return (cx_, cy_)
        for r in range(0, 400, 4):
            for angle in range(0, 360, 10):
                px = cx_ + r * math.cos(math.radians(angle))
                py = cy_ + r * math.sin(math.radians(angle))
                if governed((px, py)):
                    return (px, py)
        return (cx_, cy_)

    base = probe_for(cx, cy)
    print(f"\n{'setback':>8} {'median|off|':>12} {'mean|off|':>10} {'max|off|':>9}  score")
    print("-" * 56)
    best = None
    for setback in args.setbacks:
        anchor = (base[0] - travel[0] * setback, base[1] - travel[1] * setback)
        line_along = anchor[0] * travel[0] + anchor[1] * travel[1]
        offsets = [abs(a - line_along) for a in holding]
        if not offsets:
            continue
        median = statistics.median(offsets)
        mean = sum(offsets) / len(offsets)
        worst = max(offsets)
        print(f"{setback:8.0f} {median:12.0f} {mean:10.0f} {worst:9.0f}  {median:.1f}")
        if best is None or median < best[1]:
            best = (setback, median, anchor)

    if best is None:
        raise SystemExit("no usable setback")
    setback, _, anchor = best
    s_lo = s_hi = 0.0
    for _ in range(12):
        found_lo, found_hi = 0.0, 0.0
        for lane in lanes:
            xs = [p[0] for p in lane.polygon]
            ys = [p[1] for p in lane.polygon]
            for x in range(int(min(xs)) - 300, int(max(xs)) + 300, 8):
                for y in range(int(min(ys)) - 300, int(max(ys)) + 300, 8):
                    if not point_in_polygon((x, y), lane.polygon):
                        continue
                    dx, dy = x - anchor[0], y - anchor[1]
                    s = dx * line_dir[0] + dy * line_dir[1]
                    if s < 0:
                        found_lo = max(found_lo, -s)
                    else:
                        found_hi = max(found_hi, s)
        if abs(found_lo - s_lo) < 1e-6 and abs(found_hi - s_hi) < 1e-6:
            break
        s_lo, s_hi = found_lo, found_hi
    start = (anchor[0] - line_dir[0] * s_lo, anchor[1] - line_dir[1] * s_lo)
    end = (anchor[0] + line_dir[0] * s_hi, anchor[1] + line_dir[1] * s_hi)

    line_along = anchor[0] * travel[0] + anchor[1] * travel[1]
    print(f"\nbest setback {setback:.0f}px upstream of crossing[{crossing_index}]")
    print(f"start (video px) ({start[0]:.1f}, {start[1]:.1f})")
    print(f"end   (video px) ({end[0]:.1f}, {end[1]:.1f})")
    print(f"length {s_lo + s_hi:.0f} video px = {(s_lo + s_hi) * k:.0f} authoring px")
    print(f"\nqueue leaders relative to this line: " + ", ".join(
        f"{a - line_along:+.0f}" for a in sorted(holding)))
    print(f'\n"segment": [[{round(start[0] * k)}, {round(start[1] * k)}], '
          f'[{round(end[0] * k)}, {round(end[1] * k)}]],')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
