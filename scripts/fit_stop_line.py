"""Place a stop line immediately upstream of the crossing it governs.

``red_light`` fires when a vehicle's *front* crosses the stop line, so the line
has to be where drivers actually stop, and it has to span the whole carriageway.
Fitting a position to noisy queue statistics is unstable — different signal
cycles produce leads more than a thousand pixels apart, because a queue that
has not yet reached the line and one that has overshot it are both "stops".

So the position comes from the road layout instead, which is stable and is how
stop lines are placed in practice: immediately upstream of the crossing the
queue forms at, perpendicular to travel. The queue measurement is then used as
an independent *check* — it reports where lead vehicles came to rest relative to
the chosen line, and a correct line puts them just upstream of it.

The segment is extended perpendicular to travel until it leaves the governed
carriageway on both sides, bridging the gap between adjacent lanes.

Example::

    python scripts/fit_stop_line.py debug/perception/data_video2_tracks.jsonl --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rules.motion import is_stationary  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector, point_in_polygon  # noqa: E402

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "vehicle"}


def collect_leads(path: str, lane_ids: set[int], travel, hold_sec: float,
                  gap: float) -> list[tuple[float, float, float, int]]:
    """Lead vehicle of each queue episode, in absolute projections.

    Returns (along_travel, across_line, timestamp, queue_size). Absolute
    projections do not depend on any candidate line, so the measurement is
    stable while the line is being chosen.
    """
    line_dir = (-travel[1], travel[0])
    series: dict[int, list] = defaultdict(list)
    klass: dict[int, str] = {}
    for raw in open(path, encoding="utf-8"):
        record = json.loads(raw)
        for track in record["tracks"]:
            tid = track["track_id"]
            series[tid].append((record["timestamp"], track["bbox"], track["lane_id"]))
            klass[tid] = track["class_name"]

    stops: list[tuple[float, float, float]] = []
    for tid, points in series.items():
        if klass.get(tid) not in VEHICLES:
            continue
        history: list[tuple[float, float, float]] = []
        stop_start: float | None = None
        best: tuple[float, float] | None = None
        for timestamp, bbox, lane_id in points:
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            reach = (abs(travel[0]) * (bbox[2] - bbox[0]) + abs(travel[1]) * (bbox[3] - bbox[1])) / 2.0
            front = (cx + travel[0] * reach, cy + travel[1] * reach)
            history.append((timestamp, front[0], front[1]))
            if lane_id not in lane_ids:
                stop_start, best = None, None
                continue
            if is_stationary(history, 2.5, 25.0, 250.0, timestamp):
                if stop_start is None:
                    stop_start = timestamp
                along = front[0] * travel[0] + front[1] * travel[1]
                across = front[0] * line_dir[0] + front[1] * line_dir[1]
                if best is None or along > best[0]:
                    best = (along, across)
            else:
                if stop_start is not None and timestamp - stop_start >= hold_sec and best is not None:
                    stops.append((best[0], best[1], timestamp))
                stop_start, best = None, None

    ordered = sorted(stops, key=lambda s: s[2])
    episodes: list[list[tuple[float, float, float]]] = []
    for stop in ordered:
        if episodes and stop[2] - episodes[-1][-1][2] <= gap:
            episodes[-1].append(stop)
        else:
            episodes.append([stop])
    return [(max(ep, key=lambda s: s[0])[0], max(ep, key=lambda s: s[0])[1],
             max(ep, key=lambda s: s[0])[2], len(ep)) for ep in episodes if ep]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="?", default=None,
                        help="optional track dump, used to check the chosen line")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--author-width", type=int, default=3840)
    parser.add_argument("--lanes", default="2,3")
    parser.add_argument("--crossing", type=int, default=0)
    parser.add_argument("--setback", type=float, default=26.0,
                        help="px upstream of the crossing to place the line")
    parser.add_argument("--max-offset", type=float, default=200.0,
                        help="warn if lead vehicles queue further than this upstream")
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
    line = config.stop_lines[0] if config.stop_lines else None

    # The crossing the line serves is the one the GOVERNING SIGNAL faces, not
    # simply the first crossing in the file. A signal head is mounted at the near
    # side of the crossing it controls, so the nearest crossing to the head is
    # the one traffic queues for. Anchoring to the wrong crossing places the line
    # inside the junction, where it can never be crossed legitimately.
    light = scene.signal_for_line(line) if line is not None else None
    crossing_index = args.crossing
    if light is not None:
        roi = next((l.roi for l in config.traffic_lights if l.light_id == light), None)
        if roi is not None:
            lx = (roi[0] + roi[2]) / 2.0
            ly = (roi[1] + roi[3]) / 2.0
            distances = []
            for index, poly in enumerate(config.crossings):
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                distances.append((math.hypot(cx - lx, cy - ly), index, cx, cy))
            distances.sort()
            best_distance, crossing_index, cx, cy = distances[0]
            print(f"governing signal '{light}' at ({lx:.0f},{ly:.0f})")
            print(f"  nearest crossing is [{crossing_index}] at ({cx:.0f},{cy:.0f}), "
                  f"{best_distance:.0f}px away")
            for distance, index, px, py in distances[1:]:
                print(f"    (crossing[{index}] at ({px:.0f},{py:.0f}) is {distance:.0f}px away)")
    else:
        crossing = config.crossings[crossing_index]
        cx = sum(p[0] for p in crossing) / len(crossing)
        cy = sum(p[1] for p in crossing) / len(crossing)
    crossing = config.crossings[crossing_index]
    # Walk upstream from the crossing centre until leaving the carriageway, then
    # step back onto the carriageway: that is the line's position.
    def governed(point) -> bool:
        return any(point_in_polygon(point, lane.polygon) for lane in lanes)

    probe = (cx, cy)
    if not governed(probe):
        # The crossing centre can sit just off the lane; search nearby.
        found = None
        for r in range(0, 400, 4):
            for angle in range(0, 360, 10):
                px = cx + r * math.cos(math.radians(angle))
                py = cy + r * math.sin(math.radians(angle))
                if governed((px, py)):
                    found = (px, py)
                    break
            if found:
                break
        if found is None:
            raise SystemExit("crossing does not touch the governed lanes")
        probe = found
    setback = args.setback
    anchor = (probe[0] - travel[0] * setback, probe[1] - travel[1] * setback)
    # The line must sit where the queue's LEAD vehicle parks, which is the real
    # test of the placement. If the lead vehicles are far upstream of the
    # candidate line, the line is inside the junction and is wrong.

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

    print(f"video {args.width}x{args.height}  lanes {sorted(lane_ids)}  "
          f"travel ({travel[0]:.3f},{travel[1]:.3f})")
    print(f"crossing[{crossing_index}] centre ({cx:.0f},{cy:.0f}), "
          f"line {setback:.0f}px upstream")
    print(f"start (video px) ({start[0]:.1f}, {start[1]:.1f})")
    print(f"end   (video px) ({end[0]:.1f}, {end[1]:.1f})")
    print(f"length {s_lo + s_hi:.0f} video px = {(s_lo + s_hi) * k:.0f} authoring px\n")
    print(f'"segment": [[{round(start[0] * k)}, {round(start[1] * k)}], '
          f'[{round(end[0] * k)}, {round(end[1] * k)}]],')

    if args.jsonl:
        # The decisive check: where do queue lead vehicles actually park, and are
        # they upstream of this line? A lead that stopped far past the line is
        # not a queue leader for it; the vehicle that holds the queue is the one
        # stopped furthest forward while still upstream of the crossing.
        leads = collect_leads(args.jsonl, lane_ids, travel, args.hold_sec, args.episode_gap)
        line_along = anchor[0] * travel[0] + anchor[1] * travel[1]
        crossing_along = cx * travel[0] + cy * travel[1]
        holding = sorted(o for o in (along for along, _, _, _ in leads) if o < crossing_along)
        print(f"\ncheck: {len(leads)} queue episodes; lead-vehicle nose offset from "
              f"this line (negative = still upstream, i.e. correctly queued)")
        for along, _across, t, size in sorted(leads, key=lambda item: item[2]):
            offset = along - line_along
            where = "upstream of line" if offset < 0 else "past the line"
            if offset < 0 and offset < -args.max_offset:
                where += " (*** more than %.0fpx back: line is too far downstream)" % args.max_offset
            print(f"  t={t:7.2f} size={size:3d} offset={offset:8.0f}  {where}")
        if holding:
            print(f"\n  {len(holding)}/{len(leads)} leads stopped upstream of the crossing; "
                  f"they hold the queue at offsets "
                  f"{min(a - line_along for a in holding):.0f}.."
                  f"{max(a - line_along for a in holding):.0f}px from this line")
        else:
            print("\n  *** WARNING: no lead vehicle ever queued upstream of the crossing, "
                  "so this line does not correspond to any real queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
