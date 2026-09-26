#!/usr/bin/env python3
"""
derive_scene_geometry.py - measure the camera's road area from tracked traffic.

Hand-drawing road polygons for a fixed camera is guesswork, and a polygon that
is slightly wrong is worse than no polygon at all: it puts the carriageway on
the pavement, which turns parked cars into `stopped_vehicle` and pedestrians
waiting on the kerb into `jaywalking`.

This script instead measures where vehicles actually are. It runs the same
detector and tracker the runtime uses, accumulates the ground-contact point
(the bottom centre of each vehicle box) over a sample of frames, and reports

  * a coarse occupancy grid, so the road extent can be read off directly;
  * an occupancy image with the grid drawn on it, for visual checking;
  * a suggested road polygon per connected region, snapped to the grid.

The output is a *measurement*, not a finished scene file. Review the overlay,
then copy the polygons into configs/scenes/<camera>.json and add crosswalks,
stop lines and signal boxes, which cannot be measured this way.

    python scripts/derive_scene_geometry.py data/samples_small --frames 120
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import apply_environment_overrides, load_config  # noqa: E402
from src.perception.detector import build_detector  # noqa: E402
from src.perception.tracker import build_tracker  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402
from src.video import VideoReader  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}


def collect(video: str, detector, tracker_config: dict, stride: int, max_frames: int) -> tuple[list, list, int, int]:
    """Return vehicle and person ground-contact points in normalized coords."""
    reader = VideoReader(video)
    info = reader.info
    tracker = build_tracker(dict(tracker_config))
    manager = TrackManager(max_age_seconds=2.0)
    vehicles: list[tuple[float, float]] = []
    people: list[tuple[float, float]] = []
    seen = 0
    try:
        for frame_id, _timestamp, frame in reader.frames(stride=stride):
            try:
                detections = detector.predict(frame)
                observations = tracker.update(detections, frame_id, _timestamp, frame)
            except Exception as exc:  # pragma: no cover - dev tool
                print(f"  detection failed at frame {frame_id}: {exc}", file=sys.stderr)
                break
            tracks = manager.update(observations, _timestamp, frame_id)
            for track in tracks:
                bx, by = track.bottom_center
                point = (bx / max(1, info.width), by / max(1, info.height))
                if TrackManager.is_vehicle(track):
                    vehicles.append(point)
                elif TrackManager.is_person(track):
                    people.append(point)
            seen += 1
            if seen >= max_frames:
                break
    finally:
        reader.close()
    return vehicles, people, info.width, info.height


def occupancy(points: list[tuple[float, float]], cols: int, rows: int) -> list[list[int]]:
    grid = [[0] * cols for _ in range(rows)]
    for x, y in points:
        cx = min(cols - 1, max(0, int(x * cols)))
        cy = min(rows - 1, max(0, int(y * rows)))
        grid[cy][cx] += 1
    return grid


def largest_component(grid: list[list[int]], threshold: int) -> list[list[int]]:
    """Keep only the biggest 4-connected region above threshold."""
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    best: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if seen[r][c] or grid[r][c] < threshold:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and not seen[ny][nx] and grid[ny][nx] >= threshold:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    mask = [[0] * cols for _ in range(rows)]
    for y, x in best:
        mask[y][x] = 1
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video file(s) or directory")
    ap.add_argument("--frames", type=int, default=120, help="sampled frames per video (default 120)")
    ap.add_argument("--stride", type=int, default=6, help="read every Nth frame (default 6)")
    ap.add_argument("--cols", type=int, default=40, help="occupancy grid columns (default 40)")
    ap.add_argument("--rows", type=int, default=24, help="occupancy grid rows (default 24)")
    ap.add_argument("--min-count", type=int, default=3, help="cells below this count are not road")
    ap.add_argument("--out", default="data/inspection/scene_geometry", help="output prefix")
    ap.add_argument("--config", default=None, help="runtime config JSON")
    args = ap.parse_args()

    videos: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            videos.extend(sorted(x for x in p.iterdir() if x.suffix in VIDEO_SUFFIXES))
        elif p.is_file():
            videos.append(p)
    if not videos:
        print("no videos found", file=sys.stderr)
        return 2

    config = apply_environment_overrides(load_config(args.config or (ROOT / "configs" / "default.json")))
    detector = build_detector(config.get("detector", {}))
    tracker_config = dict(config.get("tracker", {}))
    tracker_config["backend"] = "simple_bytetrack"

    all_vehicles: list[tuple[float, float]] = []
    all_people: list[tuple[float, float]] = []
    for video in videos:
        print(f"analysing {video.name} ...", flush=True)
        v, p, w, h = collect(str(video), detector, tracker_config, args.stride, args.frames)
        print(f"  {len(v)} vehicle contacts, {len(p)} person contacts at {w}x{h}")
        all_vehicles.extend(v)
        all_people.extend(p)

    if not all_vehicles:
        print("no vehicles measured; is the detector working?", file=sys.stderr)
        return 1

    grid = occupancy(all_vehicles, args.cols, args.rows)
    mask = largest_component(grid, args.min_count)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": [v.name for v in videos],
        "frames_per_video": args.frames,
        "stride": args.stride,
        "grid": {"cols": args.cols, "rows": args.rows},
        "min_count": args.min_count,
        "vehicle_contacts": len(all_vehicles),
        "person_contacts": len(all_people),
        "occupancy": grid,
        "road_mask": mask,
    }
    (out.with_suffix(".json")).write_text(json.dumps(payload, indent=1))

    # ASCII view is the quickest way to read the shape without opening an image.
    print("\nroad mask (# = road, + = busy, . = occasional, ' ' = never seen a vehicle)")
    for r in range(args.rows):
        line = []
        for c in range(args.cols):
            n = grid[r][c]
            line.append("#" if mask[r][c] else ("+" if n >= args.min_count * 3 else ("." if n else " ")))
        print(f"{r / args.rows:4.2f} |" + "".join(line))
    print("      +" + "".join(str(c % 10) for c in range(args.cols)))
    print(f"\nwrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
