#!/usr/bin/env python3
"""
preview_scene_fill.py - scene overlay with translucent area fills.

scripts/preview_scene.py draws outlines only, which makes it hard to judge a
road polygon: an outline that grazes a pavement looks identical to one that
covers it. This version fills the regions at partial opacity so the covered
area is unambiguous, which is what actually matters for the road/crosswalk
tests.

    python scripts/preview_scene_fill.py VIDEO SCENE_JSON --frame 4500 -o out.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import load_scene_config  # noqa: E402


def poly_of(points) -> np.ndarray:
    return np.asarray([(int(x), int(y)) for x, y in points], dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("scene")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("-o", "--output", default="scene_fill.jpg")
    ap.add_argument("--alpha", type=float, default=0.35)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    h, w = frame.shape[:2]
    cfg = load_scene_config(args.scene, w, h)

    overlay = frame.copy()
    for road in cfg.road_polygons:
        cv2.fillPoly(overlay, [poly_of(road)], (0, 150, 255))
    for crossing in cfg.crossings:
        cv2.fillPoly(overlay, [poly_of(crossing)], (0, 0, 255))
    # Exclusions are holes in the road fill, so they are drawn last and in a
    # third colour: green must read as "not road" while the orange road fill
    # still shows through around it.
    for exclusion in cfg.road_exclusions:
        cv2.fillPoly(overlay, [poly_of(exclusion)], (0, 255, 0))
    out = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)

    for road in cfg.road_polygons:
        cv2.polylines(out, [poly_of(road)], True, (0, 255, 255), 3)
    for crossing in cfg.crossings:
        cv2.polylines(out, [poly_of(crossing)], True, (0, 0, 255), 4)
    for index, exclusion in enumerate(cfg.road_exclusions):
        pts = poly_of(exclusion)
        cv2.polylines(out, [pts], True, (0, 255, 0), 4)
        cv2.putText(out, f"NOT ROAD {index}", (int(pts[:, 0].min()) + 6, int(pts[:, 1].min()) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for line in cfg.stop_lines:
        cv2.line(out, poly_of(line.segment)[0], poly_of(line.segment)[1], (0, 0, 255), 6)
    for line in cfg.solid_lines:
        cv2.line(out, poly_of(line.segment)[0], poly_of(line.segment)[1], (255, 0, 0), 4)
    for lane in cfg.lanes:
        cv2.polylines(out, [poly_of(lane.polygon)], True, (255, 0, 255), 4)
    for light in cfg.traffic_lights:
        x1, y1, x2, y2 = (int(v) for v in light.roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(out, light.light_id, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
