#!/usr/bin/env python3
"""
measure_islands.py -- locate the raised refuge islands by colour.

The islands are muted mauve/terracotta brick (measured OpenCV hue 115-160)
against grey asphalt (hue 108-109), so they separate cleanly in HSV without
hand-drawn coordinates. Run on both development clips because the scene file
is shared between them and the two cameras see the same junction from
different angles.

    python scripts/measure_islands.py data/samples_small/C3897_small.mp4 --frame 3150
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


def to_px(poly, w, h):
    a = np.array(poly, np.float32)
    return (a * [w, h] if a.max() <= 1.5 else a).astype(np.int32)


def island_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((h >= 115) & (h <= 160) & (s >= 40) & (s <= 150) & (v >= 105)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 3000:
            keep[lab == i] = 1
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=3150)
    ap.add_argument("--scene", default="configs/scenes/tashkent_intersection.json")
    ap.add_argument("-o", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    h, w = frame.shape[:2]
    scene = load_scene_config(str(ROOT / args.scene), w, h)

    road = np.zeros((h, w), np.uint8)
    for poly in list(scene.road_polygons) + ([scene.road_polygon] if scene.road_polygon else []):
        cv2.fillPoly(road, [to_px(poly, w, h)], 1)

    mask = island_mask(frame)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    print(f"{Path(args.video).name} frame {args.frame}")
    print(f"{'area':>7}  normalised bbox                 inside_road  inside_crossing")
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= 3000:
            continue
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Grow by the kerb, which is grey concrete and so is not in the mask.
        pad_x, pad_y = int(0.012 * w), int(0.010 * h)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        sub = np.zeros((h, w), np.uint8)
        sub[y0:y1, x0:x1] = 1
        frac_road = float((sub & road).sum()) / max(1, sub.sum())
        cross_hit = any(
            cv2.intersectConvexConvex(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32),
                np.asarray(c, np.int32),
            )[0] > 0
            for c in scene.crossings
        )
        print(f"{area:7d}  ({x0/w:.3f},{y0/h:.3f})-({x1/w:.3f},{y1/h:.3f})   "
              f"road={100*frac_road:5.1f}%   crossing={cross_hit}")

    if args.o:
        vis = frame.copy()
        vis[mask > 0] = (0, 0, 255)
        cv2.polylines(vis, [to_px(p, w, h) for p in scene.road_polygons], True, (0, 255, 255), 2)
        Path(args.o).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.o, vis)
        print("overlay ->", args.o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
