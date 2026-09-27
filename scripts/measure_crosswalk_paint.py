#!/usr/bin/env python3
"""
measure_crosswalk_paint.py -- locate the painted zebra stripes by measurement.

The scene file's crosswalk polygons were hand-drawn and one of them is
displaced. Rather than nudging coordinates by eye, this finds the paint itself:
zebra stripes are bright, low-saturation, elongated, and they sit on the
carriageway. Restricting the search to the road area removes the two things
that defeated a naive brightness threshold before -- windscreens and shop
windows, which are bright but not on the road.

Prints component bounding boxes in normalised coordinates so a polygon can be
fitted to them, and writes an overlay for visual confirmation.

    python scripts/measure_crosswalk_paint.py data/samples_small/C3902_small.mp4 --frame 3150
"""
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=3150)
    ap.add_argument("--scene", default="configs/scenes/tashkent_intersection.json")
    ap.add_argument("--min-area", type=int, default=700)
    ap.add_argument(
        "--search",
        default=None,
        help="x0,y0,x1,y1 in normalised coords. Restrict the stripe search to "
             "this window and drop the road mask, which is necessary because "
             "the bottom-left crossing lies outside the road polygons -- so "
             "restricting to the carriageway hides exactly the paint being "
             "looked for.",
    )
    ap.add_argument("--fit", action="store_true",
                    help="also fit a min-area rectangle to the stripe pixels")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="normalised padding applied to the fitted quad")
    ap.add_argument("--exclude", action="append",
                    help="x0,y0,x1,y1 normalised; drop stripe components inside "
                         "this box. Used to reject island kerbstones, which are "
                         "grey concrete and survive the brick-hue exclusion.")
    ap.add_argument("-o", default="/home/axumfa/H/diag/paint_overlay.jpg")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read frame", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    scene = load_scene_config(str(ROOT / args.scene), w, h)

    # Road area, generously dilated: stripes can sit at the very edge of paint.
    road = np.zeros((h, w), np.uint8)
    for poly in list(scene.road_polygons) + ([scene.road_polygon] if scene.road_polygon else []):
        cv2.fillPoly(road, [to_px(poly, w, h)], 1)
    road = cv2.dilate(road, np.ones((31, 31), np.uint8))
    if args.search:
        sx0, sy0, sx1, sy1 = (float(v) for v in args.search.split(","))
        win = np.zeros((h, w), np.uint8)
        win[int(sy0 * h):int(sy1 * h), int(sx0 * w):int(sx1 * w)] = 1
        road = win

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bright = ((val >= 125) & (sat <= 90)).astype(np.uint8)
    # The refuge islands are muted mauve brick: hue 115-160, which is bright and
    # unsaturated enough to pass a plain "bright paint" test and so registers as
    # a giant false stripe. Measured off this footage; asphalt sits at hue
    # 108-109 and zebra paint is desaturated, so the band separates cleanly.
    brick = ((hue >= 115) & (hue <= 160)).astype(np.uint8)
    bright[cv2.dilate(brick, np.ones((5, 5), np.uint8)) > 0] = 0
    bright = cv2.bitwise_and(bright, road)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
    found = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < args.min_area:
            continue
        # Stripes are elongated; a compact blob is a pool of light or a reflection.
        if max(bw, bh) < 2.2 * min(bw, bh):
            continue
        found.append({"area": area, "x": x, "y": y, "w": bw, "h": bh,
                      "cx": cent[i][0], "cy": cent[i][1]})

    overlay = frame.copy()
    for poly in scene.road_polygons:
        cv2.polylines(overlay, [to_px(poly, w, h)], True, (255, 0, 255), 2)
    for f in found:
        cv2.rectangle(overlay, (f["x"], f["y"]), (f["x"] + f["w"], f["y"] + f["h"]), (0, 255, 255), 2)
        cv2.putText(overlay, str(f["area"]), (f["x"], max(10, f["y"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if args.fit:
        # Fit only to the pixels of components that passed the stripe shape and
        # area tests. Fitting to the raw bright mask instead lets lane markings
        # and kerbstones drag the rectangle across paint that is not part of the
        # crossing.
        acc = np.zeros((h, w), np.uint8)
        for f in found:
            acc[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"]] = 1
        for spec in args.exclude or []:
            ex0, ey0, ex1, ey1 = (float(v) for v in spec.split(","))
            acc[int(ey0 * h):int(ey1 * h), int(ex0 * w):int(ex1 * w)] = 0
        ys, xs = np.nonzero(acc)
        if len(xs) < 200:
            print("not enough stripe pixels to fit", file=sys.stderr)
            return 1
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        box = cv2.boxPoints(((cx, cy), (rw, rh), ang))
        box = np.vstack([box, box[0]])
        quad = [[round(float(px) / w, 4), round(float(py) / h, 4)] for px, py in box]
        pad = float(args.pad)
        cxn, cyn = cx / w, cy / h
        padded = []
        for qx, qy in quad:
            px = qx + (pad if qx >= cxn else -pad)
            py = qy + (pad if qy >= cyn else -pad)
            padded.append([round(min(1.0, max(0.0, px)), 4), round(min(1.0, max(0.0, py)), 4)])
        quad = padded
        print("\nfitted quad (normalised, padded):")
        print(json.dumps(quad))
        cv2.polylines(overlay, [(np.array(quad, np.float32) * [w, h]).astype(np.int32)],
                      True, (0, 0, 255), 3)
        for i, (px, py) in enumerate(quad):
            cv2.circle(overlay, (int(px * w), int(py * h)), 7, (0, 0, 255), -1)
            cv2.putText(overlay, str(i), (int(px * w) + 8, int(py * h) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(args.o, overlay)

    print(f"{args.video} frame {args.frame}: {len(found)} stripe candidates")
    print(f"{'area':>7} {'x0':>5}{'y0':>5}{'x1':>5}{'y1':>5}   normalised x0,y0,x1,y1")
    for f in sorted(found, key=lambda z: (z["y"], z["x"])):
        print(f"{f['area']:7d} {f['x']:5d}{f['y']:5d}{f['x']+f['w']:5d}{f['y']+f['h']:5d}   "
              f"({f['x']/w:.3f}, {f['y']/h:.3f}, {(f['x']+f['w'])/w:.3f}, {(f['y']+f['h'])/h:.3f})")
    print(f"\noverlay -> {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
