"""Test whether a candidate signal head is readable from this camera.

Prints, over frames spread across the video, the strongest red / green / yellow
evidence inside each candidate ROI. A head that faces away from the camera never
saturates, so this distinguishes "signal exists but is unreadable here" from
"no signal", without inventing a state.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--rois", nargs="+", required=True,
                        help="name:x0,y0,x1,y1 ...")
    parser.add_argument("--samples", type=int, default=240)
    args = parser.parse_args()

    rois = []
    for spec in args.rois:
        name, rect = spec.split(":")
        x0, y0, x1, y1 = (int(v) for v in rect.split(","))
        rois.append((name, x0, y0, x1, y1))

    capture = cv2.VideoCapture(args.video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // args.samples)
    stats = {name: {"red": [], "green": [], "yellow": []} for name, *_ in rois}
    read = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for name, x0, y0, x1, y1 in rois:
                patch = hsv[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                h, s, v = patch[..., 0].astype(int), patch[..., 1].astype(int), patch[..., 2].astype(int)
                red = ((h < 12) | (h > 168)) & (s > 110) & (v > 90)
                green = (h > 42) & (h < 88) & (s > 110) & (v > 90)
                yellow = (h >= 20) & (h < 42) & (s > 110) & (v > 90)
                stats[name]["red"].append(int(red.sum()))
                stats[name]["green"].append(int(green.sum()))
                stats[name]["yellow"].append(int(yellow.sum()))
            read += 1
        index += 1
    capture.release()

    print(f"frames={total} sampled={read} stride={stride}")
    print(f"{'roi':<22} {'px/frame red':>13} {'green':>7} {'yellow':>7}   verdict")
    print("-" * 70)
    for name, *_ in rois:
        r = np.array(stats[name]["red"]) if stats[name]["red"] else np.zeros(1)
        g = np.array(stats[name]["green"]) if stats[name]["green"] else np.zeros(1)
        y = np.array(stats[name]["yellow"]) if stats[name]["yellow"] else np.zeros(1)
        best = max(r.max(), g.max(), y.max())
        verdict = "READABLE" if best >= 8 else "not readable from this viewpoint"
        # An out-of-frame ROI reads as an all-zero patch; np.max on an empty
        # selection is a float, so cast before formatting.
        print(f"{name:<22} {r.mean():6.2f}/{int(r.max()):<6d} {g.mean():5.2f}/{int(g.max()):<5d} "
              f"{y.mean():5.2f}/{int(y.max()):<5d}   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
