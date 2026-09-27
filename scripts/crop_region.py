"""Save annotated video crops so a scene region can be inspected directly.

Example::

    python scripts/crop_region.py data/data_video1.mp4 10 30 60 90 --rect 3100 550 3840 1350
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("times", nargs="+", type=float)
    parser.add_argument("--rect", nargs=4, type=int, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", default="data/authoring/crops")
    args = parser.parse_args()

    x0, y0, x1, y1 = args.rect
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    wanted = {round(t * args.fps): t for t in args.times}
    got: dict[float, str] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            crop = frame[y0:y1, x0:x1].copy()
            if args.scale != 1.0:
                crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale)
            label = f"t={wanted[index]:.1f}s f={index}"
            cv2.putText(crop, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2)
            path = out_dir / f"crop_{wanted[index]:07.1f}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            got[wanted[index]] = str(path)
        index += 1
        if len(got) == len(wanted):
            break
    capture.release()
    for t in sorted(got):
        print(f"t={t:7.1f}s  {got[t]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
