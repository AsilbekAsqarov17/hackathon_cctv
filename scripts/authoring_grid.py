"""Authoring aid: save video frames with a labelled pixel grid.

Used to read off geometry coordinates from the real 4K frames instead of
guessing. Supports full-frame overviews and zoomed tiles with global pixel
labels, so a coordinate read from a tile maps straight back to the source
frame.

Examples::

    python scripts/authoring_grid.py data/data_video1.mp4 --frame 7200 --mode overview
    python scripts/authoring_grid.py data/data_video1.mp4 --frame 7200 --mode tile --tile 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def draw_grid(image, step: int, offset: tuple[int, int] = (0, 0), thickness: int = 2):
    """Overlay a labelled grid. ``offset`` is the global coord of pixel (0,0)."""
    out = image.copy()
    height, width = out.shape[:2]
    ox, oy = offset
    for x in range(0, width, step):
        gx = ox + x
        major = gx % (step * 5) == 0
        cv2.line(out, (x, 0), (x, height), (0, 0, 255) if major else (0, 140, 255), thickness if major else 1)
        if major:
            cv2.putText(out, str(gx), (x + 4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for y in range(0, height, step):
        gy = oy + y
        major = gy % (step * 5) == 0
        cv2.line(out, (0, y), (width, y), (0, 0, 255) if major else (0, 140, 255), thickness if major else 1)
        if major:
            cv2.putText(out, str(gy), (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["overview", "tile", "zoom"], default="overview")
    parser.add_argument("--tile", type=int, default=0, help="0=TL 1=TR 2=BL 3=BR")
    parser.add_argument("--tile-size", type=int, default=1920)
    parser.add_argument("--crop", default="", help="zoom mode: x1,y1,x2,y2 in source pixels")
    parser.add_argument("--scale", type=float, default=4.0, help="zoom magnification")
    parser.add_argument("--step", type=int, default=200)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.frame}")
    height, width = frame.shape[:2]

    if args.mode == "overview":
        canvas = draw_grid(frame, args.step)
    elif args.mode == "zoom":
        x1, y1, x2, y2 = (int(v) for v in args.crop.split(","))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        patch = frame[y1:y2, x1:x2]
        patch = cv2.resize(patch, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        canvas = draw_grid(patch, max(10, int(args.step * args.scale)), offset=(x1, y1))
    else:
        size = args.tile_size
        col = args.tile % 2
        row = args.tile // 2
        x0 = col * (width // 2)
        y0 = row * (height // 2)
        x1 = min(width, x0 + size)
        y1 = min(height, y0 + size)
        canvas = draw_grid(frame[y0:y1, x0:x1], args.step, offset=(x0, y0))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"frame={args.frame} size={width}x{height} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
