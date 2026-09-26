#!/usr/bin/env python3
"""
make_review_sheets.py - contact sheets for annotating the sample clips.

Annotating a five-minute clip by scrubbing back and forth is the slowest part
of building a dev set, and it is also the part that decides whether any of the
rest of the work can be measured. This script renders a clip as a grid of
timestamped stills so events can be spotted in a few passes, then renders
dense strips around anything suspicious so exact boundaries can be read off.

    # overview: one still every --step seconds, as PNG grids
    python scripts/make_review_sheets.py data/samples_small/C3897_small.mp4 \
        -o data/review --step 2

    # zoom on one moment, 10 stills per second, for boundary work
    python scripts/make_review_sheets.py clip.mp4 --start 120 --duration 6 \
        --step 0.1 -o data/review --prefix zoom_120

Sheets are numbered and ordered left-to-right, top-to-bottom, and every tile is
labelled with its timestamp in seconds, so a frame can be cited directly in
the ground-truth file.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def tile(frame: np.ndarray, width: int, label: str) -> np.ndarray:
    height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
    out = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    # A dark strip keeps the timestamp legible over bright carriageway.
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="data/review")
    ap.add_argument("--prefix", default=None, help="output filename prefix (default: video stem)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--step", type=float, default=2.0, help="seconds between stills")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-width", type=int, default=384)
    ap.add_argument("--per-sheet", type=int, default=25, help="tiles per PNG (cols*rows)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(args.start * fps)))
    if args.duration:
        last_frame = min(total - 1, start_frame + int(round(args.duration * fps)))
    else:
        last_frame = total - 1
    step_frames = max(1, int(round(args.step * fps)))

    # Sequential decode is much faster than seeking per tile on long GOPs.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.video).stem

    rows = max(1, args.per_sheet // args.cols)
    tiles: list[np.ndarray] = []
    sheet_index = 0
    frame_index = start_frame
    grabbed = 0
    last_tile_time = -1.0

    def flush() -> None:
        nonlocal tiles, sheet_index
        if not tiles:
            return
        blank = np.zeros_like(tiles[0])
        while len(tiles) % (args.cols * rows):
            tiles.append(blank.copy())
        grid = [np.hstack(tiles[i : i + args.cols]) for i in range(0, len(tiles), args.cols)]
        width = max(row.shape[1] for row in grid)
        grid = [np.hstack([row, np.zeros((row.shape[0], width - row.shape[1], 3), np.uint8)]) for row in grid]
        path = out_dir / f"{prefix}_sheet{sheet_index:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(grid))
        print(f"  wrote {path.name}  ({len(tiles)} tiles)")
        tiles = []
        sheet_index += 1

    while frame_index <= last_frame:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / fps
        if grabbed % step_frames == 0:
            tiles.append(tile(frame, args.tile_width, f"t={t:7.2f}s  f={frame_index}"))
            if len(tiles) >= args.per_sheet:
                flush()
        grabbed += 1
        frame_index += 1
    cap.release()
    flush()
    print(f"{args.video}: {grabbed} frames read from t={args.start:.2f}s, step {args.step}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
