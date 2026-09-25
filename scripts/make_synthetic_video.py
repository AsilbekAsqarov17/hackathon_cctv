"""Create a tiny synthetic MP4 for smoke-testing the Part A pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (320, 240))
    for i in range(args.frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = min(270, 10 + i * 5)
        cv2.rectangle(frame, (x, 90), (x + 35, 145), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
