"""Benchmark the causal Part B curve on one video."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_a import run_part_a
from solution import RiskEstimator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    start = time.perf_counter()
    events = run_part_a(args.video, args.config)
    part_a_time = time.perf_counter() - start
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    estimator = RiskEstimator()
    estimator.reset({"video_id": Path(args.video).name, "fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0), "n_frames": n_frames})
    values = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(float(estimator.step(frame, index / fps)))
        index += 1
    cap.release()
    positive = sum(value >= 0.5 for value in values)
    print(f"events={len(events)} frames={len(values)} positive_alarm_frames={positive} max={max(values, default=0.0):.3f} part_a_sec={part_a_time:.2f} total_sec={time.perf_counter()-start:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
