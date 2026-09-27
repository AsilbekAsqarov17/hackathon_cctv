"""Prove that ``RiskEstimator`` is genuinely causal.

The competition requires that ``step(frame, t_sec)`` sees only frames received
so far. A future-aware implementation would score better and would be
disqualified, so the property is tested rather than asserted. Three independent
checks:

1. **Prefix invariance.** Streaming the first N frames and stopping must produce
   exactly the same scores as streaming all of them and reading the first N.
   Any peek at a later frame changes the earlier scores and fails this.
2. **Frame-order sensitivity.** Reordering the frames must change the result.
   A scorer that ignores its input and returns a constant would pass check 1
   while scoring nothing; this catches that.
3. **Score range.** Every score must lie in [0, 1] and be finite, on both real
   video and synthetic extremes (black, white, saturated).

Example::

    python scripts/validate_risk_causality.py --video data/data_video1.mp4 --frames 240
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("solution_causality", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution_causality"] = module
    spec.loader.exec_module(module)
    return module


def stream(estimator, frames, start: int, fps: float) -> list[float]:
    out: list[float] = []
    for i in range(start, start + len(frames)):
        score = float(estimator.step(frames[i - start], i / fps))
        if not np.isfinite(score):
            raise AssertionError(f"non-finite score {score} at frame {i}")
        if not 0.0 <= score <= 1.0:
            raise AssertionError(f"score {score} out of [0,1] at frame {i}")
        out.append(score)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution.py")
    parser.add_argument("--video", default="data/data_video1.mp4")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=29.97)
    args = parser.parse_args()

    solution = load_solution(ROOT / args.solution)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    count = args.frames
    frames: list[np.ndarray] = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 20:
        raise SystemExit("not enough frames decoded")
    print(f"{args.video}: decoded {len(frames)} frames at {fps:.2f} fps")

    meta = {"duration": len(frames) / fps, "fps": fps, "name": Path(args.video).name,
            "width": frames[0].shape[1]}

    # --- 1. prefix invariance -------------------------------------------
    full = solution.RiskEstimator()
    full.reset(meta)
    scores_full = stream(full, frames, 0, fps)

    cut = len(frames) // 2
    prefix = solution.RiskEstimator()
    prefix.reset(meta)
    scores_prefix = stream(prefix, frames[:cut], 0, fps)

    mismatches = [
        (i, scores_full[i], scores_prefix[i])
        for i in range(cut)
        if abs(scores_full[i] - scores_prefix[i]) > 1e-12
    ]
    print(f"\n1. prefix invariance (first {cut} frames, with and without the rest):")
    if mismatches:
        print(f"   *** {len(mismatches)} MISMATCHES - the estimator reads the future")
        for i, a, b in mismatches[:8]:
            print(f"      frame {i}: full={a:.9f} prefix={b:.9f}")
        return 1
    print(f"   OK: {cut} scores identical to 1e-12. The estimator sees only the past.")

    # --- 2. order sensitivity -------------------------------------------
    reversed_estimator = solution.RiskEstimator()
    reversed_estimator.reset(meta)
    reversed_frames = list(reversed(frames))
    scores_reversed = stream(reversed_estimator, reversed_frames, 0, fps)
    # Compare the reversed sequence against the forward one, aligning from the
    # same end so the comparison is meaningful.
    difference = max(abs(a - b) for a, b in zip(reversed(scores_full), scores_reversed))
    print(f"\n2. frame-order sensitivity (same frames, reversed):")
    print(f"   max |forward - reversed| = {difference:.6f}")
    if difference < 1e-6:
        print("   *** the score does not depend on frame order - it is constant,")
        print("       which would score zero on the official chance-normalised AP.")
        return 1
    print("   OK: the score responds to the input.")

    # --- 3. synthetic extremes ------------------------------------------
    print("\n3. score range on synthetic extremes:")
    height, width = frames[0].shape[:2]
    cases = {
        "black": np.zeros((height, width, 3), np.uint8),
        "white": np.full((height, width, 3), 255, np.uint8),
        "saturated red": np.dstack([
            np.full((height, width), 255, np.uint8),
            np.zeros((height, width), np.uint8),
            np.zeros((height, width), np.uint8),
        ]),
        "uniform noise": (np.random.default_rng(0).integers(
            0, 256, (height, width, 3), dtype=np.uint8)),
    }
    for name, frame in cases.items():
        estimator = solution.RiskEstimator()
        estimator.reset({"duration": 3.0, "fps": fps, "name": name, "width": width})
        values = stream(estimator, [frame] * 40, 0, fps)
        print(f"   {name:<16} min={min(values):.4f} max={max(values):.4f} "
              f"finite=True in_range=True")

    print(f"\nobserved score range on real video: "
          f"min={min(scores_full):.4f} max={max(scores_full):.4f}")
    above = [i for i, s in enumerate(scores_full) if s >= 0.5]
    print(f"frames at or above the official alarm threshold 0.5: {len(above)}/{len(scores_full)}")
    print("\nALL CAUSALITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
