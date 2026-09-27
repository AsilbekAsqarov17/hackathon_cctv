"""Report which term dominates the Part B risk score on real frames.

The score is a ``max`` over several interaction terms, so a single term that is
mis-calibrated can drive it to 1.0 forever and hide every other problem. This
replays real frames, records the per-term contributions, and prints their
distributions so the saturation can be attributed to a specific quantity.

Example::

    python scripts/measure_risk_terms.py --video data/data_video2.mp4 --frames 180
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_b import CausalRiskEstimator  # noqa: E402

KEYS = [
    "min_conflict_ttc", "min_ttc", "conflict_pair_count", "close_pair_count",
    "min_distance", "max_closing_speed", "max_deceleration", "max_acceleration",
    "pedestrian_conflict", "vehicle_count", "person_count", "stopped_count",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/data_video2.mp4")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    estimator = CausalRiskEstimator(args.config)
    estimator.reset({"duration": args.frames / fps, "fps": fps,
                     "name": Path(args.video).name, "width": width})

    rows: list[tuple[int, float, dict]] = []
    index = 0
    while index < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.step == 0:
            score = estimator.step(frame, index / fps)
            # `CausalRiskEstimator.last_features` is only populated on the
            # Part A cache path; when the estimator is running its own causal
            # runtime the features live on the runtime.
            features = estimator.last_features
            if not features and estimator.runtime is not None:
                features = estimator.runtime.last_features
            rows.append((index, float(score), dict(features)))
        index += 1
    cap.release()

    if not rows:
        raise SystemExit("no samples decoded")
    print(f"{args.video}: {len(rows)} samples over {index} frames at {fps:.2f} fps\n")
    print(f"{'feature':>22} {'p50':>10} {'p90':>10} {'max':>10}")
    print("-" * 56)
    for key in KEYS:
        values = [r[2].get(key) for r in rows]
        values = [v for v in values if isinstance(v, (int, float))]
        if not values:
            print(f"{key:>22} {'absent':>10}")
            continue
        ordered = sorted(values)
        print(f"{key:>22} {ordered[len(ordered) // 2]:10.2f} "
              f"{ordered[int(len(ordered) * 0.9)]:10.2f} {max(ordered):10.2f}")

    scores = [r[1] for r in rows]
    ordered = sorted(scores)
    print(f"\n{'score':>22} {ordered[len(ordered) // 2]:10.4f} "
          f"{ordered[int(len(ordered) * 0.9)]:10.4f} {max(ordered):10.4f}")
    alarms = sum(1 for s in scores if s >= 0.5)
    print(f"frames at or above the 0.5 alarm threshold: {alarms}/{len(scores)} "
          f"({100.0 * alarms / len(scores):.1f}%)")

    # Attribute the saturation: recompute each term in isolation.
    risk = estimator.risk_config
    horizon = float(risk.get("horizon_sec", 5.0))
    floor = float(risk.get("ttc_floor", 0.45))
    near_px = float(risk.get("near_distance_px", 180.0))
    closing_scale = float(risk.get("closing_speed_scale", 180.0))
    decel_scale = float(risk.get("deceleration_scale", 80.0))
    terms: dict[str, list[float]] = {
        "ttc_ramp": [], "proximity": [], "closing": [], "braking": [], "pedestrian": [],
    }
    for _i, _s, f in rows:
        ttc = f.get("min_conflict_ttc")
        if ttc is None and "conflict_pair_count" not in f:
            ttc = f.get("min_ttc")
        value = 0.0
        if ttc is not None and 0.0 < float(ttc) <= horizon:
            value = floor + (1.0 - floor) * max(0.0, 1.0 - float(ttc) / horizon)
        terms["ttc_ramp"].append(value)
        distance = f.get("min_distance")
        value = 0.0
        if f.get("conflict_pair_count") and distance is not None:
            near = max(0.0, 1.0 - float(distance) / near_px)
            value = 0.32 * near * near
        terms["proximity"].append(value)
        closing = float(f.get("max_closing_speed", 0.0))
        terms["closing"].append(0.35 * min(1.0, closing / closing_scale) if closing > 0 else 0.0)
        braking = float(f.get("max_deceleration", 0.0))
        terms["braking"].append(0.30 * min(1.0, braking / decel_scale) if braking > 0 else 0.0)
        terms["pedestrian"].append(0.65 * float(f.get("pedestrian_conflict", 0.0)))

    print(f"\nper-term contribution (the score is their max):")
    print(f"{'term':>12} {'p50':>8} {'p90':>8} {'max':>8} {'frames>=0.5':>13}")
    print("-" * 54)
    for name, values in terms.items():
        if not values:
            continue
        ordered_v = sorted(values)
        share = sum(1 for v in values if v >= 0.5)
        print(f"{name:>12} {ordered_v[len(ordered_v) // 2]:8.3f} "
              f"{ordered_v[int(len(ordered_v) * 0.9)]:8.3f} {max(ordered_v):8.3f} "
              f"{share:8d}/{len(values):<4d}")
    print("\nA term with many frames >= 0.5 is what saturates the score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
