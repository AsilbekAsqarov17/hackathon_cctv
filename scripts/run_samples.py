"""Run the official submission harness over the sample videos and keep the output.

This is the end-to-end check that matters: it calls ``solution.detect_events``
and ``solution.RiskEstimator`` exactly as the organizers' ``run_submission.py``
does, so it exercises the real interface, the real time budget and the real
output format. Anything that works here works in the submission.

Writes ``predictions_samples.json`` in the format ``evaluate.py`` validates.

Example::

    python scripts/run_samples.py --videos data/data_video1.mp4 data/data_video2.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFICIAL_CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("solution_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution_under_test"] = module
    spec.loader.exec_module(module)
    for name in ("CLASSES", "detect_events", "RiskEstimator"):
        if not hasattr(module, name):
            raise AttributeError(f"{path} must define {name}")
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--solution", default="solution.py")
    parser.add_argument("--out", default="predictions_samples.json")
    parser.add_argument("--time-factor", type=float, default=3.0)
    parser.add_argument("--risk-stride", type=int, default=1)
    parser.add_argument("--save-risk", action="store_true",
                        help="store the full per-frame risk curve (large)")
    args = parser.parse_args()

    solution = load_solution(ROOT / args.solution)
    labels = set(OFFICIAL_CLASSES)
    unknown = [c for c in solution.CLASSES if c not in labels]
    if unknown:
        raise SystemExit(f"solution.CLASSES has non-official labels: {unknown}")

    videos: dict[str, dict] = {}
    for name in args.videos:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"missing video: {path}")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frames / fps if fps else 0.0
        budget = args.time_factor * duration
        print(f"\n=== {path.name}: {duration:.1f}s, budget {budget:.1f}s "
              f"({args.time_factor}x) ===", flush=True)

        started = time.perf_counter()
        events = solution.detect_events(str(path))
        part_a = time.perf_counter() - started
        print(f"  Part A: {part_a:.1f}s ({part_a / max(duration, 1e-6):.2f}x video) "
              f"-> {len(events)} events", flush=True)

        # Part B is driven exactly as the harness drives it: every frame, with
        # the time already spent on Part A counted against the same budget.
        # The meta keys must match run_submission.video_meta() exactly, because
        # `video_id` is the key the Part A feature cache is published under. If
        # it does not match, the estimator silently falls back to running its own
        # detector on every frame and the budget is blown.
        risk_state = solution.RiskEstimator()
        risk_state.reset({"video_id": path.name, "fps": fps,
                          "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                          "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                          "n_frames": frames})
        started = time.perf_counter()
        curve: list[list[float]] = []
        cap = cv2.VideoCapture(str(path))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.risk_stride == 0:
                score = float(risk_state.step(frame, index / fps))
                score = min(1.0, max(0.0, score))
                if args.save_risk:
                    curve.append([round(index / fps, 3), round(score, 4)])
            index += 1
        cap.release()
        part_b = time.perf_counter() - started
        print(f"  Part B: {part_b:.1f}s ({part_b / max(duration, 1e-6):.2f}x video), "
              f"{index} frames", flush=True)

        total = part_a + part_b
        verdict = "WITHIN BUDGET" if total <= budget else "*** OVER BUDGET ***"
        print(f"  total: {total:.1f}s / {budget:.1f}s  ({total / max(duration, 1e-6):.2f}x) "
              f"{verdict}", flush=True)

        record: dict = {"duration": round(duration, 3), "fps": round(fps, 3),
                        "events": [[round(float(e[0]), 3), round(float(e[1]), 3), str(e[2])]
                                   for e in events]}
        if args.save_risk:
            record["risk"] = curve
        videos[path.name] = record

    payload = {"videos": videos}
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    total_events = sum(len(v["events"]) for v in videos.values())
    counts: dict[str, int] = {}
    for record in videos.values():
        for event in record["events"]:
            counts[event[2]] = counts.get(event[2], 0) + 1
    print(f"{total_events} events across {len(videos)} videos")
    for label in sorted(counts):
        print(f"  {label:<20} {counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
