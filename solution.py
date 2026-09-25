"""
solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Labels must come from CLASSES. You may REMOVE classes you never predict;
do not add new ids.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The organizers load this file by path, so make the project root importable
# regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file. You may open it any way you like
            (OpenCV, decord, PyAV, ffmpeg), read it several times, sample
            frames, run batched models — anything goes.

    Returns:
        A list of events, each ``[start_sec, end_sec, label]`` with
        ``0 <= start_sec < end_sec <= duration`` (floats, seconds from the
        first frame) and ``label in CLASSES``. Return ``[]`` if nothing
        happened. Segments of the same class must not overlap.

    A typical pipeline:
        1. sample frames (every 2nd–5th frame is usually enough),
        2. detect road users (YOLO / RT-DETR) and track them (ByteTrack),
        3. turn trajectories + scene layout (lanes, stop line, crossing)
           into per-frame flags for each class,
        4. merge consecutive flags into segments, drop blips < 0.5 s,
           merge gaps < 1 s,
        5. optionally re-score `accident` / `near_miss` candidates with a
           learned clip classifier.
    """
    from src.part_a import run_part_a

    return run_part_a(video_path)


from src.part_b import CausalRiskEstimator


class RiskEstimator(CausalRiskEstimator):
    """Part B causal accident-anticipation estimator.

    The implementation is causal by construction. When the harness has run
    Part A first, it consumes only compact feature samples whose timestamps
    are less than or equal to the current frame. If no cache is available, it
    falls back to its own causal detector/tracker runtime and never opens the
    video file inside ``step``.
    """
