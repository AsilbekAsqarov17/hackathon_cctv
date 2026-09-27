"""Trajectory-based motion estimation for Part A rules.

Both ``stopped_vehicle`` and ``wrong_way`` must judge motion over a time window
rather than from a single frame, because a single frame cannot distinguish
"stopped" from "jittering detector output" and one noisy frame must not flip a
rule.

Everything here works on bottom-centre history, the road position established
for this camera. Two quantities are used:

``net``   straight-line displacement across the window. This is the true motion.
``path``  summed step length. This is what separates real motion from
          detector/tracker jitter: a jittering box accumulates a large ``path``
          while its ``net`` stays near zero.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from ..scene.geometry import Point, normalized_vector

Point3 = tuple[float, float, float]


def trailing_window(points: Sequence[Point3], window: float, now: float | None = None) -> list[Point3]:
    """History samples inside ``(now - window, now]``.

    The window is bounded on *both* sides. Bounding only the start silently
    mixes in samples newer than ``now`` whenever a caller passes a history that
    extends past the evaluation time, which turns a 2.5 s window into the whole
    track and reports a single large displacement as if it were instantaneous
    motion. With ``now=None`` the newest sample is the end of the window, so
    this matches the old one-sided behaviour exactly.
    """
    if not points:
        return []
    end = points[-1][0] if now is None else float(now)
    cutoff = end - max(0.0, float(window))
    return [p for p in points if cutoff < p[0] <= end]


def net_displacement(points: Sequence[Point3]) -> float:
    """Straight-line distance between the first and last window sample."""
    if len(points) < 2:
        return 0.0
    return math.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2])


def path_length(points: Sequence[Point3]) -> float:
    """Summed step length across the window."""
    total = 0.0
    for index in range(1, len(points)):
        total += math.hypot(
            points[index][1] - points[index - 1][1],
            points[index][2] - points[index - 1][2],
        )
    return total


def window_speed(points: Sequence[Point3], window: float, now: float | None = None) -> float:
    """Net speed in px/s over the window (0.0 when the window is too short)."""
    recent = trailing_window(points, window, now)
    if len(recent) < 2:
        return 0.0
    span = recent[-1][0] - recent[0][0]
    if span <= 1e-3:
        return 0.0
    return net_displacement(recent) / span


def window_heading(
    points: Sequence[Point3], window: float, now: float | None = None
) -> tuple[Point, float] | None:
    """Unit heading across the window plus the net distance covered.

    Returns ``None`` when the window is too short or the vehicle has barely
    moved, so callers must treat "unknown heading" as "not wrong-way" rather
    than defaulting to a direction.
    """
    recent = trailing_window(points, window, now)
    if len(recent) < 2:
        return None
    span = recent[-1][0] - recent[0][0]
    if span <= 1e-3:
        return None
    vector = (recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2])
    unit = normalized_vector(vector)
    if unit is None:
        return None
    return unit, net_displacement(recent)


def is_stationary(
    points: Sequence[Point3],
    window: float,
    net_max: float,
    path_max: float,
    now: float | None = None,
    min_samples: int = 3,
) -> bool:
    """True when the vehicle is effectively parked over the whole window.

    Requires *both* a small net displacement and a small path length. The path
    check is what tolerates jitter without also tolerating a slow crawl.
    """
    recent = trailing_window(points, window, now)
    if len(recent) < min_samples:
        return False
    return net_displacement(recent) <= net_max and path_length(recent) <= path_max


def is_upstream_of_line(point: Point, line: Any) -> bool:
    """True when ``point`` sits on the approach side of a directed line.

    ``line`` is a :class:`~src.scene.config.SceneLine`; its ``direction`` is the
    permitted travel direction, so a vehicle upstream of it is still approaching
    rather than having already crossed.
    """
    a, b = line.segment
    if line.direction is None:
        return False
    along = (point[0] - a[0]) * line.direction[0] + (point[1] - a[1]) * line.direction[1]
    return along < 0.0
