"""Trajectory and scene-geometry helpers shared by the Part A rules.

Everything here works on track state plus the calibrated scene, never on pixels
directly, so each helper is unit-testable without a video. The functions fall
into four groups:

* **line frame** — expressing a point in the coordinates of a calibrated line,
  which is what "upstream of the stop line" and "past its end" mean;
* **line crossing** — direction-aware detection that a vehicle's FRONT actually
  passed over a calibrated line, rejecting vehicles that are leaving the
  junction, reversing, or merely drifting along it;
* **manoeuvres** — recovering a U-turn or a turn from trajectory history rather
  than from one frame;
* **interaction** — time-to-collision, closing rate and evasive magnitude for
  the ``accident`` / ``near_miss`` pair.

All helpers are causal: they read only history at or before the supplied
timestamp.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..contracts import TrackState
from ..scene.geometry import (
    Point,
    angle_between,
    normalized_vector,
    point_segment_distance,
    segments_intersect,
    vector_dot,
)

Point3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# line frame
# ---------------------------------------------------------------------------

def line_length(line: Any) -> float:
    (ax, ay), (bx, by) = line.segment
    return math.hypot(bx - ax, by - ay)


def along_line(point: Point, line: Any) -> float:
    """Signed distance from the line's start point, along its travel direction.

    Negative means upstream: the vehicle has not reached the line yet. This is
    the quantity queue suppression is built on, so it is defined for any line
    that declares a direction and returns NaN otherwise.
    """
    if line.direction is None:
        return float("nan")
    a = line.segment[0]
    return (point[0] - a[0]) * line.direction[0] + (point[1] - a[1]) * line.direction[1]


def across_line(point: Point, line: Any) -> float:
    """Position of ``point`` along the line segment, 0 at its start.

    A value outside ``[0, line_length]`` means the point lies beyond one end of
    the segment, which is how "the line does not cover this part of the
    carriageway" is detected.
    """
    (ax, ay), (bx, by) = line.segment
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-9:
        return 0.0
    ux, uy = (bx - ax) / length, (by - ay) / length
    return (point[0] - ax) * ux + (point[1] - ay) * uy


def distance_to_line(point: Point, line: Any) -> float:
    return point_segment_distance(point, line.segment[0], line.segment[1])


def on_segment(point: Point, line: Any, margin: float = 0.0) -> bool:
    """True when ``point`` lies over the drawn extent of ``line``."""
    length = line_length(line)
    return -margin <= across_line(point, line) <= length + margin


def side_of_line(point: Point, line: Any) -> int:
    """+1 or -1 for the two sides of the drawn segment, 0 when on it.

    This is the geometric test for "was the vehicle upstream and is it now
    downstream". It is deliberately *not* the same as the sign of
    :func:`along_line`: that measures from the segment's start point along the
    declared direction, so a lateral offset contributes to it, whereas the side
    test only depends on which side of the painted line the point falls. Using
    the side test keeps the crossing definition correct even when the declared
    direction is not exactly perpendicular to the drawn segment.
    """
    (ax, ay), (bx, by) = line.segment
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-9:
        return 0
    ux, uy = (bx - ax) / length, (by - ay) / length
    value = (point[0] - ax) * (-uy) + (point[1] - ay) * ux
    if abs(value) <= 1e-6:
        return 0
    return 1 if value > 0 else -1


# ---------------------------------------------------------------------------
# line crossing
# ---------------------------------------------------------------------------

def front_path(track: TrackState, direction: Point, window: float,
               timestamp: float) -> list[Point3]:
    """Leading edge of the vehicle over the trailing window.

    Prefers the historical box record, which is exact, and falls back to the
    bottom-centre path so the helper still works on a track whose box history is
    not populated (for example a hand-built track in a test).
    """
    path = track.front_path(direction, window, timestamp)
    if len(path) >= 2:
        return path
    return [p for p in track.bottom_history
            if p[0] > timestamp - max(0.0, window) and p[0] <= timestamp + 1e-6]


def front_crossing(
    track: TrackState,
    line: Any,
    timestamp: float,
    window: float = 3.0,
    require_approach: bool = True,
    min_dot: float = 0.25,
) -> float | None:
    """Timestamp at which the vehicle's front crossed ``line``, else ``None``.

    Direction-aware, which is what keeps the rule honest:

    * the heading must have a positive dot product with the line's direction, so
      a vehicle leaving the junction, reversing, or already driving away cannot
      register a crossing;
    * the front path must actually intersect the drawn segment, so a vehicle
      that passes beside its end is not counted;
    * with ``require_approach`` the vehicle must be upstream before and
      downstream after, so a merge along the line is not a crossing.

    Returns the timestamp of the sample *before* the crossing, so the reported
    event start is the last moment the vehicle was still on the approach side.
    """
    direction = line.direction
    unit = normalized_vector(direction) if direction is not None else None
    if unit is None:
        return None

    heading = normalized_vector(track.velocity)
    if heading is not None and vector_dot(heading, unit) < min_dot:
        return None

    path = front_path(track, direction, window, timestamp)
    if len(path) < 2:
        return None
    for i in range(len(path) - 1, 0, -1):
        t0, ax0, ay0 = path[i - 1]
        t1, ax1_, ay1_ = path[i]
        if not segments_intersect((ax0, ay0), (ax1_, ay1_), line.segment[0], line.segment[1]):
            continue
        if require_approach:
            # The vehicle must actually change sides of the painted line. The
            # side test is used rather than the along-line sign because the
            # along value is measured from the segment's start point, so a
            # lateral offset along the carriageway contributes to it and can
            # mask a crossing that is geometrically unambiguous.
            before_side = side_of_line((ax0, ay0), line)
            after_side = side_of_line((ax1_, ay1_), line)
            if after_side == 0 or before_side == 0 or before_side == after_side:
                continue
        return t0
    return None


def crossed_line(
    track: TrackState,
    line: Any,
    timestamp: float,
    window: float = 3.0,
    min_dot: float = 0.25,
) -> bool:
    """True when the vehicle's front crossed ``line`` in either direction.

    Used by ``solid_line_crossing``, where crossing matters regardless of which
    way the vehicle was permitted to travel.
    """
    direction = line.direction
    unit = normalized_vector(direction) if direction is not None else None
    if unit is None:
        return False
    heading = normalized_vector(track.velocity)
    if heading is not None and abs(vector_dot(heading, unit)) < min_dot:
        return False
    path = front_path(track, direction, window, timestamp)
    if len(path) < 2:
        return False
    for i in range(len(path) - 1, 0, -1):
        if segments_intersect((path[i - 1][1], path[i - 1][2]), (path[i][1], path[i][2]),
                              line.segment[0], line.segment[1]):
            return True
    return False


# ---------------------------------------------------------------------------
# manoeuvres
# ---------------------------------------------------------------------------

def net_heading(track: TrackState, window: float, timestamp: float) -> Point | None:
    """Unit heading of the net displacement across the trailing window."""
    points = [p for p in track.bottom_history
              if p[0] > timestamp - max(0.0, window) and p[0] <= timestamp + 1e-6]
    if len(points) < 2:
        points = [p for p in track.history
                  if p[0] > timestamp - max(0.0, window) and p[0] <= timestamp + 1e-6]
    if len(points) < 2:
        return None
    if points[-1][0] - points[0][0] <= 1e-3:
        return None
    return normalized_vector((points[-1][1] - points[0][1], points[-1][2] - points[0][2]))


def heading_reversal(track: TrackState, window: float, timestamp: float,
                     min_angle: float = 2.0) -> tuple[float, float, float] | None:
    """Detect a U-turn from trajectory history.

    Returns ``(angle, duration, peak_speed)`` where ``angle`` is the turn in
    radians, or ``None`` when the track merely curved through the junction. A
    curve has a large net displacement per second; a U-turn returns towards its
    origin, so it is separated by both the turn angle and the net-displacement
    rate.
    """
    points = [p for p in track.bottom_history
              if p[0] > timestamp - max(0.0, window) and p[0] <= timestamp + 1e-6]
    if len(points) < 6:
        return None
    first, last = points[0], points[-1]
    span = last[0] - first[0]
    if span <= 0.2:
        return None
    net = math.hypot(last[1] - first[1], last[2] - first[2])
    if net < 1e-6 or net / span > 10.0:
        return None
    mid = points[len(points) // 2]
    outbound = normalized_vector((mid[1] - first[1], mid[2] - first[2]))
    inbound = normalized_vector((last[1] - mid[1], last[2] - mid[2]))
    if outbound is None or inbound is None:
        return None
    angle = angle_between(outbound, inbound)
    if angle < min_angle:
        return None
    return (angle, span, net / span)


# ---------------------------------------------------------------------------
# interaction
# ---------------------------------------------------------------------------

def time_to_collision(a: TrackState, b: TrackState) -> float | None:
    """Constant-velocity TTC between two track centres, in seconds.

    Uses the separating axis, so a pair that is diverging, overtaking or merely
    passing in another lane never produces a collision time.
    """
    rx = b.center[0] - a.center[0]
    ry = b.center[1] - a.center[1]
    vx = b.velocity[0] - a.velocity[0]
    vy = b.velocity[1] - a.velocity[1]
    vv = vx * vx + vy * vy
    if vv <= 1e-6:
        return None
    dot = rx * vx + ry * vy
    if dot >= 0:
        return None
    return -dot / vv


def closing_speed(a: TrackState, b: TrackState) -> float:
    """Rate at which the gap between two tracks is shrinking, px/s."""
    rx = b.center[0] - a.center[0]
    ry = b.center[1] - a.center[1]
    distance = math.hypot(rx, ry)
    if distance <= 1e-6:
        return float("inf")
    vx = b.velocity[0] - a.velocity[0]
    vy = b.velocity[1] - a.velocity[1]
    return -((rx * vx + ry * vy) / distance)


def deceleration(track: TrackState) -> float:
    """Component of acceleration opposing the current velocity, px/s^2."""
    speed = track.speed
    if speed <= 1e-3:
        return 0.0
    return -vector_dot(track.acceleration, track.velocity) / speed


def path_length(points: Sequence[Point3]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][1] - points[i - 1][1], points[i][2] - points[i - 1][2])
    return total


def swept_gap(a: TrackState, b: TrackState) -> float:
    """Smallest centre distance, corrected for box size, in px.

    Two vehicles in a queue are bumper to bumper, so raw centre distance is a
    poor proxy for how close they are. Subtracting the half-extents along the
    line of centres gives the actual gap.
    """
    rx = b.center[0] - a.center[0]
    ry = b.center[1] - a.center[1]
    distance = math.hypot(rx, ry)
    if distance <= 1e-6:
        return 0.0
    ux, uy = rx / distance, ry / distance
    reach_a = abs(ux) * a.width / 2.0 + abs(uy) * a.height / 2.0
    reach_b = abs(ux) * b.width / 2.0 + abs(uy) * b.height / 2.0
    return max(0.0, distance - reach_a - reach_b)


def road_frame_gap(
    a: TrackState, b: TrackState, axis: Point
) -> tuple[float, float]:
    """Separation of two vehicles resolved along and across the road.

    Returns ``(along, across)`` in pixels. This exists because of a specific
    failure of image-space geometry at this camera: it looks obliquely down at
    the carriageway, so the box of a lead vehicle in a queue *covers* the
    vehicle behind it. Two cars a car length apart on the road can therefore
    overlap almost completely in the image, which is exactly the evidence a
    naive collision detector reads as contact.

    Resolving the separation in the road frame separates the two cases: a queue
    is separated ALONG the road by a car length while a collision is not
    separated in either direction.
    """
    unit = normalized_vector(axis)
    if unit is None:
        distance = math.hypot(b.center[0] - a.center[0], b.center[1] - a.center[1])
        return (distance, 0.0)
    dx = b.bottom_center[0] - a.bottom_center[0]
    dy = b.bottom_center[1] - a.bottom_center[1]
    along = abs(dx * unit[0] + dy * unit[1])
    across = abs(dx * (-unit[1]) + dy * unit[0])
    return (along, across)
