from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point_in_polygon(point: Point, polygon: Sequence[Sequence[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def polygon_center(polygon: Sequence[Sequence[float]]) -> Point:
    if not polygon:
        return (0.0, 0.0)
    return (
        sum(float(p[0]) for p in polygon) / len(polygon),
        sum(float(p[1]) for p in polygon) / len(polygon),
    )


def bbox_center(bbox: BBox) -> Point:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_bottom_center(bbox: BBox) -> Point:
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def bbox_distance(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return math.hypot(dx, dy)


def point_segment_distance(point: Point, a: Point, b: Point) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = clamp(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return True
    return (
        abs(o1) <= 1e-9
        and point_segment_distance(c, a, b) <= 1e-9
        or abs(o2) <= 1e-9
        and point_segment_distance(d, a, b) <= 1e-9
        or abs(o3) <= 1e-9
        and point_segment_distance(a, c, d) <= 1e-9
        or abs(o4) <= 1e-9
        and point_segment_distance(b, c, d) <= 1e-9
    )


def segment_crossed_by_track(
    previous: Point, current: Point, line: Sequence[Sequence[float]]
) -> bool:
    if len(line) < 2:
        return False
    return segments_intersect(
        previous,
        current,
        (float(line[0][0]), float(line[0][1])),
        (float(line[1][0]), float(line[1][1])),
    )


def point_in_bbox(point: Point, bbox: BBox, margin: float = 0.0) -> bool:
    x, y = point
    return (
        bbox[0] - margin <= x <= bbox[2] + margin
        and bbox[1] - margin <= y <= bbox[3] + margin
    )


def normalized_vector(vector: Point) -> Point | None:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        return None
    return vector[0] / length, vector[1] / length


def vector_dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def angle_between(a: Point, b: Point) -> float:
    aa, bb = normalized_vector(a), normalized_vector(b)
    if aa is None or bb is None:
        return 0.0
    return math.acos(clamp(vector_dot(aa, bb), -1.0, 1.0))


def rescale_points(
    points: Iterable[Sequence[float]], width: int, height: int, normalized: bool
) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if normalized:
            x *= width
            y *= height
        result.append([x, y])
    return result


def compute_homography(
    image_points: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Compute a plane homography from four or more corresponding points."""
    if len(image_points) < 4 or len(world_points) < 4:
        raise ValueError("at least four image/world point pairs are required")
    src = np.asarray(image_points, dtype=np.float32)
    dst = np.asarray(world_points, dtype=np.float32)
    matrix, _ = cv2.findHomography(src, dst)
    if matrix is None:
        raise ValueError("homography could not be computed")
    return (matrix / matrix[2, 2]).tolist()
