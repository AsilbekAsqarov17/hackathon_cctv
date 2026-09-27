from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .geometry import (
    Point,
    point_in_polygon,
    rescale_points,
    scene_scale,
)


@dataclass
class Lane:
    lane_id: int
    polygon: list[list[float]]
    direction: tuple[float, float] = (0.0, 1.0)
    allowed_moves: tuple[str, ...] = ("straight",)
    approach_point: Point | None = None
    name: str = ""


@dataclass
class SceneLine:
    line_id: str
    segment: tuple[Point, Point]
    kind: str = "stop"
    direction: tuple[float, float] | None = None


@dataclass
class TrafficLightROI:
    light_id: str
    roi: tuple[float, float, float, float]
    direction: tuple[float, float] | None = None


@dataclass
class SceneConfig:
    scene_id: str
    width: int
    height: int
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[SceneLine] = field(default_factory=list)
    solid_lines: list[SceneLine] = field(default_factory=list)
    crossings: list[list[list[float]]] = field(default_factory=list)
    road_polygon: list[list[float]] = field(default_factory=list)
    road_polygons: list[list[list[float]]] = field(default_factory=list)
    traffic_lights: list[TrafficLightROI] = field(default_factory=list)
    homography: list[list[float]] | None = None
    normalized: bool = False
    auto_road_fallback: bool = True

    @property
    def has_geometry(self) -> bool:
        return bool(
            self.lanes
            or self.stop_lines
            or self.solid_lines
            or self.crossings
            or self.road_polygon
            or self.road_polygons
        )

    @property
    def has_road(self) -> bool:
        return bool(self.road_polygon or self.road_polygons or self.lanes)


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def _scale_bbox(
    values: Sequence[float],
    width: int,
    height: int,
    normalized: bool,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in values]
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    else:
        sx, sy = scale
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _parse_line(
    item: dict[str, Any],
    index: int,
    kind: str,
    width: int,
    height: int,
    normalized: bool,
    scale: tuple[float, float] = (1.0, 1.0),
) -> SceneLine:
    values = item.get("segment", item.get("points", []))
    if len(values) < 2:
        raise ValueError(f"{kind} line {index} needs two points")
    points = rescale_points(values[:2], width, height, normalized, scale)
    direction_raw = item.get("direction")
    # A direction is a vector in image space, so it takes the same per-axis
    # factor as a point; callers normalise it, so only the ratio matters.
    direction = (
        (direction_raw[0] * scale[0], direction_raw[1] * scale[1])
        if direction_raw
        else None
    )
    return SceneLine(str(item.get("id", f"{kind}_{index}")), (_point(points[0]), _point(points[1])), kind, direction)


def load_scene_config(
    path: str | Path,
    width: int,
    height: int,
) -> SceneConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    normalized = bool(data.get("normalized", False))
    # One authored scene serves every resolution of the same camera: coordinates
    # are declared in the authoring resolution and mapped onto the actual video.
    scale = scene_scale(data.get("reference_size"), width, height)
    lanes: list[Lane] = []
    for index, item in enumerate(data.get("lanes", [])):
        polygon = rescale_points(item.get("polygon", []), width, height, normalized, scale)
        if len(polygon) < 3:
            continue
        raw_direction = item.get("direction", [0.0, 1.0])
        direction = (float(raw_direction[0]) * scale[0], float(raw_direction[1]) * scale[1])
        lanes.append(
            Lane(
                lane_id=int(item.get("id", index)),
                polygon=polygon,
                direction=direction,
                allowed_moves=tuple(str(x) for x in item.get("allowed_moves", ["straight"])),
                approach_point=(
                    _point(rescale_points([item["approach_point"]], width, height, normalized, scale)[0])
                    if item.get("approach_point") is not None
                    else None
                ),
                name=str(item.get("name", "")),
            )
        )
    stop_lines = [
        _parse_line(item, i, "stop", width, height, normalized, scale)
        for i, item in enumerate(data.get("stop_lines", []))
    ]
    solid_lines = [
        _parse_line(item, i, "solid", width, height, normalized, scale)
        for i, item in enumerate(data.get("solid_lines", data.get("lane_markings", [])))
    ]
    crossings = [
        rescale_points(item.get("polygon", item), width, height, normalized, scale)
        for item in data.get("crosswalks", data.get("crossings", []))
    ]
    crossings = [item for item in crossings if len(item) >= 3]
    road = rescale_points(data.get("road_polygon", []), width, height, normalized, scale)
    road_polygons = [
        rescale_points(item, width, height, normalized, scale)
        for item in data.get("road_polygons", [])
    ]
    road_polygons = [item for item in road_polygons if len(item) >= 3]
    lights: list[TrafficLightROI] = []
    for index, item in enumerate(data.get("traffic_lights", [])):
        roi_raw = item.get("roi", item.get("bbox"))
        if not roi_raw or len(roi_raw) < 4:
            continue
        roi = _scale_bbox(roi_raw, width, height, normalized, scale)
        direction_raw = item.get("direction")
        lights.append(
            TrafficLightROI(
                light_id=str(item.get("id", f"light_{index}")),
                roi=roi,
                direction=(
                    (float(direction_raw[0]) * scale[0], float(direction_raw[1]) * scale[1])
                    if direction_raw
                    else None
                ),
            )
        )
    return SceneConfig(
        scene_id=str(data.get("scene_id", path.stem)),
        width=width,
        height=height,
        lanes=lanes,
        stop_lines=stop_lines,
        solid_lines=solid_lines,
        crossings=crossings,
        road_polygon=road,
        road_polygons=road_polygons,
        traffic_lights=lights,
        homography=data.get("homography"),
        normalized=normalized,
        auto_road_fallback=bool(data.get("auto_road_fallback", True)),
    )


class SceneContext:
    """Runtime geometry and traffic-light state for one video."""

    def __init__(self, config: SceneConfig | None, width: int = 0, height: int = 0):
        self.config = config
        self.width = config.width if config else int(width)
        self.height = config.height if config else int(height)
        self.lights: dict[str, Any] = {}
        self.dynamic_lights: dict[str, Any] = {}

    @property
    def scene_id(self) -> str:
        return self.config.scene_id if self.config else "unconfigured"

    @property
    def has_geometry(self) -> bool:
        return bool(self.config and self.config.has_geometry)

    @property
    def has_road(self) -> bool:
        # A conservative image-space road fallback keeps trajectory rules
        # useful when a camera has not been calibrated yet. It is only a
        # fallback; explicit road polygons always take precedence.
        return bool(
            (self.config and self.config.has_road)
            or (
                self.width > 0
                and self.height > 0
                and (not self.config or self.config.auto_road_fallback)
            )
        )

    def lane_for_point(self, point: Point) -> Lane | None:
        if not self.config:
            return None
        for lane in self.config.lanes:
            if point_in_polygon(point, lane.polygon):
                return lane
        return None

    def crossing_for_point(self, point: Point) -> list[list[float]] | None:
        if not self.config:
            return None
        for crossing in self.config.crossings:
            if point_in_polygon(point, crossing):
                return crossing
        return None

    @staticmethod
    def side_of_line(point: Point, line: Any) -> float:
        a, b = line.segment
        value = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        return 1.0 if value > 1e-9 else -1.0 if value < -1e-9 else 0.0

    def to_world(self, point: Point) -> Point | None:
        if not self.config or not self.config.homography:
            return None
        import cv2
        import numpy as np

        matrix = np.asarray(self.config.homography, dtype=np.float64)
        if matrix.shape != (3, 3):
            return None
        projected = cv2.perspectiveTransform(np.asarray([[point]], dtype=np.float64), matrix).reshape(2)
        return float(projected[0]), float(projected[1])

    def is_road_point(self, point: Point, carriageway_only: bool = False) -> bool:
        """Is this point on drivable road?

        ``carriageway_only`` restricts the answer to the lane polygons, which
        are the directional bands fitted to measured traffic. That distinction
        matters for pedestrians: the road polygons at this camera deliberately
        include the verges and footways that border the junction, and treating
        those as carriageway made 53 percent of all pedestrian samples count as
        "on the road", which made ``jaywalking`` fire constantly on people
        standing on a pavement.
        """
        if not self.config or not self.config.has_road:
            if self.config is not None and not self.config.auto_road_fallback:
                return False
            # Without calibration, use the lower central road-like region.
            # This is intentionally conservative and can be replaced by an
            # explicit road polygon in the scene file.
            return self.height > 0 and self.height * 0.45 <= point[1] <= self.height * 0.98
        if carriageway_only:
            return any(point_in_polygon(point, lane.polygon) for lane in self.config.lanes)
        if self.config.road_polygon and point_in_polygon(point, self.config.road_polygon):
            return True
        if any(point_in_polygon(point, polygon) for polygon in self.config.road_polygons):
            return True
        return any(point_in_polygon(point, lane.polygon) for lane in self.config.lanes)

    def line_for_track(self, track: Any) -> SceneLine | None:
        if not self.config:
            return None
        point = track.bottom_center
        best: tuple[float, SceneLine] | None = None
        for line in self.config.stop_lines:
            distance = min(
                _distance_to_segment(point, line.segment[0], line.segment[1]),
                _distance_to_segment(track.center, line.segment[0], line.segment[1]),
            )
            if best is None or distance < best[0]:
                best = (distance, line)
        return best[1] if best else None

    def nearest_red_light(self, track: Any) -> str | None:
        """Return the closest currently-red configured or detected light."""
        candidates: list[tuple[float, str]] = []
        if self.config:
            for light in self.config.traffic_lights:
                state = self.lights.get(light.light_id)
                if state is not None and state.color == "red":
                    candidates.append((_bbox_distance(track.bbox, light.roi), light.light_id))
        for light_id, state in self.dynamic_lights.items():
            if state.color == "red":
                candidates.append((_bbox_distance(track.bbox, state.bbox), light_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def signal_for_line(self, line: Any) -> str | None:
        """The signal head that governs a given stop line.

        A head governs a line when it faces the same direction the line permits.
        That is the physically meaningful test, and unlike an image-space
        "is it past the line" test it is not fooled by pole height: a signal head
        is mounted several metres up, so its ROI projects well above the road
        point it governs and can appear upstream in the image.

        Requiring the declared approach directions to agree is also what stops
        an unrelated head elsewhere in the frame from deciding whether a
        violation occurred.
        """
        if not self.config or line.direction is None:
            return None
        want = _unit(line.direction)
        if want is None:
            return None
        best: tuple[float, str] | None = None
        for light in self.config.traffic_lights:
            have = _unit(light.direction) if light.direction is not None else None
            if have is None:
                continue
            agreement = have[0] * want[0] + have[1] * want[1]
            if agreement < 0.5:
                continue
            # Among heads facing the same way, prefer the one nearest the line.
            distance = _distance_to_segment(
                ((light.roi[0] + light.roi[2]) / 2.0, (light.roi[1] + light.roi[3]) / 2.0),
                line.segment[0], line.segment[1],
            )
            if best is None or distance < best[0]:
                best = (distance, light.light_id)
        return best[1] if best else None

    def signal_color_for_line(self, line: Any) -> str:
        """Current colour of the signal governing ``line``, or ``unknown``.

        Returns ``unknown`` rather than guessing when the head is not readable,
        so a rule that needs a definite state stays silent instead of inventing
        one.
        """
        light_id = self.signal_for_line(line)
        if light_id is None:
            return "unknown"
        state = self.lights.get(light_id)
        if state is None:
            return "unknown"
        return str(getattr(state, "color", "unknown"))


def _unit(vector: Sequence[float]) -> tuple[float, float] | None:
    x, y = float(vector[0]), float(vector[1])
    length = (x * x + y * y) ** 0.5
    if length < 1e-9:
        return None
    return (x / length, y / length)


def _distance_to_segment(point: Point, a: Point, b: Point) -> float:
    # Local import avoids a circular import in the scene package.
    from .geometry import point_segment_distance

    return point_segment_distance(point, a, b)


def _bbox_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def choose_scene_path(video_path: str, configured: str = "", default_path: str = "") -> Path | None:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    env_path = os.getenv("TRAFFIC_SCENE_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    video_stem = Path(video_path).stem
    candidates.extend(
        [
            Path("configs/scenes") / f"{video_stem}.json",
            Path("configs/scenes/default.json"),
            Path(default_path) if default_path else Path("configs/scenes/default.json"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
