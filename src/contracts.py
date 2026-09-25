from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class Detection:
    """One detector output in pixel coordinates."""

    bbox: BBox
    score: float
    class_id: int
    class_name: str


@dataclass
class TrackObservation:
    """One tracker output in pixel coordinates."""

    track_id: int
    bbox: BBox
    score: float
    class_id: int
    class_name: str


@dataclass
class TrackState:
    """Canonical temporal state shared by every Part A rule."""

    track_id: int
    class_id: int
    class_name: str
    bbox: BBox
    score: float
    first_seen: float
    last_seen: float
    center: Point
    center_world: Point | None = None
    velocity: Point = (0.0, 0.0)
    velocity_world: Point | None = None
    acceleration: Point = (0.0, 0.0)
    history: Deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=300)
    )
    bottom_history: Deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=300)
    )
    lane_id: Optional[int] = None
    lane_history: list[int] = field(default_factory=list)
    stopped_since: Optional[float] = None
    last_seen_frame: int = -1
    missed_frames: int = 0

    @property
    def speed(self) -> float:
        return float((self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5)

    @property
    def acceleration_magnitude(self) -> float:
        return float(
            (self.acceleration[0] ** 2 + self.acceleration[1] ** 2) ** 0.5
        )

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def bottom_center(self) -> Point:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])

    def recent_points(self, seconds: float, timestamp: float) -> list[tuple[float, float, float]]:
        cutoff = timestamp - max(0.0, seconds)
        return [p for p in self.history if p[0] >= cutoff]

    def recent_bottom_points(self, seconds: float, timestamp: float) -> list[tuple[float, float, float]]:
        cutoff = timestamp - max(0.0, seconds)
        return [p for p in self.bottom_history if p[0] >= cutoff]

    def heading(self, seconds: float = 1.0, timestamp: float | None = None) -> Point | None:
        """Return a recent image-space displacement vector."""
        if timestamp is None:
            timestamp = self.last_seen
        points = self.recent_points(seconds, timestamp)
        if len(points) < 2:
            return None
        first = points[0]
        last = points[-1]
        dt = last[0] - first[0]
        if dt <= 1e-3:
            return None
        return ((last[1] - first[1]) / dt, (last[2] - first[2]) / dt)

    def heading_world(self) -> Point | None:
        return self.velocity_world


@dataclass
class TrafficLightState:
    light_id: str
    color: str
    confidence: float
    bbox: BBox


@dataclass
class SceneState:
    scene_id: str
    width: int
    height: int
    lights: dict[str, TrafficLightState] = field(default_factory=dict)
    context: Any = None


@dataclass
class FrameState:
    frame_id: int
    timestamp: float
    tracks: list[TrackState]
    scene: SceneState


@dataclass
class RuleSignal:
    label: str
    active: bool
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    start_hint: float | None = None


@dataclass
class EventSegment:
    start: float
    end: float
    label: str
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_list(self) -> list[float | str]:
        return [round(float(self.start), 3), round(float(self.end), 3), self.label]
