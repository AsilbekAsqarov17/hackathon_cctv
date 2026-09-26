from __future__ import annotations

import math
from collections import deque
from typing import Iterable

from ..contracts import TrackObservation, TrackState
from ..scene.config import SceneContext


class TrackManager:
    """Owns the one track representation used by all Part A rules."""

    def __init__(self, max_age_seconds: float = 2.0, history_seconds: float = 8.0):
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.history_seconds = max(1.0, float(history_seconds))
        self._states: dict[int, TrackState] = {}

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def update(
        self,
        observations: Iterable[TrackObservation],
        timestamp: float,
        frame_id: int,
        scene: SceneContext | None = None,
    ) -> list[TrackState]:
        seen: set[int] = set()
        for observation in observations:
            track_id = int(observation.track_id)
            seen.add(track_id)
            center = self._center(observation.bbox)
            state = self._states.get(track_id)
            previous_world = state.center_world if state is not None else None
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    class_id=observation.class_id,
                    class_name=observation.class_name,
                    bbox=observation.bbox,
                    score=observation.score,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    center=center,
                    center_world=scene.to_world(center) if scene is not None else None,
                    history=deque(maxlen=300),
                    last_seen_frame=frame_id,
                )
                state.history.append((timestamp, center[0], center[1]))
                state.bottom_history.append((timestamp, state.bottom_center[0], state.bottom_center[1]))
                self._states[track_id] = state
            else:
                previous_center = state.center
                previous_velocity = state.velocity
                dt = max(1e-3, timestamp - state.last_seen)
                raw_velocity = (
                    (center[0] - previous_center[0]) / dt,
                    (center[1] - previous_center[1]) / dt,
                )
                # Exponential smoothing reduces detector jitter while staying
                # responsive enough for collision and braking signals.
                alpha = 0.65
                velocity = (
                    alpha * raw_velocity[0] + (1.0 - alpha) * previous_velocity[0],
                    alpha * raw_velocity[1] + (1.0 - alpha) * previous_velocity[1],
                )
                acceleration = (
                    (velocity[0] - previous_velocity[0]) / dt,
                    (velocity[1] - previous_velocity[1]) / dt,
                )
                state.acceleration = (
                    alpha * acceleration[0] + (1.0 - alpha) * state.acceleration[0],
                    alpha * acceleration[1] + (1.0 - alpha) * state.acceleration[1],
                )
                current_world = scene.to_world(center) if scene is not None else None
                if current_world is not None and previous_world is not None:
                    raw_world_velocity = (
                        (current_world[0] - previous_world[0]) / dt,
                        (current_world[1] - previous_world[1]) / dt,
                    )
                    state.velocity_world = (
                        alpha * raw_world_velocity[0] + (1.0 - alpha) * (state.velocity_world[0] if state.velocity_world else 0.0),
                        alpha * raw_world_velocity[1] + (1.0 - alpha) * (state.velocity_world[1] if state.velocity_world else 0.0),
                    )
                state.bbox = observation.bbox
                state.score = observation.score
                state.class_id = observation.class_id
                state.class_name = observation.class_name
                state.center = center
                state.center_world = current_world
                state.velocity = velocity
                state.last_seen = timestamp
                state.last_seen_frame = frame_id
                state.missed_frames = 0
                state.history.append((timestamp, center[0], center[1]))
                state.bottom_history.append((timestamp, state.bottom_center[0], state.bottom_center[1]))
            if scene is not None:
                self._assign_lane(state, scene)
        for track_id, state in self._states.items():
            if track_id not in seen:
                state.missed_frames += 1
        self._prune(timestamp)
        return self.states()

    def _assign_lane(self, state: TrackState, scene: SceneContext) -> None:
        lane = scene.lane_for_point(state.center)
        lane_id = lane.lane_id if lane is not None else None
        if lane_id != state.lane_id:
            if lane_id is not None:
                state.lane_history.append(lane_id)
                state.lane_history = state.lane_history[-20:]
        state.lane_id = lane_id

    def _prune(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, state in self._states.items()
            if timestamp - state.last_seen > self.max_age_seconds
        ]
        for track_id in stale:
            del self._states[track_id]

    def states(self) -> list[TrackState]:
        return sorted(self._states.values(), key=lambda state: state.track_id)

    def get(self, track_id: int) -> TrackState | None:
        return self._states.get(track_id)

    @staticmethod
    def is_vehicle(state: TrackState) -> bool:
        name = state.class_name.lower()
        return name in {
            "car",
            "vehicle",
            "truck",
            "bus",
            "motorcycle",
            "motorbike",
            "bicycle",
            "bike",
        } or state.class_id in {1, 2, 3, 5, 7}

    @staticmethod
    def is_person(state: TrackState) -> bool:
        name = state.class_name.lower()
        return name in {"person", "pedestrian", "human"} or state.class_id == 0

    @staticmethod
    def is_road_user(state: TrackState) -> bool:
        name = state.class_name.lower().replace("-", "_")
        special = {"obstacle", "debris", "animal", "cone", "barrier", "fallen", "fire", "smoke", "flame"}
        return (
            TrackManager.is_vehicle(state)
            or TrackManager.is_person(state)
            or any(word in name for word in special)
        )

    @staticmethod
    def center_distance(a: TrackState, b: TrackState) -> float:
        return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])

    @staticmethod
    def relative_speed(a: TrackState, b: TrackState) -> float:
        dx = a.velocity[0] - b.velocity[0]
        dy = a.velocity[1] - b.velocity[1]
        return math.hypot(dx, dy)

    @staticmethod
    def time_to_collision(a: TrackState, b: TrackState) -> float | None:
        """Constant-velocity TTC for the center positions, in seconds.

        Superseded by :meth:`predicted_collision_time` for anything that
        matters. Centre convergence is not a collision: in a busy intersection
        some pair of road users is always converging, so the minimum over all
        pairs is near zero almost always, which is useless as a risk signal.
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

    @staticmethod
    def predicted_collision_time(
        a: TrackState, b: TrackState, horizon: float = 6.0, min_closing_mps: float = 1.0
    ) -> float | None:
        """Earliest time in ``horizon`` seconds at which two boxes would overlap.

        This is the quantity a risk score actually wants, and it differs from
        centre convergence in three ways that each mattered here:

        * it predicts **box** overlap, so a pedestrian walking past a stopped
          car a metre away is not a collision;
        * it requires meaningful **relative speed**, so two objects drifting
          together at walking pace are not a conflict;
        * it returns ``None`` when no overlap is predicted at all, instead of a
          tiny positive number that then dominates every ``max`` downstream.

        Boxes are treated as axis-aligned and grown by half the relative motion,
        which is the standard approximation and is cheap.
        """
        scale_a = a.metres_per_pixel
        scale_b = b.metres_per_pixel
        scales = [s for s in (scale_a, scale_b) if s]
        if not scales:
            return None
        scale = max(scales)

        rvx = (b.velocity[0] - a.velocity[0]) * scale
        rvy = (b.velocity[1] - a.velocity[1]) * scale
        if math.hypot(rvx, rvy) < min_closing_mps:
            return None

        # Half-extents in metres, averaged over the two scale estimates.
        half_w = 0.25 * (a.width * (scale_a or scale) + b.width * (scale_b or scale))
        half_h = 0.25 * (a.height * (scale_a or scale) + b.height * (scale_b or scale))
        dx = (b.center[0] - a.center[0]) * scale
        dy = (b.center[1] - a.center[1]) * scale

        # Each axis overlaps when |offset| <= the summed half extents. Both axes
        # must overlap, so the collision time is the later of the two; an axis
        # that never closes rules the pair out entirely.
        latest = 0.0
        for offset, closing, extent in ((dx, rvx, half_w), (dy, rvy, half_h)):
            span = abs(offset) - extent
            if span <= 0:
                continue  # already overlapping on this axis
            if closing >= -1e-9:
                return None  # this axis never closes
            t = span / (-closing)
            if t > horizon:
                return None
            latest = max(latest, t)
        return latest

    def snapshot(self) -> list[dict]:
        return [
            {
                "track_id": state.track_id,
                "class_name": state.class_name,
                "bbox": state.bbox,
                "center": state.center,
                "velocity": state.velocity,
                "speed": state.speed,
                "center_world": state.center_world,
                "velocity_world": state.velocity_world,
                "lane_id": state.lane_id,
            }
            for state in self.states()
        ]
