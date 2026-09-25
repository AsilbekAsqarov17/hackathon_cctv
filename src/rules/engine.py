from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

from ..contracts import FrameState, RuleSignal, TrackState
from ..scene.geometry import (
    angle_between,
    bbox_iou,
    normalized_vector,
    point_segment_distance,
    segment_crossed_by_track,
    vector_dot,
)
from ..tracking.track_manager import TrackManager

OFFICIAL_LABELS = (
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
)


class RuleEngine:
    """Evaluates frame-level rules from a canonical :class:`FrameState`.

    Rules deliberately do not perform inference. They only consume track and
    scene state, which keeps imported repository code and our competition logic
    independently testable.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._condition_start: dict[tuple[Any, ...], float] = {}
        self._wrong_way_counts: dict[int, int] = {}
        self._red_fired: dict[tuple[int, str], float] = {}
        self._last_evidence: dict[str, dict[str, Any]] = {}

    def _rule_config(self, label: str) -> dict[str, Any]:
        value = self.config.get(label, {})
        return value if isinstance(value, dict) else {}

    def _enabled(self, label: str) -> bool:
        return bool(self._rule_config(label).get("enabled", True))

    @staticmethod
    def _inactive(label: str) -> RuleSignal:
        return RuleSignal(label=label, active=False, confidence=0.0)

    def _condition_since(self, key: tuple[Any, ...], active: bool, timestamp: float) -> float | None:
        if not active:
            self._condition_start.pop(key, None)
            return None
        return self._condition_start.setdefault(key, timestamp)

    def _retain_conditions(self, label: str, active_keys: set[tuple[Any, ...]]) -> None:
        """Keep ongoing timers while dropping conditions that ended."""
        self._condition_start = {
            key: value
            for key, value in self._condition_start.items()
            if key[0] != label or key in active_keys
        }

    def _scene(self, state: FrameState) -> Any:
        return state.scene.context

    @staticmethod
    def _road_track(track: TrackState, scene: Any) -> bool:
        return bool(scene and getattr(scene, "has_road", False) and scene.is_road_point(track.center))

    @staticmethod
    def _special(track: TrackState, words: Iterable[str]) -> bool:
        name = track.class_name.lower().replace("-", "_")
        return any(word in name for word in words)

    @staticmethod
    def _line_cross_time(track: TrackState, line: Any, timestamp: float, window: float = 2.0) -> float | None:
        points = track.recent_bottom_points(window, timestamp)
        if not points:
            points = track.recent_points(window, timestamp)
        if len(points) < 2:
            return None
        for i in range(len(points) - 1, 0, -1):
            t1, x1, y1 = points[i - 1]
            t2, x2, y2 = points[i]
            if segment_crossed_by_track((x1, y1), (x2, y2), line.segment):
                return t1
        return None

    @staticmethod
    def _near_line(track: TrackState, line: Any, distance: float) -> bool:
        return min(
            point_segment_distance(track.bottom_center, line.segment[0], line.segment[1]),
            point_segment_distance(track.center, line.segment[0], line.segment[1]),
        ) <= distance

    @staticmethod
    def _heading_change(track: TrackState, timestamp: float, seconds: float = 4.0) -> float:
        points = track.recent_points(seconds, timestamp)
        if len(points) < 3:
            return 0.0
        first = points[0]
        middle = points[len(points) // 2]
        last = points[-1]
        v1 = (middle[1] - first[1], middle[2] - first[2])
        v2 = (last[1] - middle[1], last[2] - middle[2])
        if normalized_vector(v1) is None or normalized_vector(v2) is None:
            return 0.0
        return angle_between(v1, v2)

    @staticmethod
    def _deceleration(track: TrackState) -> float:
        speed = track.speed
        if speed <= 1e-3:
            return 0.0
        return -vector_dot(track.acceleration, track.velocity) / speed

    def _wrong_way(self, state: FrameState) -> RuleSignal:
        label = "wrong_way"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False):
            return self._inactive(label)
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or track.speed < float(cfg.get("min_speed", 4.0)):
                self._wrong_way_counts.pop(track.track_id, None)
                continue
            lane = scene.lane_for_point(track.center)
            if lane is None:
                self._wrong_way_counts.pop(track.track_id, None)
                continue
            direction = normalized_vector(lane.direction)
            # Lane direction vectors in the scene file are image-space by
            # default; do not mix them with optional world-space velocity.
            velocity = normalized_vector(track.velocity)
            if direction is None or velocity is None:
                self._wrong_way_counts.pop(track.track_id, None)
                continue
            dot = vector_dot(direction, velocity)
            key = (label, track.track_id)
            if dot < float(cfg.get("direction_dot", -0.45)):
                self._wrong_way_counts[track.track_id] = self._wrong_way_counts.get(track.track_id, 0) + 1
                active_keys.add(key)
                if self._wrong_way_counts[track.track_id] < int(cfg.get("debounce_frames", 3)):
                    continue
                start = self._condition_since(key, True, state.timestamp)
                return RuleSignal(
                    label,
                    True,
                    confidence=min(1.0, abs(dot)),
                    evidence={"track_id": track.track_id, "lane_id": lane.lane_id, "dot": dot},
                    start_hint=start,
                )
            self._wrong_way_counts.pop(track.track_id, None)
            self._condition_start.pop(key, None)
        self._retain_conditions(label, active_keys)
        self._wrong_way_counts = {
            track_id: count for track_id, count in self._wrong_way_counts.items() if track_id in {k[1] for k in active_keys}
        }
        return self._inactive(label)

    def _queued_at_signal(self, track: TrackState, state: FrameState) -> bool:
        scene = self._scene(state)
        line = scene.line_for_track(track)
        if line is None or not self._near_line(track, line, 120.0):
            return False
        for other in state.tracks:
            if other.track_id == track.track_id or not TrackManager.is_vehicle(other):
                continue
            if self._near_line(other, line, 140.0) and other.speed <= 5.0:
                return True
        return False

    def _stopped_vehicle(self, state: FrameState) -> RuleSignal:
        label = "stopped_vehicle"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return self._inactive(label)
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or not self._road_track(track, scene):
                continue
            stopped = track.speed <= float(cfg.get("speed", 3.0))
            key = (label, track.track_id)
            if not stopped or self._queued_at_signal(track, state):
                self._condition_start.pop(key, None)
                continue
            active_keys.add(key)
            since = self._condition_since(key, True, state.timestamp)
            if since is not None and state.timestamp - since >= float(cfg.get("duration", 10.0)):
                self._retain_conditions(label, active_keys)
                return RuleSignal(
                    label,
                    True,
                    confidence=0.9,
                    evidence={"track_id": track.track_id, "stopped_since": since},
                    start_hint=since,
                )
        self._retain_conditions(label, active_keys)
        return self._inactive(label)

    def _congestion(self, state: FrameState) -> RuleSignal:
        label = "congestion"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return self._inactive(label)
        groups: dict[Any, list[TrackState]] = {}
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or not self._road_track(track, scene):
                continue
            if track.lane_id is not None:
                key: Any = ("lane", track.lane_id)
            else:
                # A direction split is a useful fallback when lanes are not
                # annotated, but it is still gated by road geometry.
                key = ("direction", 1 if track.velocity[1] >= 0 else -1)
            groups.setdefault(key, []).append(track)
        active_keys: set[tuple[Any, ...]] = set()
        for key, tracks in groups.items():
            slow = [track for track in tracks if track.speed <= float(cfg.get("speed", 8.0))]
            if len(tracks) < int(cfg.get("min_vehicles", 4)) or len(slow) < int(cfg.get("min_vehicles", 4)):
                continue
            condition_key = (label, key)
            active_keys.add(condition_key)
            since = self._condition_since(condition_key, True, state.timestamp)
            if since is not None and state.timestamp - since >= float(cfg.get("duration", 5.0)):
                self._retain_conditions(label, active_keys)
                return RuleSignal(
                    label,
                    True,
                    confidence=min(1.0, len(slow) / max(1, int(cfg.get("min_vehicles", 4))) * 0.7),
                    evidence={"group": repr(key), "vehicles": len(tracks), "slow": len(slow)},
                    start_hint=since,
                )
        self._retain_conditions(label, active_keys)
        return self._inactive(label)

    def _line_rule(self, state: FrameState, label: str, require_red: bool = False) -> RuleSignal:
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False):
            return self._inactive(label)
        lines = scene.config.solid_lines if label == "solid_line_crossing" else scene.config.stop_lines
        for track in state.tracks:
            if not TrackManager.is_vehicle(track):
                continue
            for line in lines:
                cross_time = self._line_cross_time(track, line, state.timestamp)
                if require_red and scene.nearest_red_light(track) is None:
                    continue
                if cross_time is None and not (
                    label == "stop_line" and self._near_line(track, line, 35.0) and track.speed <= 8.0
                ):
                    continue
                key = (label, track.track_id, line.line_id)
                active = cross_time is not None or self._near_line(track, line, 35.0)
                start = self._condition_since(key, active, state.timestamp)
                if start is None:
                    continue
                # Hold the line event briefly so a one-frame crossing is not
                # lost between sampled detector frames.
                if cross_time is not None and state.timestamp - cross_time > float(cfg.get("hold", 0.8)):
                    continue
                if label == "stop_line" and not require_red and track.speed > 8.0 and cross_time is None:
                    continue
                return RuleSignal(
                    label,
                    True,
                    confidence=0.85,
                    evidence={"track_id": track.track_id, "line_id": line.line_id, "cross_time": cross_time},
                    start_hint=cross_time if cross_time is not None else start,
                )
        self._condition_start = {k: v for k, v in self._condition_start.items() if k[0] != label}
        return self._inactive(label)

    def _red_light(self, state: FrameState) -> RuleSignal:
        label = "red_light"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.stop_lines:
            return self._inactive(label)
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or scene.nearest_red_light(track) is None:
                self._red_fired = {
                    key: value for key, value in self._red_fired.items() if key[0] != track.track_id
                }
                continue
            lane = scene.lane_for_point(track.center)
            if lane is None:
                continue
            points = track.recent_bottom_points(2.5, state.timestamp)
            if not points:
                points = track.recent_points(2.5, state.timestamp)
            if len(points) < 2:
                continue
            heading = normalized_vector(track.velocity)
            lane_direction = normalized_vector(lane.direction)
            moving_into_intersection = (
                heading is not None
                and lane_direction is not None
                and vector_dot(heading, lane_direction) > 0.2
            )
            for line in scene.config.stop_lines:
                previous = points[-2]
                current = points[-1]
                previous_side = scene.side_of_line((previous[1], previous[2]), line)
                current_side = scene.side_of_line((current[1], current[2]), line)
                approach_side = (
                    scene.side_of_line(lane.approach_point, line)
                    if lane.approach_point is not None
                    else previous_side
                )
                trajectory_crossed = segment_crossed_by_track(
                    (previous[1], previous[2]),
                    (current[1], current[2]),
                    line.segment,
                )
                crossed = trajectory_crossed and moving_into_intersection
                if previous_side != 0 and current_side != 0 and approach_side != 0:
                    crossed = crossed and previous_side == approach_side and current_side == -approach_side
                key = (track.track_id, line.line_id)
                if crossed and key not in self._red_fired:
                    self._red_fired[key] = previous[0]
                start = self._red_fired.get(key)
                if start is not None and state.timestamp - start <= float(cfg.get("hold", 0.8)):
                    return RuleSignal(
                        label,
                        True,
                        confidence=0.9,
                        evidence={"track_id": track.track_id, "line_id": line.line_id, "cross_time": start},
                        start_hint=start,
                    )
        return self._inactive(label)

    def _stop_line(self, state: FrameState) -> RuleSignal:
        # The competition definition treats stop_line as a vehicle stopped
        # beyond the line during a red phase, rather than any line contact.
        return self._line_rule(state, "stop_line", require_red=True)

    def _solid_line_crossing(self, state: FrameState) -> RuleSignal:
        return self._line_rule(state, "solid_line_crossing", require_red=False)

    def _turn_rule(self, state: FrameState, label: str) -> RuleSignal:
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if label == "illegal_u_turn":
            if not getattr(scene, "has_road", False):
                return self._inactive(label)
        elif not getattr(scene, "has_geometry", False):
            return self._inactive(label)
        for track in state.tracks:
            if not TrackManager.is_vehicle(track):
                continue
            if label == "illegal_u_turn":
                current_lane = track.lane_id
                lane = next(
                    (
                        item
                        for item in (scene.config.lanes if scene.config else [])
                        if item.lane_id == current_lane
                    ),
                    None,
                )
            else:
                if len(track.lane_history) < 2:
                    continue
                previous_lane, current_lane = track.lane_history[-2:]
                if previous_lane == current_lane or current_lane is None:
                    continue
                lane = next((item for item in scene.config.lanes if item.lane_id == current_lane), None)
                if lane is None:
                    continue
            heading = track.heading(2.5, state.timestamp)
            if heading is None:
                continue
            lane_direction = normalized_vector(lane.direction) if lane is not None else normalized_vector(track.velocity)
            heading_unit = normalized_vector(heading)
            if lane_direction is None or heading_unit is None:
                continue
            angle = angle_between(heading_unit, lane_direction)
            if label == "illegal_u_turn":
                turn_angle = self._heading_change(track, state.timestamp)
                allowed = (
                    {x.lower().replace("-", "_") for x in lane.allowed_moves}
                    if lane is not None
                    else set()
                )
                condition = turn_angle >= float(cfg.get("angle", 2.35)) and not (
                    "u_turn" in allowed or "uturn" in allowed
                )
                evidence = {"track_id": track.track_id, "turn_angle": turn_angle}
            else:
                # A large heading deviation from the lane is treated as a turn;
                # only flag it when the destination lane does not allow it.
                movement = "right" if heading[0] > 0 else "left"
                if angle < 0.55:
                    movement = "straight"
                allowed = {x.lower().replace("-", "_") for x in lane.allowed_moves}
                condition = angle >= 0.55 and movement not in allowed
                evidence = {"track_id": track.track_id, "angle": angle, "movement": movement}
            key = (label, track.track_id)
            since = self._condition_since(key, condition, state.timestamp)
            if condition and since is not None and state.timestamp - since <= float(cfg.get("hold", 0.8)) + 1.0:
                return RuleSignal(label, True, confidence=0.7, evidence=evidence, start_hint=since)
        self._condition_start = {k: v for k, v in self._condition_start.items() if k[0] != label}
        return self._inactive(label)

    def _illegal_turn(self, state: FrameState) -> RuleSignal:
        return self._turn_rule(state, "illegal_turn")

    def _illegal_u_turn(self, state: FrameState) -> RuleSignal:
        return self._turn_rule(state, "illegal_u_turn")

    def _jaywalking(self, state: FrameState) -> RuleSignal:
        label = "jaywalking"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return self._inactive(label)
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if not TrackManager.is_person(track) or not self._road_track(track, scene):
                continue
            key = (label, track.track_id)
            if scene.crossing_for_point(track.center) is not None:
                self._condition_start.pop(key, None)
                continue
            active_keys.add(key)
            since = self._condition_since(key, True, state.timestamp)
            if since is not None and state.timestamp - since >= float(cfg.get("duration", 1.0)):
                self._retain_conditions(label, active_keys)
                return RuleSignal(label, True, confidence=0.8, evidence={"track_id": track.track_id}, start_hint=since)
        self._retain_conditions(label, active_keys)
        return self._inactive(label)

    def _failure_to_yield(self, state: FrameState) -> RuleSignal:
        label = "failure_to_yield"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.crossings:
            return self._inactive(label)
        people = [t for t in state.tracks if TrackManager.is_person(t)]
        vehicles = [t for t in state.tracks if TrackManager.is_vehicle(t) and t.speed > 3.0]
        for person in people:
            crossing = scene.crossing_for_point(person.center)
            if crossing is None:
                continue
            for vehicle in vehicles:
                vehicle_crossing = scene.crossing_for_point(vehicle.center) or scene.crossing_for_point(vehicle.bottom_center)
                if vehicle_crossing is not crossing and not (
                    vehicle_crossing is not None
                    and vehicle_crossing[0] == crossing[0]
                    and vehicle_crossing[1] == crossing[1]
                ):
                    continue
                key = (label, person.track_id, vehicle.track_id)
                since = self._condition_since(key, True, state.timestamp)
                if since is not None:
                    return RuleSignal(
                        label,
                        True,
                        confidence=0.75,
                        evidence={"person_id": person.track_id, "vehicle_id": vehicle.track_id},
                        start_hint=since,
                    )
        self._condition_start = {k: v for k, v in self._condition_start.items() if k[0] != label}
        return self._inactive(label)

    def _interaction_rule(self, state: FrameState, label: str) -> RuleSignal:
        cfg = self._rule_config(label)
        road_users = [track for track in state.tracks if TrackManager.is_road_user(track)]
        for i, first in enumerate(road_users):
            for second in road_users[i + 1 :]:
                if not (
                    TrackManager.is_vehicle(first)
                    or TrackManager.is_person(first)
                    or TrackManager.is_vehicle(second)
                    or TrackManager.is_person(second)
                ):
                    continue
                key = (label, first.track_id, second.track_id)
                distance = TrackManager.center_distance(first, second)
                ttc = TrackManager.time_to_collision(first, second)
                relative_speed = TrackManager.relative_speed(first, second)
                overlap = bbox_iou(first.bbox, second.bbox)
                if label == "near_miss":
                    evasive = (
                        self._deceleration(first) > 18.0
                        or self._deceleration(second) > 18.0
                        or max(first.acceleration_magnitude, second.acceleration_magnitude) > 35.0
                    )
                    condition = (
                        ttc is not None
                        and 0.0 < ttc <= float(cfg.get("ttc", 2.5))
                        and distance <= float(cfg.get("distance", 120.0))
                        and relative_speed > 8.0
                        and evasive
                    )
                    evidence = {
                        "first_id": first.track_id,
                        "second_id": second.track_id,
                        "ttc": ttc,
                        "distance": distance,
                    }
                else:  # accident
                    abrupt = max(first.acceleration_magnitude, second.acceleration_magnitude) > 45.0
                    condition = (
                        (overlap > 0.08 or distance <= float(cfg.get("distance", 55.0)))
                        and (relative_speed > 12.0 or abrupt)
                    )
                    evidence = {
                        "first_id": first.track_id,
                        "second_id": second.track_id,
                        "distance": distance,
                        "overlap": overlap,
                    }
                since = self._condition_since(key, condition, state.timestamp)
                if condition and since is not None and state.timestamp - since <= float(cfg.get("hold", 0.8)) + 1.0:
                    return RuleSignal(label, True, confidence=0.75, evidence=evidence, start_hint=since)
        self._condition_start = {k: v for k, v in self._condition_start.items() if k[0] != label}
        return self._inactive(label)

    def _near_miss(self, state: FrameState) -> RuleSignal:
        return self._interaction_rule(state, "near_miss")

    def _accident(self, state: FrameState) -> RuleSignal:
        return self._interaction_rule(state, "accident")

    def _special_visual(self, state: FrameState, label: str) -> RuleSignal:
        cfg = self._rule_config(label)
        scene = self._scene(state)
        words = ("fire", "smoke", "flame") if label == "fire_smoke" else ("obstacle", "debris", "animal", "cone", "barrier", "fallen")
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if not self._special(track, words):
                continue
            if getattr(scene, "has_road", False) and not self._road_track(track, scene):
                continue
            key = (label, track.track_id)
            active_keys.add(key)
            since = self._condition_since(key, True, state.timestamp)
            if since is not None and state.timestamp - since >= float(cfg.get("duration", 0.5)):
                self._retain_conditions(label, active_keys)
                return RuleSignal(label, True, confidence=0.8, evidence={"track_id": track.track_id}, start_hint=since)
        self._retain_conditions(label, active_keys)
        return self._inactive(label)

    def _road_obstacle(self, state: FrameState) -> RuleSignal:
        return self._special_visual(state, "road_obstacle")

    def _fire_smoke(self, state: FrameState) -> RuleSignal:
        return self._special_visual(state, "fire_smoke")

    def evaluate(self, state: FrameState) -> dict[str, RuleSignal]:
        methods: dict[str, Callable[[FrameState], RuleSignal]] = {
            "accident": self._accident,
            "near_miss": self._near_miss,
            "red_light": self._red_light,
            "wrong_way": self._wrong_way,
            "illegal_u_turn": self._illegal_u_turn,
            "stopped_vehicle": self._stopped_vehicle,
            "jaywalking": self._jaywalking,
            "failure_to_yield": self._failure_to_yield,
            "illegal_turn": self._illegal_turn,
            "solid_line_crossing": self._solid_line_crossing,
            "stop_line": self._stop_line,
            "congestion": self._congestion,
            "road_obstacle": self._road_obstacle,
            "fire_smoke": self._fire_smoke,
        }
        output: dict[str, RuleSignal] = {}
        for label, method in methods.items():
            if not self._enabled(label):
                output[label] = self._inactive(label)
                continue
            try:
                output[label] = method(state)
            except Exception as exc:
                warnings.warn(f"rule {label} failed on frame {state.frame_id}: {exc}")
                output[label] = self._inactive(label)
        return output
