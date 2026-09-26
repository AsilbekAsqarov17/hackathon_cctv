from __future__ import annotations

import math
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
        # Per-pair contact history for accident: was this pair touching a moment
        # ago? A collision is a transition, a queue is a steady state.
        self._contact: dict[tuple[Any, ...], tuple[float, bool]] = {}

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
        return track.deceleration_mps2()

    @staticmethod
    def _speed_drop(track: TrackState, timestamp: float, window: float = 1.0) -> float:
        """Metres per second lost over the last ``window`` seconds.

        This is the braking signal, and it is deliberately a *sustained* speed
        change rather than the instantaneous acceleration. Differentiating a
        smoothed velocity over a 0.1 s sampling interval mostly measures
        detector jitter: a car held at a steady 10 m/s easily shows 5 m/s^2 of
        apparent deceleration as the box wobbles by a few pixels. Losing
        2.5 m/s over a full second, by contrast, is roughly a quarter of g and
        does not happen to a stationary box.

        Returns 0.0 when the history is too short to judge.
        """
        points = track.recent_points(window, timestamp)
        if len(points) < 2:
            return 0.0
        t0, x0, y0 = points[0]
        t1, x1, y1 = points[-1]
        dt = t1 - t0
        if dt < 0.25 * window:
            return 0.0
        start = math.hypot(x0, y0)
        end = math.hypot(x1, y1)
        scale = track.metres_per_pixel
        if scale is None:
            return 0.0
        return max(0.0, (start - end) * scale)

    @staticmethod
    def _closing_speed_mps(first: TrackState, second: TrackState) -> float:
        """Rate at which the gap between two tracks shrinks, in m/s."""
        dx = second.center[0] - first.center[0]
        dy = second.center[1] - first.center[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return 0.0
        rel_x = second.velocity[0] - first.velocity[0]
        rel_y = second.velocity[1] - first.velocity[1]
        rate = -(dx * rel_x + dy * rel_y) / distance
        # Convert with the larger of the two scales so the pair is judged in
        # metres even when one of the boxes is too small to measure.
        scales = [s for s in (first.metres_per_pixel, second.metres_per_pixel) if s]
        scale = max(scales) if scales else None
        return max(0.0, rate * scale) if scale else 0.0

    @staticmethod
    def _gap_m(first: TrackState, second: TrackState) -> float | None:
        """Edge-to-edge distance between two boxes, in metres.

        Box *centres* are a poor proxy for contact: two cars in adjacent lanes
        have close centres while their bodies are a metre apart. The gap
        between the boxes themselves is what "contact is visible" means.
        """
        dx = max(0.0, max(first.bbox[0], second.bbox[0]) - min(first.bbox[2], second.bbox[2]))
        dy = max(0.0, max(first.bbox[1], second.bbox[1]) - min(first.bbox[3], second.bbox[3]))
        gap_px = math.hypot(dx, dy)
        scales = [s for s in (first.metres_per_pixel, second.metres_per_pixel) if s]
        if not scales:
            return None
        return gap_px * max(scales)

    def _wrong_way(self, state: FrameState) -> RuleSignal:
        label = "wrong_way"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False):
            return self._inactive(label)
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            speed = track.speed_mps
            if not TrackManager.is_vehicle(track) or speed is None or speed < float(cfg.get("min_speed_mps", 1.5)):
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
            if self._near_line(other, line, 140.0) and (other.speed_mps or 0.0) <= 1.0:
                return True
        return False

    def _in_a_queue(self, track: TrackState, state: FrameState) -> bool:
        """True when this stopped vehicle is part of a cluster of stopped ones.

        The task excludes "a queue at a signal" from `stopped_vehicle`. Detecting
        a signal queue geometrically needs stop lines, which are the least
        reliable part of the scene calibration. The queue itself is easier to
        see than the signal: a queue is a dense group of stationary vehicles, a
        broken-down car is alone. That test needs no geometry, and it is
        deliberately heading-free.

        Two earlier versions of this test failed, both for the same reason: they
        used the stopped vehicle's own heading to decide who was "ahead". A
        stationary vehicle has no heading -- its velocity is sensor jitter, so
        the direction is essentially random each frame, and in dense traffic
        some stopped car is always in whatever direction that points. The first
        version compounded it by using the other vehicle's own width as the
        lateral tolerance, about 245 px here, which made the test always true and
        suppressed the single real event in the dev set.
        """
        cfg = self._rule_config("stopped_vehicle")
        stopped_mps = float(cfg.get("speed_mps", 0.6))
        # Roughly two car lengths. A queue packs tighter than this; a stranded
        # vehicle in an open junction has nothing within it.
        radius = float(cfg.get("queue_radius_m", 7.0))
        scales = [s for s in (track.metres_per_pixel,) if s]
        for other in state.tracks:
            if other.track_id == track.track_id or not TrackManager.is_vehicle(other):
                continue
            if (other.speed_mps or 0.0) > stopped_mps:
                continue
            scale = track.metres_per_pixel
            if scale is None:
                continue
            gap = math.hypot(
                other.center[0] - track.center[0], other.center[1] - track.center[1]
            ) * scale
            if gap <= radius:
                return True
        return False

    def _stopped_vehicle(self, state: FrameState) -> RuleSignal:
        label = "stopped_vehicle"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return self._inactive(label)
        # "Stationary" in the task definition, so the threshold is a real speed.
        # 0.6 m/s is about 2 km/h: slow enough to exclude crawling traffic.
        threshold = float(cfg.get("speed_mps", 0.6))
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or not self._road_track(track, scene):
                continue
            speed = track.speed_mps
            if speed is None:
                continue
            key = (label, track.track_id)
            if speed > threshold or self._queued_at_signal(track, state) or self._in_a_queue(track, state):
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
                    evidence={"track_id": track.track_id, "stopped_since": round(since, 2)},
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
        # "Standstill or crawling": 2.2 m/s is roughly 8 km/h, the usual
        # definition of congested flow. Expressed physically so the same value
        # works at any source resolution.
        crawl_mps = float(cfg.get("crawl_mps", 2.2))
        minimum = int(cfg.get("min_vehicles", 4))
        active_keys: set[tuple[Any, ...]] = set()
        for key, tracks in groups.items():
            slow = [track for track in tracks if (track.speed_mps or 0.0) <= crawl_mps]
            if len(tracks) < minimum or len(slow) < minimum:
                continue
            condition_key = (label, key)
            active_keys.add(condition_key)
            since = self._condition_since(condition_key, True, state.timestamp)
            if since is not None and state.timestamp - since >= float(cfg.get("duration", 5.0)):
                self._retain_conditions(label, active_keys)
                return RuleSignal(
                    label,
                    True,
                    confidence=min(1.0, len(slow) / max(1, minimum) * 0.7),
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
        vehicles = [t for t in state.tracks if TrackManager.is_vehicle(t) and t.speed_mps and t.speed_mps > 1.0]
        active_keys: set[tuple[Any, ...]] = set()
        for person in people:
            # The pedestrian must be *on* the crossing, not merely near it. With
            # eight or nine people in every frame of these clips, "a pedestrian
            # and a vehicle are both somewhere near this crossing" is true almost
            # all the time, which is what made this the second-largest source of
            # false positives.
            standing_on = scene.crossing_for_point(person.center)
            if standing_on is None:
                continue
            for vehicle in vehicles:
                crossing = scene.crossing_for_point(vehicle.center)
                if crossing is None:
                    crossing = scene.crossing_for_point(vehicle.bottom_center)
                if crossing is None or list(crossing) != list(standing_on):
                    continue
                key = (label, person.track_id, vehicle.track_id)
                since = self._condition_since(key, True, state.timestamp)
                if since is not None:
                    active_keys.add(key)
                    if state.timestamp - since <= float(cfg.get("hold", 2.0)) + 1.0:
                        return RuleSignal(
                            label,
                            True,
                            confidence=0.75,
                            evidence={"person_id": person.track_id, "vehicle_id": vehicle.track_id},
                            start_hint=since,
                        )
        self._retain_conditions(label, active_keys)
        return self._inactive(label)

    def _interaction_rule(self, state: FrameState, label: str) -> RuleSignal:
        cfg = self._rule_config(label)
        road_users = [track for track in state.tracks if TrackManager.is_road_user(track)]
        active_keys: set[tuple[Any, ...]] = set()
        for i, first in enumerate(road_users):
            for second in road_users[i + 1 :]:
                key = (label, first.track_id, second.track_id)
                gap = self._gap_m(first, second)
                overlap = bbox_iou(first.bbox, second.bbox)
                closing = self._closing_speed_mps(first, second)
                ttc = TrackManager.time_to_collision(first, second)
                evidence = {
                    "first_id": first.track_id,
                    "second_id": second.track_id,
                    "gap_m": None if gap is None else round(gap, 2),
                    "closing_mps": round(closing, 2),
                    "overlap": round(overlap, 3),
                }

                if label == "near_miss":
                    # An evasive manoeuvre is a transient: someone brakes or
                    # swerves, and a moment later the road users are clear of
                    # each other. Requiring an actual deceleration is what
                    # separates this from two cars merely driving near each
                    # other, which happens constantly in dense traffic.
                    evasive = max(
                        self._speed_drop(first, state.timestamp),
                        self._speed_drop(second, state.timestamp),
                    )
                    side_swipe = abs(first.center[0] - second.center[0]) > 0.25 * max(
                        first.width, second.width, 1.0
                    )
                    condition = (
                        evasive >= float(cfg.get("deceleration_mps2", 2.5))
                        and closing >= float(cfg.get("closing_mps", 2.0))
                        and (
                            (gap is not None and gap <= float(cfg.get("gap_m", 3.0)))
                            or overlap > float(cfg.get("overlap", 0.02))
                        )
                        and (side_swipe or ttc is None or ttc <= float(cfg.get("ttc", 2.0)))
                    )
                    evidence["deceleration_mps2"] = round(evasive, 2)
                else:  # accident
                    # Contact is a geometric fact, not a speed threshold: once
                    # two bodies overlap, they overlap. Requiring closing speed
                    # at the moment of contact is what used to make this rule
                    # fire on ordinary following distance, and it also made it
                    # stop firing at the collision itself, because both
                    # vehicles decelerate hard on impact.
                    if gap is None:
                        self._contact.pop(key, None)
                        continue
                    contact = gap <= float(cfg.get("contact_gap_m", 0.6)) or overlap > float(
                        cfg.get("overlap", 0.05)
                    )
                    # A real collision involves at least one heavy road user;
                    # a pedestrian brushing a bin is not an accident.
                    heavy = any(
                        TrackManager.is_vehicle(t) for t in (first, second)
                    ) or self._special(first, ("bicycle", "motorcycle", "bike"))

                    # The decisive test is the *transition*. In dense traffic
                    # adjacent cars overlap continuously while queueing, so
                    # "these boxes overlap" is true for most of a red light and
                    # means nothing. A collision is the pair going from
                    # separated to touching, and the impact is then marked by
                    # both road users shedding speed abruptly. Requiring that
                    # edge is what separates a crash from a queue.
                    was_contact, since_contact = self._contact.get(key, (0.0, False))
                    self._contact[key] = (state.timestamp, contact)
                    separated_before = (
                        not was_contact
                        or state.timestamp - since_contact >= float(cfg.get("settle_sec", 0.6))
                    )
                    onset = contact and separated_before
                    decel = max(self._deceleration(first), self._deceleration(second))
                    impact = decel >= float(cfg.get("impact_decel_mps2", 4.0))
                    if not contact:
                        continue
                    # An onset alone is not enough. In a busy intersection some
                    # pair of boxes is always touching, so "separated, then
                    # touching" happens constantly among queued cars. What
                    # separates a crash from a queue is that the pair was
                    # closing on each other: adjacent stationary cars have a
                    # closing speed near zero, a collision does not. Velocity is
                    # exponentially smoothed, so it still carries the approach
                    # speed on the frame contact is first seen.
                    approach = self._closing_speed_mps(first, second)
                    fast_approach = approach >= float(cfg.get("approach_closing_mps", 2.5))
                    if onset:
                        condition = heavy and fast_approach
                    else:
                        condition = heavy and impact
                    evidence["approach_mps"] = round(approach, 2)
                    evidence["deceleration_mps2"] = round(decel, 2)
                    evidence["onset"] = bool(onset)

                since = self._condition_since(key, condition, state.timestamp)
                if not condition:
                    continue
                active_keys.add(key)
                # A sustained condition is one event, not a new one per frame:
                # the onset is remembered, and the signal is only emitted while
                # the pair is still inside the event window.
                if since is not None and state.timestamp - since <= float(cfg.get("hold", 2.0)):
                    return RuleSignal(label, True, confidence=0.75, evidence=evidence, start_hint=since)
            # Forget pairs that no longer exist so the state cannot grow without
            # bound over a long video.
            if label == "accident":
                live = {
                    (label, first.track_id, second.track_id)
                    for i, first in enumerate(road_users)
                    for second in road_users[i + 1 :]
                }
                self._contact = {k: v for k, v in self._contact.items() if k in live}
        self._retain_conditions(label, active_keys)
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
