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
        # Candidate impacts awaiting confirmation: (onset_time, approach_mps, max_speed_mps)
        self._pending: dict[tuple[Any, ...], tuple[float, float, float]] = {}

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
    def _scale_of(a: TrackState, b: TrackState) -> float:
        scales = [x for x in (a.metres_per_pixel, b.metres_per_pixel) if x]
        return max(scales) if scales else 0.0

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
            # Same rule as _in_a_queue: an unmeasured speed is not a stopped
            # speed, so it must not make a track look like a queue member.
            if self._near_line(other, line, 140.0) and other.speed_mps is not None and other.speed_mps <= 1.0:
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
            # A track seen only once has no measured speed. Treating that as
            # "stopped" would let a brand-new track vouch for a queue it has
            # not been observed in, so unknown speed is skipped rather than
            # assumed.
            if other.speed_mps is None or other.speed_mps > stopped_mps:
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
        # A box clipped by the frame boundary is a poor basis for any judgement:
        # its apparent length is truncated, so the scale used to convert pixels
        # to metres is wrong, and a static object sitting against the edge of
        # the image is far more likely to be a parked vehicle or a piece of
        # street furniture than a breakdown in a traffic lane. Measured over a
        # development clip there was a track pinned to x=[0,175] at 0.02-0.05 m/s
        # for the whole clip, inside a road polygon, and it satisfied every other
        # condition for stopped_vehicle indefinitely.
        margin = float(cfg.get("frame_margin_px", 8.0))
        # Frame dimensions live on the SceneState, not the FrameState; reading
        # them off the wrong object silently disables the test.
        width = float(getattr(getattr(state, "scene", None), "width", 0) or 0)
        height = float(getattr(getattr(state, "scene", None), "height", 0) or 0)
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or not self._road_track(track, scene):
                continue
            if (
                width
                and height
                and (
                    track.bbox[0] <= margin
                    or track.bbox[1] <= margin
                    or track.bbox[2] >= width - margin
                    or track.bbox[3] >= height - margin
                )
            ):
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
            # Only tracks with a measured speed can be called slow. A track
            # seen once reports no speed at all, and counting it as slow would
            # manufacture congestion out of the first frame of every object.
            slow = [
                track
                for track in tracks
                if track.speed_mps is not None and track.speed_mps <= crawl_mps
            ]
            moving = [track for track in tracks if track.speed_mps is not None]
            if len(moving) < minimum or len(slow) < minimum:
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
        # A person standing still in the road is a different event from one
        # crossing it, and the definition is about crossing: steps onto the
        # road, leaves the road. Measured over 30 s this rule was active on
        # 36% of frames from 8 distinct tracks, and the only test it applied was
        # "is a person inside a road polygon and outside a crossing polygon" --
        # which is also true of anyone waiting at a kerb that the hand-drawn
        # polygon happens to cover, and of anyone on a median.
        #
        # So require sustained presence and real movement. Movement is the part
        # that separates a person using the road from a person standing on it,
        # and requiring the whole box to stay inside the carriageway removes the
        # kerbside cases that a centre-point test cannot.
        min_speed = float(cfg.get("min_speed_mps", 0.4))
        for track in state.tracks:
            if not TrackManager.is_person(track) or not self._road_track(track, scene):
                continue
            key = (label, track.track_id)
            if scene.crossing_for_point(track.center) is not None:
                self._condition_start.pop(key, None)
                continue
            speed = track.speed_mps
            # The whole body must be over the carriageway, not one toe: a box
            # that is mostly off the road polygon is someone on the kerb.
            corners = (
                (track.bbox[0], track.bbox[1]), (track.bbox[2], track.bbox[1]),
                (track.bbox[0], track.bbox[3]), (track.bbox[2], track.bbox[3]),
                track.center, track.bottom_center,
            )
            inside = sum(1 for point in corners if scene.is_road_point(point))
            # A majority, not unanimity. The road polygons are hand-drawn against
            # a coordinate grid, so demanding that all six sample points fall
            # inside suppressed every event in the clip -- the geometry is not
            # accurate enough to justify that. Requiring a majority still
            # excludes someone standing on a kerb, which is what the test is for.
            if speed is None or speed < min_speed or inside * 2 < len(corners):
                self._condition_start.pop(key, None)
                continue
            since = self._condition_since(key, True, state.timestamp)
            if since is not None and state.timestamp - since >= float(cfg.get("duration", 1.0)):
                self._retain_conditions(label, {key})
                return RuleSignal(
                    label,
                    True,
                    confidence=0.8,
                    evidence={
                        "track_id": track.track_id,
                        "speed_mps": round(speed, 2),
                        "on_road_points": f"{inside}/{len(corners)}",
                    },
                    start_hint=since,
                )
        self._retain_conditions(label, set())
        return self._inactive(label)

    def _failure_to_yield(self, state: FrameState) -> RuleSignal:
        label = "failure_to_yield"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.crossings:
            return self._inactive(label)
        people = [t for t in state.tracks if TrackManager.is_person(t)]
        vehicles = [
            t
            for t in state.tracks
            if TrackManager.is_vehicle(t)
            and t.speed_mps is not None
            and t.speed_mps > float(cfg.get("moving_mps", 1.0))
        ]
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
                # The pedestrian has to be in the vehicle's path, and the
                # vehicle has to be arriving rather than already past. "Some
                # vehicle is somewhere on the same crossing" is otherwise true
                # for every vehicle that traverses it, which is most of them.
                if not self._in_path(vehicle, person, state):
                    continue
                # The discriminator. A vehicle that fails to yield is one that
                # arrives at an occupied crossing and *keeps its speed*; a
                # vehicle that yields brakes. The previous version of this rule
                # only asked for a vehicle moving faster than 1 m/s, which a
                # car decelerating from 12 m/s to a stop is still doing for most
                # of a second -- so the rule was firing on correct behaviour.
                # Requiring the absence of braking is what separates the two.
                # 2.0 m/s of speed lost in a second is a light brake; a driver
                # yielding to a pedestrian on a crossing loses considerably
                # more than that.
                if self._speed_drop(vehicle, state.timestamp) >= float(
                    cfg.get("yield_brake_mps", 2.0)
                ):
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
                            evidence={
                                "person_id": person.track_id,
                                "vehicle_id": vehicle.track_id,
                                "vehicle_mps": round(vehicle.speed_mps, 2),
                                "brake_mps": round(
                                    self._speed_drop(vehicle, state.timestamp), 2
                                ),
                            },
                            start_hint=since,
                        )
        self._retain_conditions(label, active_keys)
        return self._inactive(label)

    def _in_path(self, vehicle: TrackState, person: TrackState, state: FrameState) -> bool:
        """True when the person lies ahead of the vehicle, inside its width.

        Uses the vehicle's direction of travel rather than the image axes, so
        it works for an approach from any side of the junction. The person
        counts as in-path when they are within half a car width of the line
        the vehicle is travelling along: inside the corridor means a conflict,
        outside it means the pedestrian is crossing somewhere else.
        """
        vx, vy = vehicle.velocity
        norm = math.hypot(vx, vy)
        if norm < 1e-3:
            return False
        ux, uy = vx / norm, vy / norm
        dx = person.center[0] - vehicle.center[0]
        dy = person.center[1] - vehicle.center[1]
        ahead = dx * ux + dy * uy
        if ahead <= 0.0:
            return False
        lateral = abs(-dx * uy + dy * ux)
        return lateral <= max(1.0, 0.6 * vehicle.width)

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
                    # Longitudinal following behind an already-stopped or still
                    # rolling vehicle is how every queue forms, and a hard brake
                    # at a red light is the commonest thing that happens at this
                    # intersection. Measured over 300 frames there were 382
                    # close-and-braking pairs -- 1.3 per frame of ordinary
                    # braking in dense traffic.
                    #
                    # Two candidate discriminators were tried and both were
                    # refuted by measurement. Lateral closing speed does not
                    # separate the cases at all: 87% of those pairs exceed
                    # 1.0 m/s with a median of 5.78 m/s, because adjacent lanes at
                    # different depths produce large image-space horizontal
                    # velocities, so the quantity measures perspective rather
                    # than a manoeuvre. A "leader is stopped" test fails too,
                    # because a queue leader is still rolling when the follower
                    # brakes.
                    #
                    # So the only conflict this camera can actually resolve is a
                    # vulnerable road user in a vehicle's path: 5 of the 382
                    # pairs. That is the condition used here. It is a deliberate
                    # precision-for-recall trade -- vehicle-to-vehicle near
                    # misses are not separable from normal queue braking at this
                    # camera height and resolution, and emitting 1.3 candidates
                    # per frame would guarantee a zero for the class.
                    vulnerable = any(TrackManager.is_person(t) for t in (first, second)) or any(
                        self._special(t, ("bicycle", "motorcycle", "bike"))
                        for t in (first, second)
                    )
                    condition = (
                        vulnerable
                        and evasive >= float(cfg.get("speed_drop_mps", 2.5))
                        and closing >= float(cfg.get("closing_mps", 2.0))
                        and (
                            (gap is not None and gap <= float(cfg.get("gap_m", 3.0)))
                            or overlap > float(cfg.get("overlap", 0.02))
                        )
                    )
                    evidence["deceleration_mps2"] = round(evasive, 2)
                    evidence["vulnerable"] = bool(vulnerable)
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
                    if not fast_approach:
                        self._pending.pop(key, None)
                        continue
                    if onset:
                        # Start of a candidate impact. Nothing is reported yet:
                        # contact while closing happens constantly between cars
                        # in adjacent lanes, and measured over a 30 s window
                        # this rule was active on 26% of frames with a median
                        # box overlap of 0.007, which is two boxes grazing, not
                        # two vehicles colliding.
                        self._pending[key] = (
                            state.timestamp,
                            approach,
                            # Both parties are closing at a measured rate here,
                            # so both speeds are known; max() over a pair that
                            # might include an unmeasured track would silently
                            # record 0.0 m/s and make every later "did it slow
                            # down?" test trivially true.
                            max(
                                first.speed_mps if first.speed_mps is not None else 0.0,
                                second.speed_mps if second.speed_mps is not None else 0.0,
                            ),
                        )
                        continue
                    # Confirmation: a collision destroys momentum. The pair that
                    # was closing at `approach` must have shed it, either as a
                    # hard deceleration or as the gap ceasing to close. Ordinary
                    # following does neither -- the follower keeps rolling up at
                    # a steady rate -- so this separates an impact from a car
                    # pulling up behind another.
                    started = self._pending.get(key)
                    if started is None:
                        continue
                    onset_t, onset_approach, onset_speed = started
                    if state.timestamp - onset_t > float(cfg.get("confirm_sec", 1.5)):
                        self._pending.pop(key, None)
                        continue
                    collapsed = approach <= onset_approach * float(
                        cfg.get("collapse_fraction", 0.5)
                    )
                    braked = max(
                        self._speed_drop(first, state.timestamp),
                        self._speed_drop(second, state.timestamp),
                    ) >= float(cfg.get("confirm_speed_drop_mps", 3.0))
                    # "It slowed down" is only evidence if the speed after the
                    # contact is actually measured. An unmeasured track reports
                    # no speed, and 0.0 would satisfy this test on its own.
                    slowed = (
                        onset_speed > 0
                        and first.speed_mps is not None
                        and second.speed_mps is not None
                        and max(first.speed_mps, second.speed_mps)
                        <= onset_speed * float(cfg.get("collapse_fraction", 0.5))
                    )
                    evidence["approach_mps"] = round(approach, 2)
                    evidence["onset_approach_mps"] = round(onset_approach, 2)
                    evidence["deceleration_mps2"] = round(decel, 2)
                    evidence["onset"] = True
                    if not (collapsed or braked or slowed):
                        continue
                    self._pending.pop(key, None)
                    condition = heavy

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
                self._pending = {k: v for k, v in self._pending.items() if k in live}
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
