"""Frame-level rule evaluation for the Part A event classes.

Design notes that matter for correctness
----------------------------------------

**Candidates, not a single flag.** Several classes can be true for several
objects at once (two cars run a red light, a crowd jaywalks). A rule that
returns one boolean per frame would silently collapse them into a single
segment, so every rule exposes ``_<label>_candidates`` returning one signal per
distinct object or group, and ``evaluate`` returns the highest-confidence one
for the per-frame API. ``collect_signals`` is what the pipeline uses.

**Object identity.** A condition that spans many frames has to be attributed to
something stable, otherwise its timer resets every time the tracker re-IDs. Keys
are therefore ``(label, track_id, ...)`` tuples, and timers survive only while
the key keeps being reported.

**Causality.** Nothing reads a future frame. Every test uses a trailing window
closed at the current timestamp, so replaying the same track dump reproduces
the same decisions.
"""
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
    point_in_polygon,
    vector_dot,
)
from ..tracking.track_manager import TrackManager
from .kinematics import (
    across_line,
    closing_speed,
    deceleration,
    distance_to_line,
    front_crossing,
    heading_reversal,
    line_length,
    net_heading,
    road_frame_gap,
    swept_gap,
    time_to_collision,
)
from .motion import is_stationary, window_heading

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

# Colours that mean "must stop". Yellow is deliberately excluded: the official
# definition is a red-light violation, and a signal caught mid-amber-change or
# partially occluded must not be promoted to red.
STOPPING_COLORS = frozenset({"red"})


class RuleEngine:
    """Evaluates frame-level rules from a canonical :class:`FrameState`.

    Rules deliberately do not perform inference. They only consume track and
    scene state, which keeps the competition logic independently testable
    without a video or a GPU.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._condition_start: dict[tuple[Any, ...], float] = {}
        self._wrong_way_counts: dict[int, int] = {}
        # (track_id, line_id) -> crossing timestamp, so one crossing is reported
        # once even though it stays true for many frames.
        self._crossings: dict[tuple[Any, ...], float] = {}
        self._last_evidence: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # configuration and small helpers
    # ------------------------------------------------------------------
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

    def _signal_color(self, scene: Any, line: Any) -> str:
        """Colour of the head governing ``line``; ``unknown`` when unreadable."""
        getter = getattr(scene, "signal_color_for_line", None)
        if getter is None:
            return "unknown"
        try:
            return str(getter(line))
        except Exception:  # a malformed scene must not crash a rule
            return "unknown"

    def _vehicle_lane(self, scene: Any, track: TrackState) -> Any:
        """Lane a vehicle occupies, preferring the bottom centre (road contact)."""
        lane = scene.lane_for_point(track.bottom_center)
        if lane is None:
            lane = scene.lane_for_point(track.center)
        return lane

    # ------------------------------------------------------------------
    # wrong_way  (validated in the previous milestone; behaviour preserved)
    # ------------------------------------------------------------------
    def _wrong_way_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Every calibrated-lane vehicle currently moving against its lane.

        Only lanes listed in ``lane_allowlist`` (or, when that is absent, every
        lane that declares a direction) are considered, so regions deliberately
        left without a direction - the east leg and lower intersection - can
        never raise this event.
        """
        label = "wrong_way"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False):
            return []
        allow = cfg.get("lane_allowlist")
        allow_set = {int(v) for v in allow} if allow is not None else None
        window = float(cfg.get("window", 2.5))
        dot_threshold = float(cfg.get("direction_dot", -0.45))
        min_travel = float(cfg.get("min_travel_px", 45.0))
        min_speed = float(cfg.get("min_speed", 6.0))
        confirm_sec = float(cfg.get("confirm_sec", 1.5))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            key = (label, track.track_id)
            if not TrackManager.is_vehicle(track):
                self._wrong_way_counts.pop(track.track_id, None)
                self._condition_start.pop(key, None)
                continue
            # Bottom-centre is the road position for this camera.
            lane = scene.lane_for_point(track.bottom_center)
            if lane is None or (allow_set is not None and lane.lane_id not in allow_set):
                # No calibrated direction here: never flag, never accumulate.
                self._wrong_way_counts.pop(track.track_id, None)
                self._condition_start.pop(key, None)
                continue
            expected = normalized_vector(lane.direction)
            heading = window_heading(track.bottom_history, window, state.timestamp)
            # Suppress: stationary, jittering, entering/leaving frame, or simply
            # too slow to establish a reliable direction.
            if expected is None or heading is None:
                self._wrong_way_counts.pop(track.track_id, None)
                self._condition_start.pop(key, None)
                continue
            observed, travelled = heading
            if travelled < min_travel or track.speed < min_speed:
                self._wrong_way_counts.pop(track.track_id, None)
                self._condition_start.pop(key, None)
                continue
            dot = vector_dot(expected, observed)
            if dot >= dot_threshold:
                self._wrong_way_counts.pop(track.track_id, None)
                self._condition_start.pop(key, None)
                continue
            self._wrong_way_counts[track.track_id] = self._wrong_way_counts.get(track.track_id, 0) + 1
            active_keys.add(key)
            start = self._condition_since(key, True, state.timestamp)
            if start is None or state.timestamp - start < confirm_sec:
                continue
            found.append(
                RuleSignal(
                    label,
                    True,
                    confidence=min(1.0, abs(dot)),
                    evidence={
                        "track_id": track.track_id,
                        "lane_id": lane.lane_id,
                        "dot": round(float(dot), 4),
                        "expected": [round(float(v), 4) for v in expected],
                        "observed": [round(float(v), 4) for v in observed],
                        "travelled_px": round(float(travelled), 1),
                    },
                    start_hint=start,
                )
            )
        self._retain_conditions(label, active_keys)
        self._wrong_way_counts = {
            track_id: count
            for track_id, count in self._wrong_way_counts.items()
            if track_id in {k[1] for k in active_keys}
        }
        return found

    def _wrong_way(self, state: FrameState) -> RuleSignal:
        found = self._wrong_way_candidates(state)
        return found[0] if found else self._inactive("wrong_way")

    # ------------------------------------------------------------------
    # stopped_vehicle  (validated in the previous milestone)
    # ------------------------------------------------------------------
    def _upstream_distance(self, point: tuple[float, float], line: Any) -> float | None:
        """Signed distance of ``point`` upstream of a directed stop line.

        Negative values are upstream of the line's start point; more negative
        means further back in the queue.
        """
        if line.direction is None:
            return None
        a = line.segment[0]
        return (point[0] - a[0]) * line.direction[0] + (point[1] - a[1]) * line.direction[1]

    def _queued_at_signal(self, track: TrackState, state: FrameState) -> bool:
        """True when a stationary vehicle is simply waiting in a signal queue.

        A queue is identified from scene context, not from how long the vehicle
        has been stopped. The vehicle must be upstream of a configured stop
        line, and then either

        * the signal governing that approach is currently red, so it is waiting
          for the light; or
        * another stationary vehicle sits between it and the line in the same
          lane, so it is part of a standing queue.

        The second test matters because the tail of a long queue can be well
        over a thousand pixels back from the line, so a simple "near the line"
        radius would wrongly report it. Without this rule every car waiting at a
        red light would also be reported as ``stopped_vehicle``.
        """
        scene = self._scene(state)
        cfg = self._rule_config("stopped_vehicle")
        window = float(cfg.get("window", 2.5))
        net_max = float(cfg.get("net_max_px", 25.0))
        path_max = float(cfg.get("path_max_px", 250.0))
        line = scene.line_for_track(track)
        if line is None:
            return False
        mine = self._upstream_distance(track.bottom_center, line)
        if mine is None or math.isnan(mine) or mine >= 0.0:
            # Already past the line, or the line has no direction: not queueing.
            return False

        # Waiting for a red light, however far back the queue has grown.
        if self._signal_color(scene, line) in STOPPING_COLORS:
            return True

        # Otherwise: something stationary between this vehicle and the line.
        my_lane = scene.lane_for_point(track.bottom_center)
        for other in state.tracks:
            if other.track_id == track.track_id or not TrackManager.is_vehicle(other):
                continue
            if not is_stationary(other.bottom_history, window, net_max, path_max, state.timestamp):
                continue
            theirs = self._upstream_distance(other.bottom_center, line)
            if theirs is None or math.isnan(theirs) or not (mine < theirs < 0.0):
                continue
            if my_lane is not None:
                other_lane = scene.lane_for_point(other.bottom_center)
                if other_lane is None or other_lane.lane_id != my_lane.lane_id:
                    continue
            return True
        return False

    def _stopped_vehicle_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Every vehicle stationary on the carriageway for ``duration`` seconds.

        Stationarity is judged from bottom-centre history over a trailing window
        (net displacement plus path length), not from an instantaneous box
        comparison. Brief jitter is tolerated for up to ``tolerance_gap``
        seconds so a parked vehicle yields one long event rather than a string of
        fragments, and signal queues are excluded via :meth:`_queued_at_signal`.
        """
        label = "stopped_vehicle"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return []
        window = float(cfg.get("window", 2.5))
        net_max = float(cfg.get("net_max_px", 25.0))
        path_max = float(cfg.get("path_max_px", 250.0))
        duration = float(cfg.get("duration", 10.0))
        tolerance = float(cfg.get("tolerance_gap", 1.5))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            key = (label, track.track_id)
            if not TrackManager.is_vehicle(track) or not self._road_track(track, scene):
                self._condition_start.pop(key, None)
                continue
            stationary = is_stationary(
                track.bottom_history, window, net_max, path_max, state.timestamp
            )
            if stationary and self._queued_at_signal(track, state):
                stationary = False
            since = self._condition_start.get(key)
            if not stationary:
                # Tolerate a short excursion before tearing the timer down, but
                # never start a timer on a frame that is not stationary.
                if since is None or state.timestamp - since > tolerance:
                    self._condition_start.pop(key, None)
                    continue
                # Inside the tolerance window: keep the existing timer alive so
                # jitter does not fragment one long stop into many events.
                active_keys.add(key)
            else:
                if since is None:
                    since = self._condition_since(key, True, state.timestamp)
                active_keys.add(key)
            if since is None:
                continue
            held = state.timestamp - since
            if held >= duration:
                found.append(
                    RuleSignal(
                        label,
                        True,
                        confidence=0.9,
                        evidence={
                            "track_id": track.track_id,
                            "stopped_since": round(since, 3),
                            "held_sec": round(held, 2),
                            "class_name": track.class_name,
                            "queued_at_signal": False,
                        },
                        start_hint=since,
                    )
                )
        self._retain_conditions(label, active_keys)
        return found

    def _stopped_vehicle(self, state: FrameState) -> RuleSignal:
        found = self._stopped_vehicle_candidates(state)
        return found[0] if found else self._inactive("stopped_vehicle")

    # ------------------------------------------------------------------
    # red_light
    # ------------------------------------------------------------------
    def _red_light_candidates(self, state: FrameState) -> list[RuleSignal]:
        """A vehicle whose FRONT crossed a stop line while the signal was red.

        Official definition: the event starts when the front of the vehicle
        crosses the stop line, and ends when the vehicle leaves the
        intersection or the frame.

        The crossing is direction-aware (``front_crossing`` requires a positive
        dot product between the heading and the line's direction and an
        upstream-to-downstream transition), so a vehicle leaving the junction,
        reversing, or already past the line cannot trigger it. The signal is the
        one that *governs* the line, matched by declared approach direction; an
        unreadable head yields ``unknown`` and the rule stays silent rather than
        assuming red.
        """
        label = "red_light"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.stop_lines:
            return []
        window = float(cfg.get("window", 3.0))
        min_dot = float(cfg.get("approach_dot", 0.25))
        # How far past the line a vehicle is still "in" the junction. A vehicle
        # is considered to have left once it is well downstream of the crossing.
        found: list[RuleSignal] = []
        for track in state.tracks:
            if not TrackManager.is_vehicle(track):
                self._drop_crossings(track.track_id)
                continue
            lane = self._vehicle_lane(scene, track)
            for line in scene.config.stop_lines:
                key = ("red_light", track.track_id, line.line_id)
                if self._signal_color(scene, line) not in STOPPING_COLORS:
                    # Not red: forget any crossing so a later red phase reports
                    # a fresh event rather than reviving an old one.
                    self._crossings.pop(key, None)
                    self._condition_start.pop(key, None)
                    continue
                if lane is None:
                    continue
                crossed_at = front_crossing(
                    track, line, state.timestamp, window=window, min_dot=min_dot
                )
                if crossed_at is None:
                    continue
                if key not in self._crossings:
                    self._crossings[key] = crossed_at
                start = self._crossings[key]
                if state.timestamp - start > float(cfg.get("max_duration", 12.0)):
                    # A crossing that never resolved is dropped rather than
                    # reported as an event lasting the whole video.
                    self._crossings.pop(key, None)
                    continue
                colour = self._signal_color(scene, line)
                found.append(
                    RuleSignal(
                        label,
                        True,
                        confidence=0.9,
                        evidence={
                            "track_id": track.track_id,
                            "line_id": line.line_id,
                            "signal": scene.signal_for_line(line),
                            "signal_color": colour,
                            "cross_time": round(start, 3),
                            "front_along": round(self._front_along(track, line), 1),
                        },
                        start_hint=start,
                    )
                )
        self._condition_start = {
            key: value for key, value in self._condition_start.items() if key[0] != label
        }
        return found

    def _front_along(self, track: TrackState, line: Any) -> float:
        from .kinematics import front_path

        path = front_path(track, line.direction, 0.2, track.last_seen)
        if not path:
            return float("nan")
        point = (path[-1][1], path[-1][2])
        if line.direction is None:
            return float("nan")
        a = line.segment[0]
        return (point[0] - a[0]) * line.direction[0] + (point[1] - a[1]) * line.direction[1]

    def _drop_crossings(self, track_id: int) -> None:
        for key in [k for k in self._crossings if len(k) > 1 and k[1] == track_id]:
            del self._crossings[key]

    def _red_light(self, state: FrameState) -> RuleSignal:
        found = self._red_light_candidates(state)
        return found[0] if found else self._inactive("red_light")

    # ------------------------------------------------------------------
    # stop_line
    # ------------------------------------------------------------------
    def _stop_line_candidates(self, state: FrameState) -> list[RuleSignal]:
        """A vehicle stopped PAST the stop line on red, without entering the junction.

        Official definition: the event starts when the vehicle stops and ends
        when the signal turns green. This is deliberately different from
        ``red_light`` (which is about *crossing* on red) and from
        ``stopped_vehicle`` (which is about standing still anywhere on the
        carriageway for ten seconds or more).

        "Past the line" is measured on the vehicle's front, so a long vehicle
        overhangs the line while its body is still behind it. "Without entering
        the junction" is measured against the crossing the line serves.
        """
        label = "stop_line"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.stop_lines:
            return []
        window = float(cfg.get("window", 2.5))
        net_max = float(cfg.get("net_max_px", 25.0))
        path_max = float(cfg.get("path_max_px", 250.0))
        # How far past the line still counts as "at the line" rather than
        # "inside the junction".
        max_past = float(cfg.get("max_past_px", 120.0))
        min_red = float(cfg.get("min_red_sec", 0.4))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            key = (label, track.track_id)
            if not TrackManager.is_vehicle(track):
                self._condition_start.pop(key, None)
                continue
            progressed = False
            for line in scene.config.stop_lines:
                sub = (label, track.track_id, line.line_id)
                colour = self._signal_color(scene, line)
                if colour not in STOPPING_COLORS:
                    # Green ends the event, as the official definition requires.
                    self._condition_start.pop(sub, None)
                    continue
                along = self._front_along(track, line)
                if math.isnan(along) or not (0.0 <= along <= max_past):
                    self._condition_start.pop(sub, None)
                    continue
                if not is_stationary(track.bottom_history, window, net_max, path_max, state.timestamp):
                    self._condition_start.pop(sub, None)
                    continue
                # A vehicle that has reached the crossing has entered the
                # junction, which is red_light's business, not stop_line's.
                if any(point_in_polygon((track.center[0], track.center[1]), crossing)
                       for crossing in scene.config.crossings):
                    self._condition_start.pop(sub, None)
                    continue
                red_since = self._condition_start.get(("stop_line_red", line.line_id))
                if red_since is None or state.timestamp - red_since > 1.0:
                    self._condition_start[("stop_line_red", line.line_id)] = state.timestamp
                    red_since = state.timestamp
                if state.timestamp - red_since < min_red:
                    continue
                start = self._condition_since(sub, True, state.timestamp)
                if start is None:
                    continue
                active_keys.add(sub)
                progressed = True
                found.append(
                    RuleSignal(
                        label,
                        True,
                        confidence=0.85,
                        evidence={
                            "track_id": track.track_id,
                            "line_id": line.line_id,
                            "signal": scene.signal_for_line(line),
                            "signal_color": colour,
                            "stopped_since": round(start, 3),
                            "front_past_px": round(along, 1),
                        },
                        start_hint=start,
                    )
                )
            if not progressed:
                self._condition_start.pop(key, None)
        self._retain_conditions(label, active_keys)
        return found

    def _stop_line(self, state: FrameState) -> RuleSignal:
        found = self._stop_line_candidates(state)
        return found[0] if found else self._inactive("stop_line")

    # ------------------------------------------------------------------
    # congestion
    # ------------------------------------------------------------------
    def _direction_groups(self, scene: Any, state: FrameState) -> dict[tuple[float, ...], list[TrackState]]:
        """Road users grouped by the *direction of travel* of the lane they are in.

        The official definition is standstill or crawling traffic "across all
        lanes of a direction", so the unit of the event is a direction, not a
        single lane and not a single road user. Grouping by the quantised lane
        direction achieves that without needing an explicit direction id, and
        keeps the two carriageways of this two-way road strictly separate.
        """
        groups: dict[tuple[float, ...], list[TrackState]] = {}
        for track in state.tracks:
            if not TrackManager.is_vehicle(track) or not self._road_track(track, scene):
                continue
            lane = self._vehicle_lane(scene, track)
            if lane is None or lane.direction is None:
                continue
            unit = normalized_vector(lane.direction)
            if unit is None:
                continue
            key = (round(unit[0], 2), round(unit[1], 2))
            groups.setdefault(key, []).append((track, lane))  # type: ignore[arg-type]
        return groups  # type: ignore[return-value]

    def _congestion_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Standstill or crawling traffic across every lane of one direction.

        Deliberately stricter than "several stopped cars":

        * every lane of the direction must be occupied, because the definition
          says "across all lanes of a direction";
        * a signal queue is excluded, since standing at a red light is the
          designed behaviour of the road and not congestion;
        * the condition must persist for ``duration``, so a momentary gap
          between two green waves does not register.
        """
        label = "congestion"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return []
        speed = float(cfg.get("speed", 8.0))
        duration = float(cfg.get("duration", 5.0))
        min_share = float(cfg.get("lane_share", 1.0))
        window = float(cfg.get("window", 2.5))
        net_max = float(cfg.get("net_max_px", 40.0))
        path_max = float(cfg.get("path_max_px", 400.0))
        tolerance = float(cfg.get("tolerance_gap", 2.0))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for key, members in self._direction_groups(scene, state).items():
            lane_ids = {lane.lane_id for _, lane in members}
            slow: list[tuple[TrackState, Any]] = []
            for track, lane in members:
                moving = is_stationary(track.bottom_history, window, net_max, path_max, state.timestamp)
                if not moving and track.speed <= speed:
                    moving = True
                if moving:
                    slow.append((track, lane))
            if not lane_ids:
                continue
            occupied = {lane.lane_id for track, lane in slow}
            share = len(occupied) / len(lane_ids)
            condition = share >= min_share and len(slow) >= int(cfg.get("min_vehicles", 3))
            if condition:
                # A standing queue at a red light is not congestion.
                if all(self._in_signal_queue(track, scene, state) for track, _ in slow):
                    condition = False
            sub = (label, key)
            since = self._condition_start.get(sub)
            if not condition:
                if since is None or state.timestamp - since > tolerance:
                    self._condition_start.pop(sub, None)
                continue
            if since is None:
                since = self._condition_since(sub, True, state.timestamp)
                active_keys.add(sub)
                continue
            active_keys.add(sub)
            if state.timestamp - since >= duration:
                found.append(
                    RuleSignal(
                        label,
                        True,
                        confidence=min(1.0, 0.5 + 0.5 * share),
                        evidence={
                            "direction": list(key),
                            "lanes": sorted(lane_ids),
                            "lanes_stopped": sorted(occupied),
                            "vehicles": len(slow),
                            "since": round(since, 3),
                        },
                        start_hint=since,
                    )
                )
        self._retain_conditions(label, active_keys)
        return found

    def _in_signal_queue(self, track: TrackState, scene: Any, state: FrameState) -> bool:
        line = scene.line_for_track(track)
        if line is None:
            return False
        mine = self._upstream_distance(track.bottom_center, line)
        if mine is None or math.isnan(mine) or mine >= 0.0:
            return False
        return self._signal_color(scene, line) in STOPPING_COLORS

    def _congestion(self, state: FrameState) -> RuleSignal:
        found = self._congestion_candidates(state)
        return found[0] if found else self._inactive("congestion")

    # ------------------------------------------------------------------
    # jaywalking
    # ------------------------------------------------------------------
    def _jaywalking_candidates(self, state: FrameState) -> list[RuleSignal]:
        """A pedestrian on the carriageway outside a crossing.

        The official definition is about *entering the carriageway*, so three
        exclusions matter and each is enforced explicitly:

        * a pedestrian inside a calibrated crossing is crossing legally;
        * a pedestrian not on the carriageway is on a pavement, median or island,
          which is not road at all;
        * a pedestrian who only clips the road edge for a moment is filtered by
          ``min_duration``.

        The start is the moment the pedestrian stepped onto the road, taken from
        the tracker's own history rather than from the frame the rule is first
        evaluated on.
        """
        label = "jaywalking"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_road", False):
            return []
        min_duration = float(cfg.get("duration", 1.0))
        window = float(cfg.get("window", 2.5))
        net_max = float(cfg.get("net_max_px", 20.0))
        path_max = float(cfg.get("path_max_px", 400.0))
        # A pedestrian must be moving: someone standing on a traffic island is
        # not jaywalking, and one standing in the road after an accident is not
        # jaywalking either.
        require_motion = bool(cfg.get("require_motion", True))
        min_speed = float(cfg.get("min_speed", 4.0))
        # How far outside a crossing a pedestrian may be and still count. A
        # crossing polygon is wider than the paint, so a small margin absorbs
        # detection jitter without admitting the footway beside it.
        crossing_margin = float(cfg.get("crossing_margin_px", 40.0))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            key = (label, track.track_id)
            if not TrackManager.is_person(track):
                self._condition_start.pop(key, None)
                continue
            # A pedestrian counts as being on the carriageway only when inside a
            # *lane* polygon. The road polygons include the verges and footways
            # around this junction, and using them put 53 percent of all
            # pedestrian samples "on the road".
            try:
                on_carriageway = scene.is_road_point(track.bottom_center, carriageway_only=True)
            except TypeError:
                on_carriageway = self._road_track(track, scene)
            if not on_carriageway:
                self._condition_start.pop(key, None)
                continue
            in_crossing = (
                scene.crossing_for_point(track.bottom_center) is not None
                or self._crossing_near(track, scene, crossing_margin) is not None
            )
            if in_crossing:
                self._condition_start.pop(key, None)
                continue
            if require_motion and track.speed < min_speed:
                # Standing still on the carriageway is not "entering" it.
                self._condition_start.pop(key, None)
                continue
            start = self._condition_since(key, True, state.timestamp)
            if start is None:
                continue
            active_keys.add(key)
            if state.timestamp - start < min_duration:
                continue
            # Prefer the timestamp at which the pedestrian first reached the
            # road, which is the official start of the event.
            entry = self._road_entry_time(track, scene)
            found.append(
                RuleSignal(
                    label,
                    True,
                    confidence=0.8,
                    evidence={
                        "track_id": track.track_id,
                        "on_road_since": round(entry, 3),
                        "duration": round(state.timestamp - entry, 2),
                        "bottom_center": [round(v, 1) for v in track.bottom_center],
                    },
                    start_hint=entry,
                )
            )
        self._retain_conditions(label, active_keys)
        return found

    def _road_entry_time(self, track: TrackState, scene: Any) -> float:
        """Most recent time the pedestrian's bottom centre was off the carriageway.

        The start of a jaywalking event is when the pedestrian steps onto the
        carriageway, so the timer has to be anchored to that transition rather
        than to the frame the rule happens to notice. The same
        ``carriageway_only`` test is used, so the anchor and the trigger cannot
        disagree about what counts as road.
        """
        points = [p for p in track.bottom_history if p[0] <= track.last_seen + 1e-6]
        if not points:
            return track.first_seen
        try:
            off_road_at: float | None = None
            for timestamp, x, y in reversed(points):
                if not scene.is_road_point((x, y), carriageway_only=True):
                    off_road_at = timestamp
                else:
                    break
        except TypeError:
            # A scene without the carriageway_only keyword: fall back to the
            # plain road test rather than anchoring the event to first sighting.
            off_road_at = None
            for timestamp, x, y in reversed(points):
                if not scene.is_road_point((x, y)):
                    off_road_at = timestamp
                else:
                    break
        if off_road_at is None:
            return track.first_seen
        # One frame later is the first sample that is on the road.
        return min(track.last_seen, off_road_at + 1.0 / 30.0)

    def _jaywalking(self, state: FrameState) -> RuleSignal:
        found = self._jaywalking_candidates(state)
        return found[0] if found else self._inactive("jaywalking")

    # ------------------------------------------------------------------
    # failure_to_yield
    # ------------------------------------------------------------------
    def _failure_to_yield_candidates(self, state: FrameState) -> list[RuleSignal]:
        """A vehicle proceeds through a crossing while a pedestrian is on it.

        The official definition needs an interaction, so proximity alone is
        never enough. All of the following must hold on the same frame:

        * the pedestrian is inside a calibrated crossing, or stepping onto one;
        * the vehicle is approaching that same crossing, and its front passes
          through it while still moving;
        * the vehicle does not stop while the pedestrian occupies the crossing.

        A vehicle that halts for a pedestrian is compliant, and a pedestrian on
        the pavement who never reaches the crossing is not a conflict at all.
        """
        label = "failure_to_yield"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.crossings:
            return []
        min_speed = float(cfg.get("min_speed", 20.0))
        approach_px = float(cfg.get("approach_px", 420.0))
        window = float(cfg.get("window", 2.0))
        net_max = float(cfg.get("net_max_px", 20.0))
        path_max = float(cfg.get("path_max_px", 200.0))
        # A pedestrian is only "on" the crossing if inside it, or within a small
        # margin of stepping on. A larger kerb allowance swept in every pedestrian
        # standing on the footway beside the crossing, which at this camera is
        # most of them, and the rule then fired 842 times.
        kerb_px = float(cfg.get("kerb_px", 30.0))
        # How close the vehicle has to be to the pedestrian, measured in the road
        # frame. Proximity in the image is not proximity on the road: at this
        # viewing angle a pedestrian on the far pavement projects close to a
        # vehicle in the near lane.
        conflict_px = float(cfg.get("conflict_px", 60.0))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        people = [t for t in state.tracks if TrackManager.is_person(t)]
        vehicles = [t for t in state.tracks if TrackManager.is_vehicle(t)]
        for person in people:
            crossing = scene.crossing_for_point(person.bottom_center)
            if crossing is None:
                # On the kerb about to step in, so the event is not lost to a
                # frame of latency.
                crossing = self._crossing_near(person, scene, kerb_px)
            if crossing is None:
                continue
            for vehicle in vehicles:
                vehicle_crossing = scene.crossing_for_point(vehicle.center)
                if vehicle_crossing is None or vehicle_crossing != crossing:
                    continue
                if vehicle.speed < min_speed:
                    continue
                if distance_to_line(vehicle.bottom_center, _AnyLine(crossing)) > approach_px:
                    continue
                if swept_gap(vehicle, person) > conflict_px:
                    # Not actually near each other on the road.
                    continue
                # Yielded if the vehicle was stationary while the pedestrian was
                # on the crossing.
                halted = is_stationary(vehicle.bottom_history, window, net_max, path_max, state.timestamp)
                key = (label, person.track_id, vehicle.track_id)
                condition = not halted
                start = self._condition_since(key, condition, state.timestamp)
                if condition and start is not None:
                    active_keys.add(key)
                    found.append(
                        RuleSignal(
                            label,
                            True,
                            confidence=0.75,
                            evidence={
                                "person_id": person.track_id,
                                "vehicle_id": vehicle.track_id,
                                "vehicle_speed": round(vehicle.speed, 1),
                                "gap_px": round(swept_gap(vehicle, person), 1),
                                "crossing": crossing[0],
                            },
                            start_hint=start,
                        )
                    )
        self._retain_conditions(label, active_keys)
        return found

    def _crossing_near(self, track: TrackState, scene: Any, margin: float) -> list[list[float]] | None:
        """Crossing polygon within ``margin`` px of a point, if any."""
        best: tuple[float, list[list[float]]] | None = None
        for crossing in scene.config.crossings:
            distance = _polygon_distance((track.bottom_center), crossing)
            if distance <= margin and (best is None or distance < best[0]):
                best = (distance, crossing)
        return best[1] if best else None

    def _failure_to_yield(self, state: FrameState) -> RuleSignal:
        found = self._failure_to_yield_candidates(state)
        return found[0] if found else self._inactive("failure_to_yield")

    # ------------------------------------------------------------------
    # solid_line_crossing
    # ------------------------------------------------------------------
    def _solid_line_crossing_candidates(self, state: FrameState) -> list[RuleSignal]:
        """A manoeuvre across a calibrated solid marking.

        The official definition is a lane change or manoeuvre across a solid
        line, starting when the wheel crosses the line and ending when the
        vehicle is fully in the new lane. Three things are excluded explicitly:

        * a vehicle travelling ALONG a line, which drifts across it without
          changing sides - the test requires the sign of the side to flip;
        * a vehicle that clips the end of a marking, which is not a crossing of
          it, so the crossing must be over the drawn segment;
        * a turn through the junction, where solid lines legitimately end.
        """
        label = "solid_line_crossing"
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False) or not scene.config.solid_lines:
            return []
        window = float(cfg.get("window", 3.0))
        min_dot = float(cfg.get("crossing_dot", 0.2))
        hold = float(cfg.get("hold", 0.8))
        margin = float(cfg.get("edge_margin_px", 12.0))
        # The vehicle must be moving across the marking, not along it. The
        # calibrated markings are the road edges and the median kerb, so traffic
        # travels *parallel* to most of them; without this test a few pixels of
        # tracker drift across a kerb line reads as a crossing.
        cross_dot = float(cfg.get("perpendicular_dot", 0.45))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            if not TrackManager.is_vehicle(track):
                continue
            for line in scene.config.solid_lines:
                axis = _line_axis(line)
                heading = normalized_vector(track.velocity)
                if heading is None or axis is None:
                    continue
                along = abs(vector_dot(heading, axis))
                if along > (1.0 - cross_dot):
                    # Travelling along the marking, not across it.
                    continue
                path = _front_track(track, line, window, state.timestamp)
                if len(path) < 2:
                    continue
                before = _side_of(path[0], line)
                after = _side_of(path[-1], line)
                if before == 0 or after == 0 or before == after:
                    # Did not change sides, so no lane change happened.
                    continue
                crossing_at = _first_cross(path, line, margin)
                if crossing_at is None:
                    continue
                key = (label, track.track_id, line.line_id)
                if key in self._crossings:
                    continue
                self._crossings[key] = crossing_at
                since = self._condition_since(key, True, state.timestamp)
                if since is None:
                    continue
                if state.timestamp - crossing_at > hold:
                    self._crossings.pop(key, None)
                    self._condition_start.pop(key, None)
                    continue
                active_keys.add(key)
                found.append(
                    RuleSignal(
                        label,
                        True,
                        confidence=0.75,
                        evidence={
                            "track_id": track.track_id,
                            "line_id": line.line_id,
                            "cross_time": round(crossing_at, 3),
                            "from_side": before,
                            "to_side": after,
                        },
                        start_hint=crossing_at,
                    )
                )
        self._retain_conditions(label, active_keys)
        for key in [k for k in self._crossings if k[0] == label]:
            if key not in active_keys:
                self._crossings.pop(key, None)
        return found

    def _solid_line_crossing(self, state: FrameState) -> RuleSignal:
        found = self._solid_line_crossing_candidates(state)
        return found[0] if found else self._inactive("solid_line_crossing")

    # ------------------------------------------------------------------
    # turn rules
    # ------------------------------------------------------------------
    def _turn_candidates(self, state: FrameState, label: str) -> list[RuleSignal]:
        """Dispatch for the two turn rules, matching the candidate-method shape.

        The two rules share a signature so ``_candidate_methods`` can find them
        uniformly; without it the turn rules would silently degrade to a single
        flag or, worse, raise on every frame.
        """
        cfg = self._rule_config(label)
        scene = self._scene(state)
        if not getattr(scene, "has_geometry", False):
            return []
        if label == "illegal_u_turn":
            return self._u_turn_hits(state, cfg, scene)
        return self._illegal_turn_hits(state, cfg, scene)

    def _u_turn_hits(self, state: FrameState, cfg: dict[str, Any],
                     scene: Any) -> list[RuleSignal]:
        """A U-turn where the road markings prohibit it.

        Both halves of the official definition have to be observable, and at this
        camera only one of them is. What the geometry gives us:

        * the manoeuvre is recoverable from trajectory history, as a sustained
          heading reversal that returns towards the origin;
        * prohibition is implied by the road layout: the main carriageway is a
          two-way road with a kerbed median, so a U-turn across it is prohibited
          by the median itself. There is no "No U-turn" sign in view, so sign
          evidence is *not* claimed.

        A U-turn is only reported when the vehicle actually ends up travelling
        against the direction of the lane it started in, which is what
        distinguishes it from a vehicle that merely swings wide.
        """
        label = "illegal_u_turn"
        window = float(cfg.get("window", 6.0))
        # How recent the reversal must be: a U-turn is reported while the
        # vehicle is still executing or just after it turned around.
        current_window = float(cfg.get("current_window", 1.2))
        min_angle = float(cfg.get("angle", 2.35))
        min_speed = float(cfg.get("min_speed", 10.0))
        hold = float(cfg.get("hold", 0.8))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            key = (label, track.track_id)
            if not TrackManager.is_vehicle(track):
                self._condition_start.pop(key, None)
                continue
            lane = self._vehicle_lane(scene, track)
            if lane is None or lane.direction is None:
                self._condition_start.pop(key, None)
                continue
            reversal = heading_reversal(track, window, state.timestamp, min_angle=min_angle)
            if reversal is None:
                self._condition_start.pop(key, None)
                continue
            angle, duration, rate = reversal
            if rate < min_speed:
                self._condition_start.pop(key, None)
                continue
            # A U-turn is prohibited here because the carriageway is divided by
            # a kerbed median; the manoeuvre has to end up against the flow.
            # The *current* heading is read over a short window, not over the
            # whole manoeuvre: a completed U-turn returns near its origin, so
            # the net heading across the manoeuvre is degenerate and says
            # nothing about which way the vehicle is now travelling.
            unit = normalized_vector(lane.direction)
            heading = net_heading(track, current_window, state.timestamp)
            if unit is None or heading is None:
                self._condition_start.pop(key, None)
                continue
            now_against = vector_dot(unit, heading) < 0.0
            if not now_against:
                self._condition_start.pop(key, None)
                continue
            start = self._condition_since(key, True, state.timestamp)
            if start is None:
                continue
            active_keys.add(key)
            if state.timestamp - start > hold:
                self._condition_start.pop(key, None)
                continue
            found.append(
                RuleSignal(
                    label,
                    True,
                    confidence=0.7,
                    evidence={
                        "track_id": track.track_id,
                        "lane_id": lane.lane_id,
                        "turn_angle": round(angle, 3),
                        "duration": round(duration, 2),
                        "prohibited_by": "kerbed median (divided carriageway)",
                    },
                    start_hint=start,
                )
            )
        self._retain_conditions(label, active_keys)
        return found

    def _illegal_turn_hits(self, state: FrameState, cfg: dict[str, Any],
                           scene: Any) -> list[RuleSignal]:
        """A turn from the wrong lane, or in a direction the lane does not allow.

        Legality is read from the lane's declared ``allowed_moves`` and from the
        geometry, never from the mere fact that a vehicle turned:

        * a turn inside the junction, where the vehicle leaves the calibrated
          lanes entirely, is not judged - there is no lane to be wrong about;
        * a vehicle that changes lane while continuing to follow its lane
          direction is a lane change, not a turn, and is not this rule;
        * only a heading deviation beyond ``min_angle`` counts, and only when
          the movement is absent from the destination lane's ``allowed_moves``.
        """
        label = "illegal_turn"
        min_angle = float(cfg.get("min_angle", 0.6))
        window = float(cfg.get("window", 3.0))
        hold = float(cfg.get("hold", 0.8))
        min_speed = float(cfg.get("min_speed", 15.0))
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        for track in state.tracks:
            key = (label, track.track_id)
            if not TrackManager.is_vehicle(track):
                self._condition_start.pop(key, None)
                continue
            if len(track.lane_history) < 2 or track.lane_id is None:
                self._condition_start.pop(key, None)
                continue
            previous_id, current_id = track.lane_history[-2], track.lane_id
            if previous_id == current_id:
                self._condition_start.pop(key, None)
                continue
            lane = next((l for l in scene.config.lanes if l.lane_id == current_id), None)
            if lane is None or lane.direction is None:
                self._condition_start.pop(key, None)
                continue
            unit = normalized_vector(lane.direction)
            heading = net_heading(track, window, state.timestamp)
            if unit is None or heading is None or track.speed < min_speed:
                self._condition_start.pop(key, None)
                continue
            angle = angle_between(unit, heading)
            if angle < min_angle:
                # Still following its lane: a lane change, not a turn.
                self._condition_start.pop(key, None)
                continue
            allowed = {str(m).lower().replace("-", "_") for m in lane.allowed_moves}
            movement = _movement_name(unit, heading)
            if movement in allowed:
                self._condition_start.pop(key, None)
                continue
            start = self._condition_since(key, True, state.timestamp)
            if start is None:
                continue
            active_keys.add(key)
            if state.timestamp - start > hold:
                self._condition_start.pop(key, None)
                continue
            found.append(
                RuleSignal(
                    label,
                    True,
                    confidence=0.65,
                    evidence={
                        "track_id": track.track_id,
                        "from_lane": previous_id,
                        "to_lane": current_id,
                        "angle": round(angle, 3),
                        "movement": movement,
                        "allowed_moves": sorted(allowed),
                    },
                    start_hint=start,
                )
            )
        self._retain_conditions(label, active_keys)
        return found

    def _illegal_turn_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Candidate entry point for ``illegal_turn`` (see _turn_candidates)."""
        return self._turn_candidates(state, "illegal_turn")

    def _illegal_turn(self, state: FrameState) -> RuleSignal:
        found = self._turn_candidates(state, "illegal_turn")
        return found[0] if found else self._inactive("illegal_turn")

    def _illegal_u_turn_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Candidate entry point for ``illegal_u_turn`` (see _turn_candidates)."""
        return self._turn_candidates(state, "illegal_u_turn")

    def _illegal_u_turn(self, state: FrameState) -> RuleSignal:
        found = self._turn_candidates(state, "illegal_u_turn")
        return found[0] if found else self._inactive("illegal_u_turn")

    # ------------------------------------------------------------------
    # accident / near_miss
    # ------------------------------------------------------------------
    def _interaction_candidates(self, state: FrameState, label: str) -> list[RuleSignal]:
        """Collision and near-miss candidates.

        Thresholds come from measured percentiles over 286 639 real track pairs
        (``scripts/measure_rule_metrics.py`` on data_video2): the median
        inter-vehicle gap is 518 px and p90 is 1325 px, and box IoU is 0.00 at
        p90 and 0.12 at p99. A "close" pair at 140 px is therefore already deep
        in the tail, and an overlap threshold has to sit above p99 or ordinary
        queueing trips it.

        Three further conditions do most of the work:

        * the pair must share a lane, or one member must be a pedestrian. Two
          vehicles in different lanes of a wide junction pass close constantly
          and that is not an interaction;
        * an evasive response is required. Without one, only substantial box
          overlap counts, because a small gap with no reaction is two vehicles
          passing close - the archetypal false positive here, and exactly what
          a standing queue looks like;
        * the condition must persist for ``hold`` seconds, so a single frame of
          close boxes is treated as measurement noise.
        """
        cfg = self._rule_config(label)
        scene = self._scene(state)
        active_keys: set[tuple[Any, ...]] = set()
        found: list[RuleSignal] = []
        if label == "near_miss":
            ttc_max = float(cfg.get("ttc", 2.5))
            gap_max = float(cfg.get("gap_px", 140.0))
            decel_min = float(cfg.get("decel", 300.0))
            swerve_min = float(cfg.get("swerve", 0.9))
        else:
            gap_max = float(cfg.get("gap_px", 20.0))
            iou_min = float(cfg.get("iou", 0.30))
            decel_min = float(cfg.get("decel", 450.0))
            swerve_min = float(cfg.get("swerve", 1.1))
            # Without an evasive response, demand this much overlap to call it.
            calm_iou = float(cfg.get("calm_iou", 0.55))
        hold = float(cfg.get("hold", 1.0))
        require_shared_lane = bool(cfg.get("require_shared_lane", True))
        # A conflict requires real motion: a standing queue has a gap of zero
        # and no relative velocity, so without these gates the nearest pair in
        # every queue reads as a near miss.
        min_speed = float(cfg.get("min_speed", 25.0))
        min_relative = float(cfg.get("min_relative_speed", 40.0))
        # Contact thresholds in the ROAD frame, in pixels. A queue separates
        # along the road by roughly a car length, so `contact_along` has to sit
        # well under one; `contact_across` allows for lateral wobble.
        contact_along = float(cfg.get("contact_along_px", 45.0))
        contact_across = float(cfg.get("contact_across_px", 70.0))
        # For near_miss the pair must be closer than a car length along the
        # road; beyond that it is ordinary following distance in a queue.
        queue_along = float(cfg.get("queue_along_px", 70.0))

        road_users = [t for t in state.tracks if TrackManager.is_road_user(t)]
        for i, first in enumerate(road_users):
            for second in road_users[i + 1:]:
                if not (TrackManager.is_vehicle(first) or TrackManager.is_person(first)):
                    continue
                if not (TrackManager.is_vehicle(second) or TrackManager.is_person(second)):
                    continue
                gap = swept_gap(first, second)
                ttc = time_to_collision(first, second)
                closing = closing_speed(first, second)
                decel = max(deceleration(first), deceleration(second))
                swerve = _swerve(first, second, state.timestamp)
                overlap = bbox_iou(first.bbox, second.bbox)
                involves_person = TrackManager.is_person(first) or TrackManager.is_person(second)
                shared = self._shared_lane(scene, first, second)
                # A standing queue has a gap of zero and no relative motion, so
                # without this the nearest pair in every queue is a "near miss".
                # A conflict needs somebody actually moving.
                moving = max(first.speed, second.speed) >= min_speed
                relative = abs(TrackManager.relative_speed(first, second))
                contact = False

                if label == "near_miss":
                    # Defined as one road user evading another. At this camera
                    # that is two vehicles: a pedestrian near a car is a yield
                    # conflict, which failure_to_yield judges from the crossing
                    # geometry, and pedestrians walking past each other produce
                    # hundreds of pairs with a finite TTC and no conflict at
                    # all. Counting those here would charge one class for
                    # ordinary pavement activity.
                    if not (TrackManager.is_vehicle(first) and TrackManager.is_vehicle(second)):
                        continue
                    if gap > gap_max:
                        continue
                    if require_shared_lane and not shared:
                        continue
                    if not moving or relative < min_relative:
                        continue
                    if ttc is None or ttc > ttc_max:
                        continue
                    if not (decel >= decel_min or swerve >= swerve_min):
                        # A near miss is defined by the evasive action itself, so
                        # close-but-calm traffic is excluded by construction.
                        continue
                    # A standing or crawling queue is also "close with a finite
                    # TTC", because smoothed velocities never reach exactly zero.
                    # What separates it is that a queue is separated ALONG the
                    # road by roughly a car length, so the same road-frame test
                    # used for accident is applied here.
                    if shared:
                        lane = scene.lane_for_point(first.bottom_center)
                        if lane is not None and lane.direction is not None:
                            along_gap, across_gap = road_frame_gap(first, second, lane.direction)
                            if along_gap > queue_along:
                                # Following distance, not a conflict.
                                continue
                    else:
                        continue
                else:
                    # Accident is a collision between road users. At this camera
                    # a vehicle and a pedestrian standing near each other is a
                    # yield conflict, which failure_to_yield judges properly from
                    # the crossing geometry; counting it here too would charge
                    # one real event to two classes, which the metric punishes
                    # twice. So this class requires two vehicles.
                    if not (TrackManager.is_vehicle(first) and TrackManager.is_vehicle(second)):
                        continue
                    if require_shared_lane and not shared:
                        continue
                    # Image overlap is not contact evidence at this camera: the
                    # view is oblique, so the lead vehicle's box covers the one
                    # behind it and a queue reads as a pile-up. Contact is
                    # therefore judged on the road plane, and without a shared
                    # lane there is no road frame to measure in, so the pair is
                    # rejected rather than judged on geometry known to mislead.
                    contact = False
                    if shared:
                        lane = scene.lane_for_point(first.bottom_center)
                        if lane is not None and lane.direction is not None:
                            along_gap, across_gap = road_frame_gap(first, second, lane.direction)
                            contact = along_gap <= contact_along and across_gap <= contact_across
                    if not contact:
                        continue
                    # The crash signature is a TRANSITION from motion to rest.
                    # The official end condition is "all involved objects stop
                    # moving or leave the frame", so the pair must be at rest
                    # now - but they must have been moving before, and either
                    # one coming to rest is enough, because in a rear-end
                    # collision only the striking vehicle stops abruptly.
                    #
                    # Both halves are load-bearing, and each was added because
                    # the other alone admitted a false positive:
                    #   * "at rest now" alone admits every pair in a standing
                    #     queue, which are in contact and stationary from the
                    #     start;
                    #   * "was moving" alone admits a fast pass-through, where
                    #     the tracker carries the boxes through each other and
                    #     the overlap and swerve tests both look like contact.
                    window = float(cfg.get("stop_window", 2.0))
                    net_max = float(cfg.get("stop_net_px", 30.0))
                    path_max = float(cfg.get("stop_path_px", 200.0))
                    approach = float(cfg.get("approach_window", 4.0))
                    at_rest = (
                        is_stationary(first.bottom_history, window, net_max, path_max, state.timestamp)
                        and is_stationary(second.bottom_history, window, net_max, path_max, state.timestamp)
                    )
                    if not at_rest:
                        continue
                    was_moving = (
                        not is_stationary(first.bottom_history, approach, net_max, path_max, state.timestamp)
                        or not is_stationary(second.bottom_history, approach, net_max, path_max, state.timestamp)
                    )
                    if not was_moving:
                        continue
                    halted = True
                key = (label, first.track_id, second.track_id)
                start = self._condition_since(key, True, state.timestamp)
                if start is None:
                    continue
                if state.timestamp - start > hold:
                    self._condition_start.pop(key, None)
                    continue
                active_keys.add(key)
                found.append(
                    RuleSignal(
                        label,
                        True,
                        confidence=0.7 if label == "accident" else 0.6,
                        evidence={
                            "first_id": first.track_id,
                            "second_id": second.track_id,
                            "gap_px": round(gap, 1),
                            "ttc": None if ttc is None else round(ttc, 2),
                            "closing_px_s": round(closing, 1) if math.isfinite(closing) else None,
                            "decel": round(decel, 1),
                            "swerve": round(swerve, 3),
                            "iou": round(overlap, 3),
                            "shared_lane": shared,
                            "road_contact": bool(contact),
                            "post_impact_stop": bool(halted) if label == "accident" else None,
                        },
                        start_hint=start,
                    )
                )
        self._retain_conditions(label, active_keys)
        return found

    @staticmethod
    def _shared_lane(scene: Any, first: TrackState, second: TrackState) -> bool:
        """Do two road users occupy the same calibrated lane?

        A vehicle in one lane and a vehicle in the adjacent lane of the same
        carriageway pass within a car width of each other constantly; that is
        normal traffic, not an interaction.
        """
        lane_a = scene.lane_for_point(first.bottom_center)
        lane_b = scene.lane_for_point(second.bottom_center)
        if lane_a is None or lane_b is None:
            return False
        return lane_a.lane_id == lane_b.lane_id

    def _near_miss_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Candidate entry point for ``near_miss``.

        Without this, ``collect_signals`` would fall back to the single-signal
        path and two simultaneous near misses would collapse into one segment.
        """
        return self._interaction_candidates(state, "near_miss")

    def _accident_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Candidate entry point for ``accident`` (see _near_miss_candidates)."""
        return self._interaction_candidates(state, "accident")

    def _near_miss(self, state: FrameState) -> RuleSignal:
        found = self._near_miss_candidates(state)
        return found[0] if found else self._inactive("near_miss")

    def _accident(self, state: FrameState) -> RuleSignal:
        found = self._accident_candidates(state)
        return found[0] if found else self._inactive("accident")

    # ------------------------------------------------------------------
    # appearance rules
    # ------------------------------------------------------------------
    def _special_visual(self, state: FrameState, label: str) -> RuleSignal:
        found = self._special_candidates(state, label)
        return found[0] if found else self._inactive(label)

    OBSTACLE_WORDS = ("obstacle", "debris", "animal", "cone", "barrier", "fallen",
                      "roadwork", "pothole", "branch", "wheel", "box", "crate")
    FIRE_WORDS = ("fire", "smoke", "flame", "burning")

    def _special_candidates(self, state: FrameState, label: str) -> list[RuleSignal]:
        """``road_obstacle`` / ``fire_smoke`` from detector classes, if present.

        Both classes depend on the detector exposing the relevant categories.
        The competition checkpoint is trained on the eight road-user classes
        (person, bicycle, car, motorcycle, bus, truck, traffic light, stop
        sign) and emits nothing that maps to an obstacle or to fire, so with
        that checkpoint these rules correctly stay silent rather than guessing.

        The discrimination that matters when a model *does* expose such classes
        is persistence plus scene context: an obstacle is something that stays
        put on the carriageway, and neither a moving road user nor a shadow is
        an obstacle.
        """
        cfg = self._rule_config(label)
        scene = self._scene(state)
        words = self.OBSTACLE_WORDS if label == "road_obstacle" else self.FIRE_WORDS
        duration = float(cfg.get("duration", 0.5))
        max_speed = float(cfg.get("max_speed", 6.0))
        found: list[RuleSignal] = []
        active_keys: set[tuple[Any, ...]] = set()
        for track in state.tracks:
            if TrackManager.is_vehicle(track) or TrackManager.is_person(track):
                continue
            if not self._special(track, words):
                continue
            if getattr(scene, "has_road", False) and not self._road_track(track, scene):
                continue
            if label == "road_obstacle" and track.speed > max_speed:
                # A road obstacle does not drive away.
                continue
            key = (label, track.track_id)
            start = self._condition_since(key, True, state.timestamp)
            if start is None:
                continue
            active_keys.add(key)
            if state.timestamp - start < duration:
                continue
            found.append(
                RuleSignal(
                    label,
                    True,
                    confidence=0.6,
                    evidence={
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "since": round(start, 3),
                    },
                    start_hint=start,
                )
            )
        self._retain_conditions(label, active_keys)
        return found

    def _road_obstacle_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Candidate entry point for ``road_obstacle``.

        A road can hold several obstacles at once, and each is a separate
        event; see _near_miss_candidates for why this wrapper matters.
        """
        return self._special_candidates(state, "road_obstacle")

    def _fire_smoke_candidates(self, state: FrameState) -> list[RuleSignal]:
        """Candidate entry point for ``fire_smoke`` (see _near_miss_candidates)."""
        return self._special_candidates(state, "fire_smoke")

    def _road_obstacle(self, state: FrameState) -> RuleSignal:
        return self._special_visual(state, "road_obstacle")

    def _fire_smoke(self, state: FrameState) -> RuleSignal:
        return self._special_visual(state, "fire_smoke")

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def _rule_table(self) -> dict[str, Callable[[FrameState], RuleSignal]]:
        return {
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

    def _candidate_methods(self) -> dict[str, Callable[[FrameState], list[RuleSignal]]]:
        """Per-rule candidate producers, i.e. the ``_<label>_candidates`` methods.

        Kept in one place so the dispatch table and the candidate table cannot
        drift apart, which would silently reduce a multi-instance rule to a
        single flag.
        """
        table: dict[str, Callable[[FrameState], list[RuleSignal]]] = {}
        for label in self._rule_table():
            method = getattr(self, f"_{label}_candidates", None)
            if method is not None:
                table[label] = method
        return table

    def collect_signals(self, state: FrameState) -> dict[str, list[RuleSignal]]:
        """Every active signal for every rule, one entry per distinct object.

        This is what the temporal segmenter consumes, so several simultaneous
        instances of one class produce several segments instead of one.
        """
        out: dict[str, list[RuleSignal]] = {}
        singles = self._rule_table()
        candidates = self._candidate_methods()
        for label, method in singles.items():
            if not self._enabled(label):
                out[label] = []
                continue
            try:
                producer = candidates.get(label)
                if producer is not None:
                    out[label] = list(producer(state))
                else:
                    signal = method(state)
                    out[label] = [signal] if signal.active else []
            except Exception as exc:  # a broken rule must not kill the frame
                warnings.warn(f"rule {label} failed on frame {state.frame_id}: {exc}")
                out[label] = []
        self._last_evidence = {
            label: signals[0].evidence for label, signals in out.items() if signals
        }
        return out

    def evaluate(self, state: FrameState) -> dict[str, RuleSignal]:
        """One representative signal per rule, for the frame-level API."""
        collected = self.collect_signals(state)
        result: dict[str, RuleSignal] = {}
        for label in OFFICIAL_LABELS:
            signals = collected.get(label) or []
            result[label] = max(signals, key=lambda s: s.confidence) if signals else self._inactive(label)
        return result


# ----------------------------------------------------------------------
# module-level geometry helpers
# ----------------------------------------------------------------------

class _AnyLine:
    """Adapts a polygon to the line interface used by ``distance_to_line``."""

    __slots__ = ("polygon", "direction", "segment")

    def __init__(self, polygon: list[list[float]]):
        self.polygon = polygon
        self.segment = (tuple(polygon[0]), tuple(polygon[-1]))
        self.direction = None


def _polygon_distance(point: tuple[float, float], polygon: list[list[float]]) -> float:
    best = float("inf")
    for i in range(len(polygon)):
        a = (float(polygon[i][0]), float(polygon[i][1]))
        b = (float(polygon[(i + 1) % len(polygon)][0]), float(polygon[(i + 1) % len(polygon)][1]))
        best = min(best, _point_segment_distance(point, a, b))
    return best


def _point_segment_distance(point: tuple[float, float], a, b) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _line_axis(line: Any) -> tuple[float, float] | None:
    """Unit vector along the drawn segment."""
    (ax, ay), (bx, by) = line.segment
    return normalized_vector((bx - ax, by - ay))


def _front_track(track: TrackState, line: Any, window: float, timestamp: float):
    from .kinematics import front_path

    direction = line.direction
    if direction is None:
        # A solid marking with no declared direction: fall back to the lane the
        # vehicle is in, which is the direction it is legitimately travelling.
        return [p for p in track.bottom_history
                if p[0] > timestamp - window and p[0] <= timestamp + 1e-6]
    return front_path(track, direction, window, timestamp)


def _side_of(point3: tuple[float, float, float], line: Any) -> int:
    """+1 / -1 for the two sides of the line, 0 when on it."""
    (ax, ay), (bx, by) = line.segment
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-9:
        return 0
    ux, uy = (bx - ax) / length, (by - ay) / length
    value = (point3[1] - ax) * (-uy) + (point3[2] - ay) * ux
    if abs(value) < 1e-6:
        return 0
    return 1 if value > 0 else -1


def _first_cross(path, line: Any, margin: float) -> float | None:
    """Timestamp of the first step whose segment crosses the drawn line."""
    from ..scene.geometry import segments_intersect

    for i in range(1, len(path)):
        a = (path[i - 1][1], path[i - 1][2])
        b = (path[i][1], path[i][2])
        if not segments_intersect(a, b, line.segment[0], line.segment[1]):
            continue
        # Require the crossing to be over the drawn extent, not past its end.
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        if not (-margin <= across_line(mid, line) <= line_length(line) + margin):
            continue
        return path[i - 1][0]
    return None


def _swerve(first: TrackState, second: TrackState, timestamp: float,
            window: float = 1.5) -> float:
    """Largest *sustained* heading change either track made over the window.

    Measured over a whole track's history this is a poor discriminator: 90% of
    real traffic at this junction turns through the box, so a window-long heading
    change is large for almost every pair. What distinguishes an evasive
    swerve is that it happens *quickly* and the vehicle then holds the new
    heading, so the direction is measured over a short baseline and the change
    is required to persist.
    """
    best = 0.0
    for track in (first, second):
        points = [p for p in track.bottom_history
                  if p[0] > timestamp - window and p[0] <= timestamp + 1e-6]
        if len(points) < 6:
            continue
        # Short baselines at both ends: how the vehicle was pointing, and how it
        # is pointing now.
        lead = normalized_vector((points[1][1] - points[0][1], points[1][2] - points[0][2]))
        tail = normalized_vector((points[-1][1] - points[-2][1], points[-1][2] - points[-2][2]))
        if lead is None or tail is None:
            continue
        change = angle_between(lead, tail)
        # Persistence: the last third of the window must agree with the final
        # heading, so a single jittery sample cannot register as a swerve.
        recent = points[-(len(points) // 3):]
        settled = normalized_vector((recent[-1][1] - recent[0][1],
                                     recent[-1][2] - recent[0][2]))
        if settled is None:
            continue
        hold = angle_between(tail, settled)
        if hold > 0.5:
            # Still turning: the manoeuvre has not resolved, so it is not yet an
            # evasive action, it is a turn through the junction.
            continue
        best = max(best, change)
    return best


def _movement_name(unit: tuple[float, float], heading: tuple[float, float]) -> str:
    """Name the manoeuvre a heading describes relative to a lane direction.

    Works in the lane's own frame: the component along the lane is "straight"
    when it dominates, otherwise the sign of the cross product says which way
    the vehicle turned.
    """
    along = vector_dot(unit, heading)
    cross = unit[0] * heading[1] - unit[1] * heading[0]
    if abs(cross) < 0.35 and along > 0:
        return "straight"
    return "right" if cross < 0 else "left"
