from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ..contracts import FrameState, RuleSignal
from ..tracking.track_manager import TrackManager


@dataclass
class RiskFeatures:
    """Causal features for one processed frame."""

    timestamp: float
    vehicle_count: int = 0
    person_count: int = 0
    close_pair_count: int = 0
    min_ttc: float | None = None
    min_distance: float | None = None
    max_closing_speed: float = 0.0
    max_deceleration: float = 0.0
    max_acceleration: float = 0.0
    pedestrian_conflict: float = 0.0
    accident_candidate: bool = False
    near_miss_candidate: bool = False
    wrong_way_candidate: bool = False
    stopped_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _closing_speed(first: Any, second: Any) -> float:
    dx = second.center[0] - first.center[0]
    dy = second.center[1] - first.center[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return 0.0
    relative = (second.velocity[0] - first.velocity[0], second.velocity[1] - first.velocity[1])
    return max(0.0, -(dx * relative[0] + dy * relative[1]) / distance)


def _deceleration(track: Any) -> float:
    speed = track.speed
    if speed <= 1e-6:
        return 0.0
    return max(0.0, -(track.acceleration[0] * track.velocity[0] + track.acceleration[1] * track.velocity[1]) / speed)


def extract_risk_features(
    state: FrameState,
    signals: dict[str, RuleSignal] | None = None,
    near_distance: float = 180.0,
    pedestrian_distance: float = 140.0,
) -> RiskFeatures:
    """Extract only information available at ``state.timestamp``.

    No future track state, event segment, or video read is used here.
    """
    signals = signals or {}
    road_users = [track for track in state.tracks if TrackManager.is_road_user(track)]
    vehicles = [track for track in road_users if TrackManager.is_vehicle(track)]
    people = [track for track in road_users if TrackManager.is_person(track)]
    features = RiskFeatures(
        timestamp=float(state.timestamp),
        vehicle_count=len(vehicles),
        person_count=len(people),
        accident_candidate=bool(signals.get("accident") and signals["accident"].active),
        near_miss_candidate=bool(signals.get("near_miss") and signals["near_miss"].active),
        wrong_way_candidate=bool(signals.get("wrong_way") and signals["wrong_way"].active),
        stopped_count=sum(1 for track in vehicles if track.speed <= 3.0),
    )
    distances: list[float] = []
    ttcs: list[float] = []
    for index, first in enumerate(road_users):
        for second in road_users[index + 1 :]:
            distance = TrackManager.center_distance(first, second)
            if distance <= near_distance:
                features.close_pair_count += 1
                distances.append(distance)
            ttc = TrackManager.time_to_collision(first, second)
            if ttc is not None and 0.0 < ttc <= 10.0:
                ttcs.append(ttc)
            features.max_closing_speed = max(features.max_closing_speed, _closing_speed(first, second))
            if TrackManager.is_person(first) != TrackManager.is_person(second):
                if distance <= pedestrian_distance:
                    features.pedestrian_conflict = max(
                        features.pedestrian_conflict,
                        1.0 - distance / max(pedestrian_distance, 1.0),
                    )
    if distances:
        features.min_distance = min(distances)
    if ttcs:
        features.min_ttc = min(ttcs)
    if road_users:
        features.max_deceleration = max(_deceleration(track) for track in road_users)
        features.max_acceleration = max(track.acceleration_magnitude for track in road_users)
    return features
