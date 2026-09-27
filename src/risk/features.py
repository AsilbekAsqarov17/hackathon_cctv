from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ..contracts import FrameState, RuleSignal
from ..tracking.track_manager import TrackManager

# Resolution the pixel thresholds below are expressed in. Matches the scene
# file's `reference_size`, so scene geometry and risk thresholds scale together.
REFERENCE_WIDTH = 3840.0


@dataclass
class RiskFeatures:
    """Causal features for one processed frame."""

    timestamp: float
    vehicle_count: int = 0
    person_count: int = 0
    close_pair_count: int = 0
    min_ttc: float | None = None
    # Minimum TTC restricted to pairs that are genuinely *conflicting*. This is
    # the quantity the score ramp is built on: a raw minimum over all pairs is
    # small in any queue, because thirty vehicles bumper to bumper always
    # contain some pair with a short time-to-collision, and a score that
    # saturates on ordinary traffic would alarm constantly.
    min_conflict_ttc: float | None = None
    conflict_pair_count: int = 0
    # True when a conflicting pair is also showing evasive behaviour (a sharp
    # deceleration or a sustained heading change). Car-following alone is not
    # risk: a follower 1.2 s behind a leader is normal traffic, because the
    # leader is going to brake. Measured on data_video1 at 4K, 9-19 pairs pass
    # the conflict test at any moment with a median TTC of 1.17 s, so the TTC
    # ramp on its own saturates. Requiring evasive behaviour reuses the
    # criteria the validated `near_miss` rule already uses.
    conflict_evasive: bool = False
    min_distance: float | None = None
    max_closing_speed: float = 0.0
    max_deceleration: float = 0.0
    max_acceleration: float = 0.0
    pedestrian_conflict: float = 0.0
    accident_candidate: bool = False
    near_miss_candidate: bool = False
    wrong_way_candidate: bool = False
    red_light_candidate: bool = False
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
    near_distance: float = 360.0,
    pedestrian_distance: float = 280.0,
    min_relative_speed: float = 280.0,
    conflict_along_px: float = 140.0,
    conflict_across_px: float = 140.0,
    evasive_decel: float = 1700.0,
    evasive_swerve: float = 0.9,
) -> RiskFeatures:
    """Extract only information available at ``state.timestamp``.

    No future track state, event segment, or video read is used here.

    The pixel arguments are in the 3840x2160 calibration resolution and are
    scaled to the actual frame; see ``REFERENCE_WIDTH`` and the note inside.
    """
    signals = signals or {}
    scene = getattr(state.scene, "context", None)
    # Every pixel threshold below is in the calibration resolution, not native
    # frame pixels. The same camera delivers 3840x2160 and 1920x1080, so a
    # fixed pixel distance is twice as permissive at 4K for the same physical
    # separation - which silently saturated the risk score on data_video1
    # (97.5% of frames above the alarm threshold) while the 1080p video looked
    # fine. Scaling against the authoring width makes the behaviour identical on
    # both, exactly as the scene geometry already does.
    width = float(getattr(state.scene, "width", 0) or 0)
    scale = (width / REFERENCE_WIDTH) if width > 0 else 1.0
    near_distance *= scale
    pedestrian_distance *= scale
    min_relative_speed *= scale
    conflict_along_px *= scale
    conflict_across_px *= scale
    evasive_decel *= scale

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
        red_light_candidate=bool(signals.get("red_light") and signals["red_light"].active),
        stopped_count=sum(1 for track in vehicles if track.speed <= 3.0),
    )
    distances: list[float] = []
    ttcs: list[float] = []
    conflict_ttcs: list[float] = []
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
                person = first if TrackManager.is_person(first) else second
                # The pedestrian has to be ON the carriageway. A pedestrian on
                # the footway projects close to a vehicle in the near lane at
                # this oblique viewing angle, and counting that put the score
                # above the alarm threshold on almost every frame of ordinary
                # pavement activity. This is the same distinction jaywalking
                # depends on, so it is reused rather than re-derived.
                if distance <= pedestrian_distance and _person_on_carriageway(scene, person):
                    features.pedestrian_conflict = max(
                        features.pedestrian_conflict,
                        1.0 - distance / max(pedestrian_distance, 1.0),
                    )
            # A pair is a conflict only if it occupies the same lane (or
            # involves a pedestrian) and is actually moving relative to the
            # other. Without both tests the minimum TTC is a property of the
            # traffic density rather than of any risk.
            if _is_conflict(scene, first, second, min_relative_speed,
                            conflict_along_px, conflict_across_px):
                features.conflict_pair_count += 1
                evasive = (
                    max(_deceleration(first), _deceleration(second)) >= evasive_decel
                    or _sustained_swerve(first, state.timestamp) >= evasive_swerve
                    or _sustained_swerve(second, state.timestamp) >= evasive_swerve
                )
                if evasive:
                    features.conflict_evasive = True
                    if ttc is not None and 0.0 < ttc <= 10.0:
                        conflict_ttcs.append(ttc)
    if distances:
        features.min_distance = min(distances)
    if ttcs:
        features.min_ttc = min(ttcs)
    if conflict_ttcs:
        features.min_conflict_ttc = min(conflict_ttcs)
    if road_users:
        features.max_deceleration = max(_deceleration(track) for track in road_users)
        features.max_acceleration = max(track.acceleration_magnitude for track in road_users)
    return features


def _sustained_swerve(track: Any, timestamp: float, window: float = 1.5) -> float:
    """Heading change over a short baseline, required to be holding.

    A window-long heading change is a poor discriminator here: 90% of traffic at
    this junction turns through the box, so almost every pair has a large value.
    What distinguishes an evasive swerve is that it happens quickly and the
    vehicle then holds the new heading. This is the same test the ``near_miss``
    rule uses, so Part A and Part B agree about what evasive means.
    """
    points = [p for p in track.bottom_history
              if p[0] > timestamp - window and p[0] <= timestamp + 1e-6]
    if len(points) < 6:
        return 0.0
    lead = _unit((points[1][1] - points[0][1], points[1][2] - points[0][2]))
    tail = _unit((points[-1][1] - points[-2][1], points[-1][2] - points[-2][2]))
    if lead is None or tail is None:
        return 0.0
    recent = points[-(len(points) // 3):]
    settled = _unit((recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2]))
    if settled is None:
        return 0.0
    hold = _angle(lead=tail, other=settled)
    if hold > 0.5:
        # Still turning: the manoeuvre has not resolved.
        return 0.0
    return _angle(lead=lead, other=tail)


def _unit(vector) -> tuple[float, float] | None:
    x, y = float(vector[0]), float(vector[1])
    length = math.hypot(x, y)
    if length < 1e-9:
        return None
    return (x / length, y / length)


def _angle(lead: tuple[float, float], other: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, lead[0] * other[0] + lead[1] * other[1]))
    return math.acos(dot)


def _person_on_carriageway(scene: Any, person: Any) -> bool:
    """Is this pedestrian standing on the carriageway rather than the footway?"""
    if scene is None or not getattr(scene, "config", None):
        return True
    try:
        return bool(scene.is_road_point(person.bottom_center, carriageway_only=True))
    except TypeError:
        return bool(scene.is_road_point(person.bottom_center))


def _is_conflict(scene: Any, first: Any, second: Any, min_relative_speed: float,
                 max_along_px: float = 70.0, max_across_px: float = 70.0) -> bool:
    """Could these two road users actually collide from here?

    Three conditions, each of which removes a measured false-positive source:

    * **real relative motion** - smoothed velocities in a standing queue never
      reach exactly zero, so a queue of creeping cars would otherwise qualify.
      The threshold is measured, not guessed: over 600 frames of data_video2 the
      relative speed of same-lane, road-frame-close vehicle pairs runs
      p50=57, p75=94, p90=127, p99=186 px/s, so 140 px/s sits near p95 and
      excludes the creep of an ordinary queue;
    * **a shared calibrated lane, or a pedestrian in the pair** - two vehicles
      on different approaches pass close constantly and that is not a conflict;
    * **closer than a car length in the road frame** - a queue is separated
      *along* the road by roughly one car length, so this is what separates a
      genuine conflict from ordinary following distance. It is the same test the
      ``near_miss`` rule uses, and reusing it keeps Part A and Part B consistent
      about what counts as a conflict.

    Measured on data_video2: with only the first two conditions, the minimum
    conflict TTC sat at 0.77 s in a stationary queue and the score read above
    the alarm threshold on 95 percent of frames.
    """
    relative = (
        second.velocity[0] - first.velocity[0],
        second.velocity[1] - first.velocity[1],
    )
    if math.hypot(*relative) < min_relative_speed:
        return False
    involves_person = TrackManager.is_person(first) or TrackManager.is_person(second)
    if scene is None or not getattr(scene, "config", None):
        return False
    lane_a = scene.lane_for_point(first.bottom_center)
    lane_b = scene.lane_for_point(second.bottom_center)
    if involves_person:
        axis = None
        if lane_a is not None and lane_b is not None and lane_a.lane_id == lane_b.lane_id:
            axis = lane_a.direction
    elif lane_a is None or lane_b is None or lane_a.lane_id != lane_b.lane_id:
        return False
    else:
        axis = lane_a.direction
    if axis is None or not any(axis):
        return True
    dx = second.bottom_center[0] - first.bottom_center[0]
    dy = second.bottom_center[1] - first.bottom_center[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return True
    ux, uy = dx / length, dy / length
    along = abs(dx * ux + dy * uy)
    across = abs(dx * (-uy) + dy * ux)
    if involves_person:
        # A pedestrian conflict is governed by the crossing geometry, so only
        # the along-road distance is gated here; lateral offset is the whole
        # point of a crossing conflict.
        return along <= max(max_along_px, max_across_px)
    return along <= max_along_px and across <= max_across_px
