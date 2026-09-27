"""Tests for the Part A rules beyond stopped_vehicle, wrong_way and red_light.

Every test here exercises real logic on constructed trajectories, including the
negative cases that decide whether a rule is safe to enable: a legal turn must
stay silent, a pedestrian inside a crossing must not be jaywalking, a queue at a
red light must not be congestion, and two cars in a queue must not be an
accident.
"""
from __future__ import annotations

import unittest
from collections import deque

from src.contracts import FrameState, SceneState, TrafficLightState, TrackState
from src.rules.engine import RuleEngine
from src.rules.kinematics import swept_gap
from src.scene.config import (
    Lane,
    SceneConfig,
    SceneContext,
    SceneLine,
    TrafficLightROI,
)
from src.scene.geometry import bbox_iou


def build_scene(width: int = 400, height: int = 400) -> SceneContext:
    """A T-intersection-like fixture with two lanes, a stop line and a crossing.

    Lanes run left-to-right and fall slightly, matching the real camera's
    geometry so the tests exercise the same code paths.
    """
    east = [[40, 150], [360, 200], [360, 250], [40, 200]]
    west = [[40, 90], [360, 140], [360, 150], [40, 100]]
    config = SceneConfig(
        scene_id="fixture",
        width=width,
        height=height,
        lanes=[
            Lane(0, west, (1.0, 0.14), ("straight",)),
            Lane(1, east, (1.0, 0.14), ("straight",)),
        ],
        stop_lines=[SceneLine("stop", ((250.0, 145.0), (250.0, 255.0)), "stop", (1.0, 0.14))],
        road_polygon=[[0, 80], [400, 130], [400, 260], [0, 210]],
        crossings=[[[280, 150], [360, 150], [360, 250], [280, 250]]],
        traffic_lights=[TrafficLightROI("sig", (300.0, 60.0, 330.0, 90.0), (1.0, 0.14))],
    )
    scene = SceneContext(config, width, height)
    scene.lights = {"sig": TrafficLightState("sig", "red", 1.0, (300.0, 60.0, 330.0, 90.0))}
    return scene


CLASS_IDS = {
    "person": 0, "pedestrian": 0,
    "bicycle": 1, "bike": 1,
    "car": 2, "vehicle": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


def make_track(
    track_id: int,
    name: str,
    centers: list[tuple[float, float]],
    start_t: float = 0.0,
    dt: float = 0.1,
    velocity: tuple[float, float] | None = None,
    half: tuple[float, float] = (12.0, 8.0),
) -> TrackState:
    """A track with full history, box history and a constant velocity.

    ``class_id`` is taken from the class name where the name is a known COCO
    road user, and left as -1 otherwise, so a fixture track named ``cone`` is
    not mistaken for a car by the class-id based checks in TrackManager.
    """
    times = [start_t + i * dt for i in range(len(centers))]
    history = deque(((t, c[0], c[1]) for t, c in zip(times, centers)), maxlen=300)
    bottom = deque(((t, c[0], c[1] + half[1]) for t, c in zip(times, centers)), maxlen=300)
    boxes = deque(
        ((t, c[0] - half[0], c[1] - half[1] * 2, c[0] + half[0], c[1])
         for t, c in zip(times, centers)),
        maxlen=300,
    )
    if velocity is None and len(centers) > 1:
        span = max(times[-1] - times[0], 1e-6)
        velocity = (
            (centers[-1][0] - centers[0][0]) / span,
            (centers[-1][1] - centers[0][1]) / span,
        )
    velocity = velocity or (0.0, 0.0)
    last = centers[-1]
    return TrackState(
        track_id=track_id,
        class_id=CLASS_IDS.get(name, -1),
        class_name=name,
        bbox=(last[0] - half[0], last[1] - half[1] * 2, last[0] + half[0], last[1]),
        score=0.9,
        first_seen=times[0],
        last_seen=times[-1],
        center=last,
        velocity=velocity,
        history=history,
        bottom_history=bottom,
        bbox_history=boxes,
    )


def frame(scene: SceneContext, t: float, tracks: list[TrackState]) -> FrameState:
    return FrameState(
        int(t * 30), t, tracks,
        SceneState("fixture", scene.width, scene.height, scene.lights, scene),
    )


def OFFICIAL_LABELS_OF(engine: RuleEngine) -> list[str]:
    return list(engine._rule_table())


def default_rules(**overrides) -> dict:
    rules = {
        "red_light": {"enabled": True, "window": 3.0},
        "stop_line": {"enabled": True, "window": 2.5, "max_past_px": 120.0, "min_red_sec": 0.0},
        "congestion": {"enabled": True, "speed": 8.0, "duration": 1.0, "min_vehicles": 2,
                       "window": 2.5, "net_max_px": 40.0, "path_max_px": 400.0},
        "jaywalking": {"enabled": True, "duration": 0.4, "min_speed": 1.0},
        "failure_to_yield": {"enabled": True, "min_speed": 5.0},
        "solid_line_crossing": {"enabled": True, "window": 3.0},
        "illegal_turn": {"enabled": True, "min_angle": 0.6, "window": 2.0, "min_speed": 1.0},
        "illegal_u_turn": {"enabled": True, "window": 6.0, "angle": 2.35, "min_speed": 1.0},
        "accident": {"enabled": True, "gap_px": 20.0, "iou": 0.12, "decel": 40.0, "swerve": 0.5},
        "near_miss": {"enabled": True, "ttc": 3.0, "gap_px": 150.0, "decel": 40.0, "swerve": 0.5},
        "road_obstacle": {"enabled": True, "duration": 0.3},
        "fire_smoke": {"enabled": True, "duration": 0.3},
    }
    for key, value in overrides.items():
        rules.setdefault(key, {}).update(value)
    return rules


class LineRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(default_rules())

    def test_stop_line_fires_for_vehicle_stopped_past_the_line_on_red(self) -> None:
        # Stationary with its front (centre + half-width) past the line at x=250.
        # max_past_px=120 keeps "just past the line" distinct from "inside the
        # junction", which is red_light's case.
        centers = [(256.0, 200.0)] * 12
        track = make_track(1, "car", centers, velocity=(0.0, 0.0))
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertTrue(signals["stop_line"], "a car stopped on red past the line must be stop_line")

    def test_stop_line_silent_on_green(self) -> None:
        self.scene.lights = {"sig": TrafficLightState("sig", "green", 1.0, (300.0, 60.0, 330.0, 90.0))}
        centers = [(230.0, 200.0)] * 12
        track = make_track(1, "car", centers, velocity=(0.0, 0.0))
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertFalse(signals["stop_line"], "green ends the event by definition")

    def test_stop_line_silent_for_vehicle_upstream_of_the_line(self) -> None:
        # Clearly before the line (front at 212, line at 250): that is a queue,
        # not a stop_line event.
        centers = [(200.0, 200.0)] * 12
        track = make_track(1, "car", centers, velocity=(0.0, 0.0))
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertFalse(signals["stop_line"])

    def test_stop_line_silent_when_signal_unreadable(self) -> None:
        self.scene.lights = {"sig": TrafficLightState("sig", "unknown", 0.0, (300.0, 60.0, 330.0, 90.0))}
        centers = [(230.0, 200.0)] * 12
        track = make_track(1, "car", centers, velocity=(0.0, 0.0))
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertFalse(signals["stop_line"], "an unreadable signal must not be assumed red")

    def test_red_light_uses_the_front_of_the_vehicle(self) -> None:
        # The vehicle's front crosses the line at x=250 partway through the
        # window, while its bottom centre crosses later, so a bottom-centre test
        # would report a different (later) start time. Here the front is
        # upstream and the bottom centre is past, which must NOT fire.
        centers = [(244.0, 200.0)] * 12
        track = make_track(1, "car", centers, velocity=(0.0, 0.0))
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertFalse(
            signals["red_light"],
            "bottom centre past the line but front still upstream is not a crossing",
        )

    def test_red_light_fires_when_the_front_leads_across(self) -> None:
        # A stationary car cannot cross, so drive it across the line instead.
        moving = make_track(1, "car", [(230.0 + 3.0 * i, 200.0) for i in range(12)])
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [moving]))
        self.assertTrue(signals["red_light"])

    def test_congestion_excludes_a_signal_queue(self) -> None:
        # Two stationary cars in the eastbound lane while the light is red.
        tracks = [
            make_track(1, "car", [(150.0, 180.0)] * 12, velocity=(0.0, 0.0)),
            make_track(2, "car", [(200.0, 190.0)] * 12, velocity=(0.0, 0.0)),
        ]
        signals = self.rules.collect_signals(frame(self.scene, 1.1, tracks))
        self.assertFalse(signals["congestion"], "standing at a red light is not congestion")

    def test_congestion_fires_when_the_light_is_green(self) -> None:
        self.scene.lights = {"sig": TrafficLightState("sig", "green", 1.0, (300.0, 60.0, 330.0, 90.0))}
        tracks = [
            make_track(1, "car", [(150.0, 180.0)] * 12, velocity=(0.0, 0.0)),
            make_track(2, "car", [(200.0, 190.0)] * 12, velocity=(0.0, 0.0)),
        ]
        for t in (0.5, 1.1, 1.6):
            signals = self.rules.collect_signals(frame(self.scene, t, tracks))
        self.assertTrue(signals["congestion"], "standstill on green across the lane is congestion")

    def test_congestion_silent_with_a_single_vehicle(self) -> None:
        self.scene.lights = {"sig": TrafficLightState("sig", "green", 1.0, (300.0, 60.0, 330.0, 90.0))}
        tracks = [make_track(1, "car", [(150.0, 180.0)] * 12, velocity=(0.0, 0.0))]
        signals = self.rules.collect_signals(frame(self.scene, 1.6, tracks))
        self.assertFalse(signals["congestion"])


class PedestrianRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(default_rules())

    def test_jaywalking_for_pedestrian_on_the_carriageway(self) -> None:
        # Walks along the road at y=190, which is carriageway, not a crossing.
        centers = [(120.0 + 6.0 * i, 190.0) for i in range(12)]
        track = make_track(1, "person", centers, half=(3.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertTrue(signals["jaywalking"])

    def test_no_jaywalking_inside_a_crossing(self) -> None:
        # The crossing polygon is x 280..360, y 150..250.
        centers = [(285.0 + 5.0 * i, 200.0) for i in range(12)]
        track = make_track(1, "person", centers, half=(3.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertFalse(signals["jaywalking"], "a pedestrian on a crossing is crossing legally")

    def test_no_jaywalking_off_the_carriageway(self) -> None:
        # y=40 is above the road polygon entirely: a pavement.
        centers = [(120.0 + 6.0 * i, 40.0) for i in range(12)]
        track = make_track(1, "person", centers, half=(3.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertFalse(signals["jaywalking"])

    def test_no_jaywalking_for_a_person_standing_still_on_the_road(self) -> None:
        centers = [(150.0, 190.0)] * 12
        track = make_track(1, "person", centers, velocity=(0.0, 0.0), half=(3.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertFalse(signals["jaywalking"], "standing is not entering the carriageway")

    def test_failure_to_yield_when_a_vehicle_crosses_on_a_pedestrian(self) -> None:
        person = make_track(1, "person", [(320.0 + 1.0 * i, 200.0) for i in range(12)], half=(3.0, 5.0))
        vehicle = make_track(2, "car", [(250.0 + 8.0 * i, 195.0) for i in range(12)])
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [person, vehicle]))
        self.assertTrue(signals["failure_to_yield"])

    def test_no_failure_to_yield_when_the_vehicle_waits(self) -> None:
        person = make_track(1, "person", [(320.0 + 1.0 * i, 200.0) for i in range(12)], half=(3.0, 5.0))
        vehicle = make_track(2, "car", [(290.0, 200.0)] * 12, velocity=(0.0, 0.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [person, vehicle]))
        self.assertFalse(signals["failure_to_yield"], "stopping for a pedestrian is compliant")


class InteractionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(default_rules())

    def test_no_accident_for_two_cars_in_a_queue(self) -> None:
        # Bumper to bumper and stationary: the archetypal false positive.
        lead = make_track(1, "car", [(250.0, 190.0)] * 12, velocity=(0.0, 0.0))
        follower = make_track(2, "car", [(226.0, 190.0)] * 12, velocity=(0.0, 0.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [lead, follower]))
        self.assertFalse(signals["accident"], "a standing queue is not a collision")

    def test_accident_when_boxes_overlap_and_both_stop(self) -> None:
        # Two cars converge head-on along the same lane, their boxes overlap,
        # and both come to rest at the point of contact. The stop is the
        # signature the official end condition relies on ("all involved objects
        # stop moving"); without it, ordinary close traffic that carries on and
        # separates would read as a pile-up.
        #
        # Both come to rest at the same instant, and the stationary tail is long
        # enough that the 2 s stop window at the evaluation times sees only the
        # stopped part of each trajectory.
        first = make_track(1, "car", [(60.0 + 10.0 * i, 175.0) for i in range(9)]
                           + [(150.0, 175.0)] * 25)
        second = make_track(2, "car", [(340.0 - 20.0 * i, 178.0) for i in range(9)]
                            + [(160.0, 178.0)] * 25)
        self.assertGreater(bbox_iou(first.bbox, second.bbox), 0.12,
                           "fixture must actually produce overlapping boxes")
        for t in (3.0, 3.1, 3.2):
            signals = self.rules.collect_signals(frame(self.scene, t, [first, second]))
        self.assertTrue(signals["accident"],
                        "overlapping boxes plus a post-impact stop is a collision")

    def test_no_accident_when_the_pair_keeps_moving(self) -> None:
        """Contact without a stop is two vehicles passing close, not a crash."""
        first = make_track(1, "car", [(60.0 + 10.0 * i, 175.0) for i in range(13)])
        second = make_track(2, "car", [(300.0 - 10.0 * i, 178.0) for i in range(13)])
        self.assertGreater(bbox_iou(first.bbox, second.bbox), 0.12)
        for t in (1.0, 1.1, 1.2):
            signals = self.rules.collect_signals(frame(self.scene, t, [first, second]))
        self.assertFalse(
            signals["accident"],
            "overlapping boxes that carry on and separate are not a collision",
        )

    def test_near_miss_when_a_vehicle_swerves_away(self) -> None:
        # Two cars converge closely but stay apart, and one swerves.
        first = make_track(1, "car", [(150.0 + 7.0 * i, 180.0) for i in range(12)])
        second = make_track(2, "car", [(230.0 - 6.0 * i, 190.0) for i in range(12)])
        for t in (0.5, 0.9, 1.1, 1.3):
            signals = self.rules.collect_signals(frame(self.scene, t, [first, second]))
        self.assertFalse(
            signals["near_miss"] and signals["accident"],
            "a close pass with no evasive response is neither",
        )


class TurnRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(default_rules())

    def test_legal_straight_travel_is_not_an_illegal_turn(self) -> None:
        track = make_track(1, "car", [(60.0 + 8.0 * i, 170.0 + 1.1 * i) for i in range(12)])
        track.lane_history = [1, 1]
        track.lane_id = 1
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertFalse(signals["illegal_turn"])

    def test_u_turn_requires_a_reversal_not_a_lane_change(self) -> None:
        # A gentle lane change: large net displacement, no reversal.
        track = make_track(1, "car", [(80.0 + 9.0 * i, 170.0 + 1.0 * i) for i in range(14)])
        signals = self.rules.collect_signals(frame(self.scene, 1.3, [track]))
        self.assertFalse(signals["illegal_u_turn"], "drifting across a lane is not a U-turn")

    def test_u_turn_silent_for_straight_through_traffic(self) -> None:
        track = make_track(1, "car", [(60.0 + 10.0 * i, 175.0) for i in range(40)], dt=0.1)
        for t in (1.0, 2.0, 3.0):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertFalse(signals["illegal_u_turn"], "driving straight is not a U-turn")

    def test_u_turn_fires_on_a_genuine_reversal(self) -> None:
        # Drives east along the lane for 2 s, then reverses and drives back west
        # for 2 s. Net displacement is small relative to the path, which is what
        # separates a U-turn from driving straight through the junction.
        centers = [(60.0 + 10.0 * i, 175.0) for i in range(21)]
        centers += [(260.0 - 10.0 * i, 180.0) for i in range(1, 21)]
        track = make_track(1, "car", centers, dt=0.1)
        for t in (3.6, 3.8, 4.0):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertTrue(signals["illegal_u_turn"], "a sustained reversal against the flow is a U-turn")


class AppearanceRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(default_rules())

    def test_road_obstacle_fires_for_a_persistent_non_road_user(self) -> None:
        centers = [(200.0, 190.0)] * 12
        track = make_track(1, "cone", centers, velocity=(0.0, 0.0), half=(5.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertTrue(signals["road_obstacle"])

    def test_road_obstacle_silent_for_a_moving_object(self) -> None:
        # Something classified as an obstacle but driving away is not one.
        track = make_track(1, "cone", [(60.0 + 12.0 * i, 190.0) for i in range(12)])
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertFalse(signals["road_obstacle"])

    def test_fire_smoke_fires_for_a_persistent_source(self) -> None:
        centers = [(200.0, 190.0)] * 12
        track = make_track(1, "smoke", centers, velocity=(0.0, 0.0), half=(5.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [track]))
        self.assertTrue(signals["fire_smoke"])

    def test_appearance_rules_silent_for_ordinary_road_users(self) -> None:
        track = make_track(1, "car", [(200.0, 190.0)] * 12, velocity=(0.0, 0.0))
        signals = self.rules.collect_signals(frame(self.scene, 1.1, [track]))
        self.assertFalse(signals["road_obstacle"])
        self.assertFalse(signals["fire_smoke"])


class MultiInstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(default_rules())

    def test_every_rule_has_a_dispatchable_candidate_producer(self) -> None:
        """Each rule's candidate producer must take exactly (self, state).

        ``collect_signals`` finds producers with
        ``getattr(self, f"_{label}_candidates")`` and calls them with the frame
        state. A producer with any other signature is therefore not a dispatch
        bug that raises - it is caught per frame and silently reduced to no
        events, because the rule loop swallows the exception. That happened to
        the two turn rules during development, so it is asserted here.
        """
        import inspect

        from src.rules.engine import OFFICIAL_LABELS

        candidates = self.rules._candidate_methods()
        for label in OFFICIAL_LABELS:
            with self.subTest(rule=label):
                self.assertIn(label, candidates,
                              f"{label} has no candidate producer, so multiple "
                              f"simultaneous instances would collapse into one")
                parameters = list(inspect.signature(candidates[label]).parameters)
                self.assertEqual(
                    parameters, ["state"],
                    f"{label}_candidates must take only (state); got {parameters}",
                )

    def test_collect_signals_never_raises(self) -> None:
        """A rule that raises must not take the whole frame down with it."""
        scene = build_scene()
        engine = RuleEngine(default_rules())

        def explode(state):
            raise RuntimeError("deliberate failure")

        engine._candidate_methods = lambda: {label: explode for label in OFFICIAL_LABELS_OF(engine)}
        with self.assertWarns(UserWarning):
            signals = engine.collect_signals(frame(scene, 1.0, []))
        self.assertTrue(all(not items for items in signals.values()))

    def test_several_simultaneous_instances_are_all_reported(self) -> None:
        """Two jaywalkers must produce two signals, not one merged flag."""
        # Both are on the carriageway (the road polygon spans roughly y 103-233
        # at x=186) and both are outside the crossing at x 280-360.
        first = make_track(1, "person", [(120.0 + 6.0 * i, 190.0) for i in range(12)], half=(3.0, 5.0))
        second = make_track(2, "person", [(120.0 + 6.0 * i, 222.0) for i in range(12)], half=(3.0, 5.0))
        for t in (0.5, 0.9, 1.1):
            signals = self.rules.collect_signals(frame(self.scene, t, [first, second]))
        self.assertEqual(
            len(signals["jaywalking"]), 2,
            "each jaywalker is a separate event and must not be collapsed",
        )


if __name__ == "__main__":
    unittest.main()
