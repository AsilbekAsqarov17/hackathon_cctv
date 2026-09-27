from __future__ import annotations

import unittest
from collections import deque

from src.contracts import FrameState, SceneState, TrackState, TrafficLightState
from src.rules.engine import RuleEngine
from src.scene.config import Lane, SceneConfig, SceneContext, SceneLine, TrafficLightROI


def build_scene() -> SceneContext:
    """A tiny scene with a directed stop line and a signal facing the same way.

    The stop line's ``direction`` and the light ROI's ``direction`` both point
    +y, which is how a head is matched to the line it governs.
    """
    config = SceneConfig(
        scene_id="test",
        width=200,
        height=200,
        lanes=[Lane(0, [[0, 0], [200, 0], [200, 200], [0, 200]], (0.0, 1.0))],
        stop_lines=[SceneLine("stop", ((0.0, 100.0), (200.0, 100.0)), "stop", (0.0, 1.0))],
        road_polygon=[[0, 0], [200, 0], [200, 200], [0, 200]],
        traffic_lights=[TrafficLightROI("north", (0.0, 0.0, 20.0, 20.0), (0.0, 1.0))],
    )
    scene = SceneContext(config)
    scene.lights = {"north": TrafficLightState("north", "red", 1.0, (0.0, 0.0, 20.0, 20.0))}
    return scene


def build_track(
    track_id: int,
    name: str,
    center: tuple[float, float],
    velocity: tuple[float, float],
    timestamp: float,
    bottom_history: list[tuple[float, float, float]] | None = None,
) -> TrackState:
    x, y = center
    state = TrackState(
        track_id=track_id,
        class_id=2 if name == "car" else 0,
        class_name=name,
        bbox=(x - 10.0, y - 10.0, x + 10.0, y + 10.0),
        score=0.9,
        first_seen=timestamp,
        last_seen=timestamp,
        center=center,
        velocity=velocity,
        history=deque(
            [(timestamp - 1.0, x - velocity[0], y - velocity[1]), (timestamp, x, y)], maxlen=300
        ),
        lane_id=0,
    )
    if bottom_history is not None:
        state.bottom_history = deque(bottom_history, maxlen=300)
    return state


def make_frame(scene: SceneContext, timestamp: float, tracks: list[TrackState]) -> FrameState:
    return FrameState(
        int(timestamp * 10),
        timestamp,
        tracks,
        SceneState("test", 200, 200, getattr(scene, "lights", {}), scene),
    )


class RuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = build_scene()
        self.rules = RuleEngine(
            {
                "wrong_way": {
                    "enabled": True,
                    "min_speed": 1.0,
                    "direction_dot": -0.4,
                    "min_travel_px": 2.0,
                    "window": 1.0,
                    "confirm_sec": 0.0,
                },
                "stopped_vehicle": {
                    "enabled": True,
                    "speed": 1.0,
                    "duration": 1.0,
                    "window": 1.0,
                    "net_max_px": 25.0,
                    "path_max_px": 250.0,
                },
                "red_light": {"enabled": True, "hold": 1.0},
            }
        )

    # ------------------------------------------------------------------
    # wrong_way
    # ------------------------------------------------------------------
    def test_wrong_way_fires_on_calibrated_lane(self) -> None:
        """Opposing travel on a calibrated lane fires after confirmation."""
        # Lane direction is (0, 1); the wrong-way vehicle travels -y.
        # History is sampled every 0.25 s so the 1.0 s window really is
        # populated at the evaluation time.
        samples = [
            (0.0, 100.0, 150.0), (0.25, 100.0, 145.0), (0.5, 100.0, 140.0),
            (0.75, 100.0, 135.0), (1.0, 100.0, 130.0),
        ]
        wrong = build_track(1, "car", (100.0, 150.0), (0.0, -20.0), 1.0, samples)
        signals = self.rules.evaluate(make_frame(self.scene, 1.0, [wrong]))
        self.assertTrue(signals["wrong_way"].active)

    def test_wrong_way_requires_sustained_motion(self) -> None:
        """One noisy frame in the wrong direction must not fire the rule."""
        still = build_track(1, "car", (100.0, 150.0), (0.0, 0.0), 1.0, [(0.0, 100.0, 150.0), (1.0, 100.0, 150.0)])
        signals = self.rules.evaluate(make_frame(self.scene, 1.0, [still]))
        self.assertFalse(signals["wrong_way"].active)

    def test_wrong_way_respects_lane_allowlist(self) -> None:
        """Only lanes in the allowlist can raise wrong_way."""
        # Lane 7 is inserted first so it wins lane_for_point for this position,
        # and carries the same opposing geometry as lane 0. The two engines
        # differ only in the allowlist, so the pair isolates that behaviour.
        other = Lane(7, [[0, 0], [200, 0], [200, 200], [0, 200]], (0.0, 1.0))
        self.scene.config.lanes.insert(0, other)
        samples = [
            (0.0, 100.0, 150.0), (0.25, 100.0, 145.0), (0.5, 100.0, 140.0),
            (0.75, 100.0, 135.0), (1.0, 100.0, 130.0),
        ]

        def engine(allow: list[int]) -> RuleEngine:
            return RuleEngine(
                {
                    "wrong_way": {
                        "enabled": True,
                        "min_speed": 1.0,
                        "direction_dot": -0.4,
                        "min_travel_px": 2.0,
                        "window": 1.0,
                        "confirm_sec": 0.0,
                        "lane_allowlist": allow,
                    }
                }
            )

        # Control: the same track does fire when its lane is allowed.
        track = build_track(1, "car", (100.0, 150.0), (0.0, -20.0), 1.0, samples)
        track.lane_id = 7
        self.assertTrue(engine([7]).evaluate(make_frame(self.scene, 1.0, [track]))["wrong_way"].active)

        track = build_track(1, "car", (100.0, 150.0), (0.0, -20.0), 1.0, samples)
        track.lane_id = 7
        self.assertFalse(engine([0]).evaluate(make_frame(self.scene, 1.0, [track]))["wrong_way"].active)

    def test_trailing_window_ignores_samples_after_now(self) -> None:
        """A window must not reach past ``now`` into future history."""
        from src.rules.motion import trailing_window

        points = [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0), (2.0, 20.0, 0.0), (3.0, 30.0, 0.0)]
        # now=1.0, window=1.5 -> (-0.5, 1.0]: the two future samples are dropped.
        windowed = trailing_window(points, 1.5, now=1.0)
        self.assertEqual([p[0] for p in windowed], [0.0, 1.0])
        # A narrower window at the same instant.
        self.assertEqual([p[0] for p in trailing_window(points, 0.5, now=1.0)], [1.0])
        # With now=None the newest sample ends the window (unchanged behaviour).
        self.assertEqual([p[0] for p in trailing_window(points, 1.5)], [2.0, 3.0])

    # ------------------------------------------------------------------
    # red_light
    # ------------------------------------------------------------------
    def _crossing_vehicle(self) -> TrackState:
        """A car whose front travels from y=70 (upstream) to y=120 (past) the line.

        The line sits at y=100 spanning the frame with permitted direction +y, so
        the vehicle is approaching downwards. Its box is 20px tall and its front
        edge (larger y) is what has to reach the line, which is why the box
        history is set explicitly rather than left at the default.
        """
        vehicle = build_track(3, "car", (100.0, 100.0), (0.0, 20.0), 1.0)
        path = ((0.0, 60.0), (0.25, 70.0), (0.5, 80.0), (0.75, 90.0), (1.0, 105.0))
        vehicle.bottom_history = deque([(t, 100.0, y) for t, y in path])
        vehicle.history = deque([(t, 100.0, y - 10.0) for t, y in path])
        # The front edge is box_bottom = bottom_centre, so the front reaches the
        # line at t=0.75 while the recorded bottom centre reaches it at t=1.0.
        vehicle.bbox_history = deque([
            (t, 90.0, y - 20.0, 110.0, y) for t, y in path
        ])
        return vehicle

    def test_red_light_fires_when_front_crosses_on_red(self) -> None:
        vehicle = self._crossing_vehicle()
        self.assertTrue(self.rules.evaluate(make_frame(self.scene, 1.0, [vehicle]))["red_light"].active)

    def test_red_light_silent_when_signal_is_green(self) -> None:
        self.scene.lights = {"north": TrafficLightState("north", "green", 1.0, (0.0, 0.0, 20.0, 20.0))}
        vehicle = self._crossing_vehicle()
        self.assertFalse(self.rules.evaluate(make_frame(self.scene, 1.0, [vehicle]))["red_light"].active)

    def test_red_light_does_not_treat_yellow_as_red(self) -> None:
        """Yellow is not red: the official definition is a red-light violation."""
        self.scene.lights = {"north": TrafficLightState("north", "yellow", 1.0, (0.0, 0.0, 20.0, 20.0))}
        vehicle = self._crossing_vehicle()
        self.assertFalse(self.rules.evaluate(make_frame(self.scene, 1.0, [vehicle]))["red_light"].active)

    def test_red_light_ignores_vehicle_leaving_the_intersection(self) -> None:
        """A vehicle moving away from the line cannot be crossing it."""
        vehicle = build_track(3, "car", (100.0, 140.0), (0.0, 20.0), 1.0)
        vehicle.bottom_history = deque([
            (0.0, 100.0, 120.0), (0.5, 100.0, 130.0), (1.0, 100.0, 140.0),
        ])
        vehicle.bbox_history = deque([
            (0.0, 90.0, 100.0, 110.0, 120.0), (0.5, 90.0, 110.0, 110.0, 130.0),
            (1.0, 90.0, 120.0, 110.0, 140.0),
        ])
        self.assertFalse(self.rules.evaluate(make_frame(self.scene, 1.0, [vehicle]))["red_light"].active)

    def test_red_light_silent_when_signal_unreadable(self) -> None:
        """An unknown signal must not be treated as red."""
        self.scene.lights = {"north": TrafficLightState("north", "unknown", 0.0, (0.0, 0.0, 20.0, 20.0))}
        vehicle = self._crossing_vehicle()
        self.assertFalse(self.rules.evaluate(make_frame(self.scene, 1.0, [vehicle]))["red_light"].active)

    # ------------------------------------------------------------------
    # stopped_vehicle
    # ------------------------------------------------------------------
    def test_stopped_vehicle_uses_road_fallback_without_scene_geometry(self) -> None:
        from src.scene.config import SceneConfig as _SceneConfig

        scene = SceneContext(_SceneConfig("empty", 200, 200), 200, 200)
        rules = RuleEngine(
            {
                "stopped_vehicle": {
                    "enabled": True,
                    "speed": 1.0,
                    "duration": 1.0,
                    "window": 1.0,
                    "net_max_px": 25.0,
                    "path_max_px": 250.0,
                }
            }
        )
        # History starts a full window before the first evaluated frame, so the
        # 1.0 s window is genuinely populated at t=0.0 (as it is for a real track).
        samples = [(t, 100.0, 150.0) for t in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]
        vehicle = build_track(1, "car", (100.0, 150.0), (0.0, 0.0), 0.0, samples)
        signal = None
        for t in (0.0, 0.5, 1.0, 1.5):
            frame = FrameState(int(t * 10), t, [vehicle], SceneState("empty", 200, 200, {}, scene))
            signal = rules.evaluate(frame)["stopped_vehicle"]
        self.assertTrue(signal.active)

    def test_stopped_vehicle_requires_duration(self) -> None:
        # The fixture scene has a red light on a stop line at y=100, so a car at
        # y=50 is queueing and must be suppressed (covered separately). This
        # test is about the duration requirement, so it uses a green light: the
        # vehicle is stationary on the carriageway with nothing to wait for.
        self.scene.lights = {"north": TrafficLightState("north", "green", 1.0, (0.0, 0.0, 20.0, 20.0))}
        samples = [(t, 100.0, 50.0) for t in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]
        vehicle = build_track(1, "car", (100.0, 50.0), (0.0, 0.0), 0.0, samples)
        self.assertFalse(self.rules.evaluate(make_frame(self.scene, 0.0, [vehicle]))["stopped_vehicle"].active)
        signal = None
        for t in (0.5, 1.0, 1.5):
            signal = self.rules.evaluate(make_frame(self.scene, t, [vehicle]))["stopped_vehicle"]
        self.assertTrue(signal.active)
        self.assertIsNotNone(signal.start_hint)

    def test_stopped_vehicle_suppresses_signal_queue(self) -> None:
        """A vehicle waiting at a red light must not be reported as stopped."""
        # Upstream of the stop line (y=100) means y>100, and the light is red.
        samples = [(t, 100.0, 160.0) for t in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)]
        vehicle = build_track(1, "car", (100.0, 150.0), (0.0, 0.0), 0.0, samples)
        signal = None
        for t in (0.0, 0.5, 1.0, 1.5, 2.0):
            signal = self.rules.evaluate(make_frame(self.scene, t, [vehicle]))["stopped_vehicle"]
        self.assertFalse(signal.active)

    def test_stopped_vehicle_suppresses_queue_tail_behind_green(self) -> None:
        """The tail of a standing queue is suppressed even with a green light.

        The lead car waits 130 (upstream), the follower waits 170. The follower
        is far from the line but is queueing because the lead car sits between it
        and the line. Clearing the light to green must not turn the tail into a
        stopped_vehicle.
        """
        self.scene.lights = {"north": TrafficLightState("north", "green", 1.0, (0.0, 0.0, 20.0, 20.0))}
        lead = build_track(1, "car", (100.0, 120.0), (0.0, 0.0), 0.0,
                           [(t, 100.0, 130.0) for t in (0.0, 0.5, 1.0, 1.5, 2.0)])
        tail = build_track(2, "car", (100.0, 160.0), (0.0, 0.0), 0.0,
                           [(t, 100.0, 170.0) for t in (0.0, 0.5, 1.0, 1.5, 2.0)])
        signal = None
        for t in (0.0, 0.5, 1.0, 1.5, 2.0):
            signal = self.rules.evaluate(make_frame(self.scene, t, [lead, tail]))["stopped_vehicle"]
        self.assertFalse(signal.active)

    def test_stopped_vehicle_fires_for_isolated_long_stop(self) -> None:
        """A lone vehicle stopped far from the line with no red light is a hit."""
        self.scene.lights = {"north": TrafficLightState("north", "green", 1.0, (0.0, 0.0, 20.0, 20.0))}
        times = [t / 10.0 for t in range(0, 41)]
        vehicle = build_track(1, "car", (100.0, 50.0), (0.0, 0.0), 0.0,
                              [(t, 100.0, 50.0) for t in times])
        fired = False
        for t in times:
            signal = self.rules.evaluate(make_frame(self.scene, t, [vehicle]))["stopped_vehicle"]
            fired = fired or signal.active
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
