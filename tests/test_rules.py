from __future__ import annotations

import unittest
from collections import deque

from src.contracts import FrameState, SceneState, TrackState, TrafficLightState
from src.rules.engine import RuleEngine
from src.scene.config import Lane, SceneConfig, SceneContext, SceneLine, TrafficLightROI


class RuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene_config = SceneConfig(
            scene_id="test",
            width=200,
            height=200,
            lanes=[Lane(0, [[0, 0], [200, 0], [200, 200], [0, 200]], (0.0, 1.0))],
            stop_lines=[SceneLine("stop", ((0.0, 100.0), (200.0, 100.0)))],
            road_polygon=[[0, 0], [200, 0], [200, 200], [0, 200]],
            traffic_lights=[TrafficLightROI("north", (0.0, 0.0, 20.0, 20.0))],
        )
        self.scene = SceneContext(self.scene_config)
        self.scene.lights = {
            "north": TrafficLightState("north", "red", 1.0, (0.0, 0.0, 20.0, 20.0))
        }
        self.rules = RuleEngine(
            {
                "wrong_way": {"enabled": True, "min_speed": 1.0, "direction_dot": -0.4},
                "stopped_vehicle": {"enabled": True, "speed": 1.0, "duration": 1.0},
                "red_light": {"enabled": True, "hold": 1.0},
            }
        )

    @staticmethod
    def track(track_id: int, name: str, center: tuple[float, float], velocity: tuple[float, float], timestamp: float) -> TrackState:
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
            history=deque([(timestamp - 1.0, x - velocity[0], y - velocity[1]), (timestamp, x, y)], maxlen=300),
            lane_id=0,
        )
        return state

    def state(self, timestamp: float, tracks: list[TrackState]) -> FrameState:
        return FrameState(
            int(timestamp * 10),
            timestamp,
            tracks,
            SceneState("test", 200, 200, self.scene.lights, self.scene),
        )

    def test_wrong_way_and_red_light(self) -> None:
        wrong = self.track(1, "car", (100.0, 120.0), (0.0, -20.0), 1.0)
        crossing = self.track(2, "car", (100.0, 102.0), (0.0, 20.0), 1.0)
        signals = self.rules.evaluate(self.state(1.0, [wrong, crossing]))
        self.assertFalse(signals["wrong_way"].active)
        for t in (1.5, 2.0):
            wrong.last_seen = t
            crossing.last_seen = t
            signals = self.rules.evaluate(self.state(t, [wrong, crossing]))
        self.assertTrue(signals["wrong_way"].active)
        self.assertTrue(self.rules._red_fired)

    def test_red_light_uses_bottom_center_crossing(self) -> None:
        vehicle = self.track(3, "car", (100.0, 100.0), (0.0, 20.0), 1.0)
        vehicle.bottom_history = deque([(0.0, 100.0, 90.0), (1.0, 100.0, 110.0)])
        vehicle.history = deque([(0.0, 100.0, 80.0), (1.0, 100.0, 100.0)])
        signal = self.rules.evaluate(self.state(1.0, [vehicle]))["red_light"]
        self.assertTrue(signal.active)

    def test_stopped_vehicle_uses_road_fallback_without_scene_geometry(self) -> None:
        from src.scene.config import SceneConfig

        scene = SceneContext(SceneConfig("empty", 200, 200), 200, 200)
        rules = RuleEngine({"stopped_vehicle": {"enabled": True, "speed": 1.0, "duration": 1.0}})
        vehicle = self.track(1, "car", (100.0, 150.0), (0.0, 0.0), 0.0)
        for t in (0.0, 0.5, 1.0, 1.5):
            frame = FrameState(int(t * 10), t, [vehicle], SceneState("empty", 200, 200, {}, scene))
            signal = rules.evaluate(frame)["stopped_vehicle"]
        self.assertTrue(signal.active)

    def test_stopped_vehicle_requires_duration(self) -> None:
        vehicle = self.track(1, "car", (100.0, 50.0), (0.0, 0.0), 0.0)
        self.assertFalse(self.rules.evaluate(self.state(0.0, [vehicle]))["stopped_vehicle"].active)
        for t in (0.5, 1.0, 1.5):
            vehicle.last_seen = t
            vehicle.history.append((t, 100.0, 50.0))
            signal = self.rules.evaluate(self.state(t, [vehicle]))["stopped_vehicle"]
        self.assertTrue(signal.active)
        self.assertIsNotNone(signal.start_hint)


if __name__ == "__main__":
    unittest.main()
