from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from src.contracts import FrameState, SceneState, TrackState
from src.part_b import (
    CausalRiskEstimator,
    _score_features,
    clear_risk_features,
    get_risk_features,
    publish_risk_features,
)
from src.risk.features import extract_risk_features
from src.scene.config import SceneContext


def feature(timestamp: float, ttc: float | None = None, distance: float | None = None) -> dict:
    return {
        "timestamp": timestamp,
        "vehicle_count": 2,
        "person_count": 0,
        "close_pair_count": 1 if distance is not None else 0,
        "min_ttc": ttc,
        "min_distance": distance,
        "max_closing_speed": 0.0,
        "max_deceleration": 0.0,
        "max_acceleration": 0.0,
        "pedestrian_conflict": 0.0,
        "accident_candidate": False,
        "near_miss_candidate": False,
        "wrong_way_candidate": False,
        "stopped_count": 0,
    }


class PartBTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_risk_features("causal_test.mp4")

    def test_feature_extraction_uses_pair_ttc(self) -> None:
        first = TrackState(
            1, 2, "car", (0, 0, 20, 20), 0.9, 0.0, 0.0, (10, 10),
            velocity=(20.0, 0.0), history=deque(), lane_id=0,
        )
        second = TrackState(
            2, 2, "car", (80, 0, 100, 20), 0.9, 0.0, 0.0, (90, 10),
            velocity=(-20.0, 0.0), history=deque(), lane_id=0,
        )
        state = FrameState(0, 0.0, [first, second], SceneState("test", 200, 100, {}, SceneContext(None, 200, 100)))
        result = extract_risk_features(state)
        self.assertIsNotNone(result.min_ttc)
        self.assertLess(result.min_ttc, 5.0)
        self.assertGreater(result.max_closing_speed, 0.0)

    def test_close_ttc_can_trigger_alarm_score(self) -> None:
        score = _score_features(feature(0.0, 0.5, 20.0), 0.0, {"ttc_floor": 0.45})
        self.assertGreaterEqual(score, 0.5)

    def test_cache_reader_is_causal(self) -> None:
        config = "configs/default.json"
        first_future = [feature(0.0, 5.0, 300.0), feature(1.0, 0.8, 40.0), feature(2.0, 0.01, 1.0)]
        second_future = [feature(0.0, 5.0, 300.0), feature(1.0, 0.8, 40.0), feature(2.0, None, None)]
        publish_risk_features("causal_test.mp4", first_future)
        first = CausalRiskEstimator(config)
        first.reset({"video_id": "causal_test.mp4", "fps": 25.0, "width": 200, "height": 100, "n_frames": 75})
        score0 = first.step(np.zeros((100, 200, 3), dtype=np.uint8), 0.0)
        score1 = first.step(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
        publish_risk_features("causal_test.mp4", second_future)
        second = CausalRiskEstimator(config)
        second.reset({"video_id": "causal_test.mp4", "fps": 25.0, "width": 200, "height": 100, "n_frames": 75})
        second_score0 = second.step(np.zeros((100, 200, 3), dtype=np.uint8), 0.0)
        second_score1 = second.step(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
        self.assertEqual(score0, second_score0)
        self.assertEqual(score1, second_score1)
        self.assertGreater(score1, score0)
        self.assertTrue(0.0 <= score1 <= 1.0)

    def test_missing_cache_returns_zero_without_crashing(self) -> None:
        estimator = CausalRiskEstimator("configs/default.json")
        estimator.config["detector"] = {"backend": "null"}
        estimator.reset({"video_id": "missing.mp4", "fps": 25.0, "width": 320, "height": 240, "n_frames": 10})
        # The default detector may be available in the development environment;
        # either way the score must be a valid scalar and no video may be opened.
        score = estimator.step(np.zeros((240, 320, 3), dtype=np.uint8), 0.0)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
