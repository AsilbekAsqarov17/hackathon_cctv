from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from src.contracts import FrameState, SceneState, TrackState
from src.part_b import (
    CausalRiskEstimator,
    _score_features,
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

    def test_close_ttc_alone_must_not_alarm(self) -> None:
        """Regression guard: a short TTC by itself is not a risk signal.

        Measured on ordinary traffic containing no accident, 8.9% of all
        road-user pairs already predict contact within 0.5 s, because a car
        following a queue always extrapolates to contact. A scorer that alarms
        on TTC alone therefore sits above the threshold almost permanently, and
        chance-normalised AP scores a constant signal exactly zero.
        """
        score = _score_features(feature(0.0, ttc=0.4, distance=20.0), 0.0, {})
        self.assertLess(score, 0.5)

    def test_conflict_signature_raises_an_alarm(self) -> None:
        """Short predicted contact plus approach speed plus braking does alarm."""
        hot = feature(0.0, ttc=0.4, distance=20.0)
        hot["max_closing_speed"] = 400.0
        hot["max_deceleration"] = 60.0
        self.assertGreaterEqual(_score_features(hot, 0.0, {}), 0.5)

    def test_ordinary_approach_does_not_alarm(self) -> None:
        """Approach and braking together, the queueing case, stays quiet."""
        queue = feature(0.0, ttc=1.2, distance=60.0)
        queue["max_closing_speed"] = 200.0
        queue["max_deceleration"] = 30.0
        self.assertLess(_score_features(queue, 0.0, {}), 0.5)

    def test_part_a_cache_is_never_consulted(self) -> None:
        """Part B must not read Part A output, even if a cache is handed to it.

        The rules state that reusing Part A output is a violation. The cache
        hooks are kept only so that older code paths cannot silently reintroduce
        the dependency; they have to stay inert.
        """
        publish_risk_features(
            "causal_test.mp4",
            [feature(0.0, 5.0, 300.0), feature(1.0, 0.8, 40.0), feature(2.0, 0.01, 1.0)],
        )
        self.assertIsNone(get_risk_features("causal_test.mp4"))

        estimator = CausalRiskEstimator("configs/default.json")
        estimator.config["detector"] = {"backend": "null"}
        estimator.reset(
            {"video_id": "causal_test.mp4", "fps": 25.0, "width": 200, "height": 100, "n_frames": 75}
        )
        # With no detector there is no evidence, so the score must stay low
        # instead of reproducing the cached high-risk samples.
        scores = [estimator.step(np.zeros((100, 200, 3), dtype=np.uint8), t) for t in (0.0, 1.0, 2.0)]
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        self.assertLess(max(scores), 0.5)

    def test_scores_up_to_t_do_not_depend_on_the_future(self) -> None:
        """Replaying with a different tail must not change the earlier scores."""
        meta = {"video_id": "causal_test.mp4", "fps": 25.0, "width": 64, "height": 48, "n_frames": 75}

        first = CausalRiskEstimator("configs/default.json")
        first.config["detector"] = {"backend": "null"}
        first.reset(meta)
        early = [first.step(np.zeros((48, 64, 3), dtype=np.uint8), t / 25.0) for t in range(25)]

        second = CausalRiskEstimator("configs/default.json")
        second.config["detector"] = {"backend": "null"}
        second.reset(meta)
        replay = []
        for t in range(75):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            if t >= 25:
                frame[:] = 255
            replay.append(second.step(frame, t / 25.0))

        self.assertEqual(early, replay[:25])

    def test_reset_clears_state_between_videos(self) -> None:
        meta = {"video_id": "a.mp4", "fps": 25.0, "width": 64, "height": 48, "n_frames": 10}
        estimator = CausalRiskEstimator("configs/default.json")
        estimator.config["detector"] = {"backend": "null"}
        estimator.reset(meta)
        for t in range(10):
            estimator.step(np.zeros((48, 64, 3), dtype=np.uint8), t / 25.0)
        estimator.reset({**meta, "video_id": "b.mp4"})
        first = estimator.step(np.zeros((48, 64, 3), dtype=np.uint8), 0.0)
        estimator.reset({**meta, "video_id": "c.mp4"})
        again = estimator.step(np.zeros((48, 64, 3), dtype=np.uint8), 0.0)
        self.assertEqual(first, again)

    def test_missing_detector_returns_valid_score(self) -> None:
        estimator = CausalRiskEstimator("configs/default.json")
        estimator.config["detector"] = {"backend": "null"}
        estimator.reset(
            {"video_id": "missing.mp4", "fps": 25.0, "width": 320, "height": 240, "n_frames": 10}
        )
        score = estimator.step(np.zeros((240, 320, 3), dtype=np.uint8), 0.0)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
