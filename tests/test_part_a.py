from __future__ import annotations

import unittest

import numpy as np

from src.contracts import Detection, RuleSignal
from src.perception.tracker import SimpleByteTracker
from src.scene.geometry import point_in_polygon, segment_crossed_by_track
from src.temporal.segmenter import TemporalSegmenter
from src.tracking.track_manager import TrackManager


class GeometryTests(unittest.TestCase):
    def test_point_in_polygon(self) -> None:
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertTrue(point_in_polygon((5, 5), polygon))
        self.assertFalse(point_in_polygon((15, 5), polygon))

    def test_segment_crossing(self) -> None:
        self.assertTrue(segment_crossed_by_track((0, 5), (10, 5), [[4, 0], [4, 10]]))
        self.assertFalse(segment_crossed_by_track((0, 20), (10, 20), [[4, 0], [4, 10]]))


class TrackingTests(unittest.TestCase):
    def test_tracker_and_manager_history(self) -> None:
        tracker = SimpleByteTracker({"high_threshold": 0.4, "low_threshold": 0.1, "min_hits": 1})
        manager = TrackManager(max_age_seconds=2.0)
        states = []
        for i in range(4):
            detection = Detection((i * 5.0, 10.0, i * 5.0 + 20.0, 30.0), 0.9, 2, "car")
            observations = tracker.update([detection], i, i * 0.1, np.zeros((40, 100, 3), dtype=np.uint8))
            states = manager.update(observations, i * 0.1, i)
        self.assertEqual(len(states), 1)
        self.assertGreater(states[0].speed, 0.0)
        self.assertGreaterEqual(len(states[0].history), 4)


class TemporalTests(unittest.TestCase):
    def test_flags_become_one_segment(self) -> None:
        cfg = {"merge_gap": 0.5, "min_duration": {"wrong_way": 0.5}}
        segmenter = TemporalSegmenter(cfg, duration=5.0)
        for t in (0.0, 0.25, 0.5, 0.75):
            segmenter.add(t, {"wrong_way": RuleSignal("wrong_way", True, 0.9)})
        events = segmenter.finalize()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][2], "wrong_way")
        self.assertGreaterEqual(events[0][1] - events[0][0], 0.5)

    def test_short_flag_is_dropped(self) -> None:
        segmenter = TemporalSegmenter({"min_duration": {"wrong_way": 1.0}}, duration=5.0)
        segmenter.add(0.0, {"wrong_way": RuleSignal("wrong_way", True, 0.9)})
        self.assertEqual(segmenter.finalize(), [])


if __name__ == "__main__":
    unittest.main()
