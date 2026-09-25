from __future__ import annotations

import unittest
from collections import deque

from src.contracts import TrackState
from src.tracking.analytics import TrafficAnalytics


class AnalyticsTests(unittest.TestCase):
    def test_lane_counts_and_speed(self) -> None:
        tracks = [
            TrackState(1, 2, "car", (0, 0, 20, 20), 0.9, 0, 0, (10, 10), velocity=(10, 0), lane_id=0, history=deque()),
            TrackState(2, 2, "car", (30, 0, 50, 20), 0.9, 0, 0, (40, 10), velocity=(20, 0), lane_id=0, history=deque()),
        ]
        result = TrafficAnalytics().update(tracks)
        self.assertEqual(result["counts"][0], 2)
        self.assertAlmostEqual(result["average_speed"][0], 15.0)


if __name__ == "__main__":
    unittest.main()
