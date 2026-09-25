from __future__ import annotations

import unittest

from src.scene.calibration import compute_homography, validate_homography


class CalibrationTests(unittest.TestCase):
    def test_homography_round_trip(self) -> None:
        image_points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        world_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        matrix = compute_homography(image_points, world_points)
        report = validate_homography(matrix, image_points, world_points)
        self.assertLess(report["mean_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
