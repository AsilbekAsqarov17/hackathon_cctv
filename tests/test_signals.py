from __future__ import annotations

import unittest

import numpy as np

from src.scene.signals import classify_light_color


class SignalTests(unittest.TestCase):
    def test_red_roi(self) -> None:
        roi = np.zeros((40, 40, 3), dtype=np.uint8)
        roi[:, :] = (0, 0, 255)
        color, confidence = classify_light_color(roi)
        self.assertEqual(color, "red")
        self.assertGreater(confidence, 0.0)

    def test_green_roi(self) -> None:
        roi = np.zeros((40, 40, 3), dtype=np.uint8)
        roi[:, :] = (0, 255, 0)
        color, _ = classify_light_color(roi)
        self.assertEqual(color, "green")


if __name__ == "__main__":
    unittest.main()
