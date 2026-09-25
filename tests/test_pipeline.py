from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.contracts import Detection
from src.part_a import run_part_a
from src.part_b import CausalRiskEstimator, clear_risk_features, get_risk_features


class PipelineTests(unittest.TestCase):
    def test_end_to_end_stopped_vehicle_with_fake_detector(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "stopped.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (320, 240))
            for _ in range(60):
                frame = np.full((240, 320, 3), 255, dtype=np.uint8)
                cv2.rectangle(frame, (100, 120), (140, 170), (0, 0, 255), -1)
                writer.write(frame)
            writer.release()
            scene = root / "scene.json"
            scene.write_text(
                json.dumps(
                    {
                        "scene_id": "test",
                        "auto_road_fallback": True,
                        "road_polygon": [],
                        "lanes": [],
                    }
                ),
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "detector": {"backend": "null", "stride": 1},
                        "tracker": {"backend": "simple_bytetrack", "min_hits": 1},
                        "scene": {"path": str(scene)},
                        "active_classes": ["stopped_vehicle"],
                        "rules": {
                            "wrong_way": {"enabled": False},
                            "stopped_vehicle": {"enabled": True, "speed": 1.0, "duration": 10.0},
                        },
                        "temporal": {"min_duration": {"stopped_vehicle": 10.0}},
                    }
                ),
                encoding="utf-8",
            )

            class FakeDetector:
                def predict(self, frame):
                    return [Detection((100.0, 120.0, 140.0, 170.0), 0.9, 2, "car")]

            with patch("src.part_a.build_detector", return_value=FakeDetector()):
                events = run_part_a(str(video), str(config))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][2], "stopped_vehicle")
            self.assertGreaterEqual(events[0][1] - events[0][0], 10.0)
            self.assertIsNotNone(get_risk_features("stopped.mp4"))
            risk = CausalRiskEstimator(str(config))
            risk.reset({"video_id": "stopped.mp4", "fps": 5.0, "width": 320, "height": 240, "n_frames": 60})
            score = risk.step(np.zeros((240, 320, 3), dtype=np.uint8), 0.0)
            self.assertGreaterEqual(score, 0.0)
            clear_risk_features("stopped.mp4")


if __name__ == "__main__":
    unittest.main()
