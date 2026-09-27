"""Hybrid perception: fine-tuned road users plus base-model traffic lights.

The fine-tuned checkpoint is strong on vehicles and people but regressed to
zero traffic-light recall, while the base COCO checkpoint still finds signal
heads. ``HybridDetector`` routes each class to the checkpoint that currently
handles it, so a single ``predict`` call still satisfies the detector contract
used by the tracker and the rules engine.

Configuration (see ``configs/data_video1_hybrid.json``)::

    "backend": "hybrid",
    "primary": {"model": "runs/detect/traffic_part_a_gpu/weights/best.pt", ...},
    "traffic_light": {"model": "weights/yolo11n.pt", ...}
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import Detection
from .detector import YoloDetector
from .traffic_lights import TrafficLightPerception, is_traffic_light


class HybridDetector:
    """Per-class detector routing over two checkpoints.

    Vehicles, people and road users come from ``primary`` (fine-tuned). Traffic
    lights come exclusively from ``traffic_light`` (base). The primary model is
    filtered so a traffic light it still does emit cannot duplicate the
    dedicated detector's output.
    """

    def __init__(self, config: dict[str, Any]):
        primary_config = dict(config.get("primary", config))
        light_config = dict(config.get("traffic_light", {}))
        if not light_config:
            raise ValueError(
                "hybrid detector needs a 'traffic_light' block with its own model"
            )
        self.primary = YoloDetector(primary_config)
        self.perception = TrafficLightPerception(light_config)
        self.config = config

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Merge primary road-user detections with dedicated light detections."""
        road_users = [
            detection
            for detection in self.primary.predict(frame)
            if not is_traffic_light(detection.class_name)
        ]
        lights = self.perception.detect_traffic_lights(frame)
        return road_users + lights

    def analyze(self, frame: np.ndarray, frame_id: int = 0) -> tuple[list[Detection], list[Any]]:
        """Single-pass fused result for the runtime loop.

        Returns ``(all_detections, traffic_light_observations)`` and runs each
        checkpoint exactly once per frame, so a caller that needs both the
        tracker input and the light state never double-infers.
        """
        road_users = [
            detection
            for detection in self.primary.predict(frame)
            if not is_traffic_light(detection.class_name)
        ]
        lights = self.perception.detect_traffic_lights(frame)
        observations = self.perception.update(frame, frame_id=frame_id, detections=lights)
        return road_users + lights, observations

    def read_traffic_lights(self, frame: np.ndarray, frame_id: int = 0) -> list[Any]:
        """Localized signal heads with debounced colour state for this frame."""
        return self.perception.update(frame, frame_id=frame_id)

    def as_states(self, observations: list[Any]) -> dict[str, Any]:
        return self.perception.as_states(observations)
