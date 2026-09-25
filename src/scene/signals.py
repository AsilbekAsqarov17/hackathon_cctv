from __future__ import annotations

from collections import Counter, deque
from typing import Any

import cv2
import numpy as np

from ..contracts import TrafficLightState


def classify_light_color(roi_bgr: np.ndarray, min_pixels: int = 8) -> tuple[str, float]:
    """Classify a configured traffic-light ROI as red/yellow/green/unknown."""
    if roi_bgr.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Two red hue ranges handle the wrap around 0 degrees.
    masks = {
        "red": cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
        | cv2.inRange(hsv, np.array([170, 80, 80]), np.array([179, 255, 255])),
        "yellow": cv2.inRange(hsv, np.array([13, 80, 80]), np.array([38, 255, 255])),
        "green": cv2.inRange(hsv, np.array([39, 50, 50]), np.array([95, 255, 255])),
    }
    color, count = "unknown", 0
    for name, mask in masks.items():
        value = int(cv2.countNonZero(mask))
        if value > count:
            color, count = name, value
    if count < min_pixels:
        return "unknown", 0.0
    confidence = min(1.0, count / max(1, roi_bgr.shape[0] * roi_bgr.shape[1] * 0.08))
    return color, float(confidence)


class TrafficLightReader:
    """Reads configured ROIs and detector-provided light boxes."""

    def __init__(self, scene: Any, debounce_frames: int = 3):
        self.scene = scene
        self.debounce_frames = max(1, int(debounce_frames))
        self._history: dict[str, deque[str]] = {}

    def _stabilize(self, light_id: str, color: str) -> str:
        history = self._history.setdefault(light_id, deque(maxlen=self.debounce_frames))
        history.append(color)
        if len(history) < self.debounce_frames:
            return "unknown"
        counts = Counter(history)
        winner, count = counts.most_common(1)[0]
        return winner if count > len(history) / 2 else "unknown"

    def update(self, frame: np.ndarray, detections: list[Any] | None = None) -> dict[str, TrafficLightState]:
        states: dict[str, TrafficLightState] = {}
        config = getattr(self.scene, "config", None)
        h, w = frame.shape[:2]
        if config is not None:
            for light in config.traffic_lights:
                x1, y1, x2, y2 = light.roi
                x1, x2 = max(0, int(x1)), min(w, int(x2))
                y1, y2 = max(0, int(y1)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = frame[y1:y2, x1:x2]
                color, confidence = classify_light_color(roi)
                stable_color = self._stabilize(light.light_id, color)
                states[light.light_id] = TrafficLightState(
                    light_id=light.light_id,
                    color=stable_color,
                    confidence=confidence,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
        self.scene.lights = states
        self.scene.dynamic_lights = {}
        for index, detection in enumerate(detections or []):
            name = str(getattr(detection, "class_name", "")).lower().replace("_", " ")
            if "traffic light" not in name:
                continue
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            color, confidence = classify_light_color(frame[y1:y2, x1:x2])
            light_id = f"det_{index}"
            stable_color = self._stabilize(light_id, color)
            self.scene.dynamic_lights[light_id] = TrafficLightState(
                light_id=light_id,
                color=stable_color,
                confidence=confidence,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            )
        return states
