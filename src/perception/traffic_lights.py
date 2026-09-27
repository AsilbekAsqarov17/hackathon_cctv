"""Traffic-light perception.

The fine-tuned checkpoint regressed to zero traffic-light recall while the base
COCO checkpoint still localizes signal heads, so lights are read from a
dedicated detector. See ``docs/GPU_TRAINING_RESULTS.md`` for the measurements.

Localization and state classification are deliberately separate concerns:
``detect_traffic_lights`` only says *where* signal heads are, and
``classify_traffic_light_state`` only says what colour one currently shows.
A detected light is therefore never by itself evidence of a red light, and a
red_light event still needs a stop-line crossing on top of the state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import BBox, Detection, TrafficLightState
from ..scene.geometry import bbox_distance
from ..scene.signals import classify_light_color

# Normalised spellings seen across COCO and custom checkpoints.
TRAFFIC_LIGHT_NAMES = {
    "traffic light",
    "traffic_light",
    "trafficlight",
    "traffic signal",
    "traffic_signal",
    "signalt",
    "stoplight",
}


def is_traffic_light(class_name: str) -> bool:
    """True when a detector class name denotes a traffic light."""
    normalized = str(class_name).lower().strip().replace("-", "_").replace(" ", "_")
    return normalized in TRAFFIC_LIGHT_NAMES or "traffic_light" in normalized


@dataclass
class TrafficLightObservation:
    """One localized signal head plus its current colour reading."""

    light_id: str
    bbox: BBox
    score: float
    color: str
    color_confidence: float
    matched: bool = True
    history: list[tuple[float, str]] = field(default_factory=list)


class TrafficLightPerception:
    """Locate signal heads with a detector, then read their colour over time.

    ``model_config`` must describe a checkpoint that still detects traffic
    lights (the base ``weights/yolo11n.pt``), not the fine-tuned one.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        min_score: float = 0.10,
        debounce: int = 3,
        min_pixels: int = 8,
        max_lost_frames: int = 30,
        max_match_distance: float = 80.0,
    ):
        from .detector import YoloDetector

        self.detector = YoloDetector(model_config)
        self.min_score = float(min_score)
        self.min_pixels = int(min_pixels)
        self.debounce = max(1, int(debounce))
        self.max_lost_frames = int(max_lost_frames)
        self.max_match_distance = float(max_match_distance)
        self._next_id = 1
        # light_id -> bbox/score/colour history/last frame seen
        self._tracks: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # localization
    # ------------------------------------------------------------------
    def detect_traffic_lights(self, frame: np.ndarray) -> list[Detection]:
        """Return only the traffic-light detections for ``frame``."""
        return [
            detection
            for detection in self.detector.predict(frame)
            if is_traffic_light(detection.class_name)
        ]

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def classify_traffic_light_state(
        self, frame: np.ndarray, bbox: BBox
    ) -> tuple[str, float]:
        """Classify one signal-head box as red/yellow/green/unknown.

        Returns ``("unknown", 0.0)`` when the crop is empty or too small, so a
        caller can never mistake a missing reading for a red light.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, x2 = max(0, int(x1)), min(width, int(x2))
        y1, y2 = max(0, int(y1)), min(height, int(y2))
        if x2 <= x1 or y2 <= y1:
            return "unknown", 0.0
        return classify_light_color(frame[y1:y2, x1:x2], min_pixels=self.min_pixels)

    # ------------------------------------------------------------------
    # per-frame association
    # ------------------------------------------------------------------
    def _assign_ids(
        self, detections: list[Detection], frame_id: int
    ) -> list[tuple[str, Detection]]:
        """Greedily pair detections with existing lights by spatial proximity.

        Signal heads are static, so proximity is a reliable identity cue and
        keeps one ``light_id`` per physical head across frames.
        """
        pairs: list[tuple[float, str, int]] = []
        for light_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                distance = bbox_distance(track["bbox"], detection.bbox)
                if distance <= self.max_match_distance:
                    pairs.append((distance, light_id, index))
        pairs.sort(key=lambda item: item[0])
        used_lights: set[str] = set()
        used_detections: set[int] = set()
        output: list[tuple[str, Detection]] = []
        for _, light_id, index in pairs:
            if light_id in used_lights or index in used_detections:
                continue
            used_lights.add(light_id)
            used_detections.add(index)
            output.append((light_id, detections[index]))
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            light_id = f"tl_{self._next_id}"
            self._next_id += 1
            self._tracks[light_id] = {
                "bbox": detection.bbox,
                "score": detection.score,
                "colors": deque(maxlen=self.debounce),
                "last_frame": frame_id,
                "history": [],
            }
            output.append((light_id, detection))
        return output

    @property
    def _max_match_distance(self) -> float:
        return float(getattr(self, "_max_dist", 80.0))

    def update(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        detections: list[Detection] | None = None,
    ) -> list[TrafficLightObservation]:
        """Detect, associate and colour-read every visible signal head.

        ``detections`` may be supplied when the caller already ran the light
        detector, so a hybrid pipeline does not pay for a second inference pass.
        """
        if detections is None:
            detections = self.detect_traffic_lights(frame)
        detections = [
            detection
            for detection in detections
            if is_traffic_light(detection.class_name) and detection.score >= self.min_score
        ]
        observations: list[TrafficLightObservation] = []
        for light_id, detection in self._assign_ids(detections, frame_id):
            track = self._tracks[light_id]
            track["bbox"] = detection.bbox
            track["score"] = detection.score
            track["last_frame"] = frame_id
            color, color_confidence = self.classify_traffic_light_state(frame, detection.bbox)
            colors: deque[str] = track["colors"]
            colors.append(color)
            stable = self._stabilize(colors)
            track["history"].append((float(frame_id), stable))
            track["history"] = track["history"][-600:]
            observations.append(
                TrafficLightObservation(
                    light_id=light_id,
                    bbox=detection.bbox,
                    score=detection.score,
                    color=stable,
                    color_confidence=color_confidence,
                    history=list(track["history"][-60:]),
                )
            )
        # Retire heads that have been absent long enough to be considered gone.
        for light_id in [
            light_id
            for light_id, track in self._tracks.items()
            if frame_id - track["last_frame"] > self.max_lost_frames
        ]:
            del self._tracks[light_id]
        return observations

    @staticmethod
    def _stabilize(colors: deque[str]) -> str:
        """Majority vote over the recent window, ``unknown`` while undecided."""
        if len(colors) < 1:
            return "unknown"
        counts: dict[str, int] = {}
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
        winner, count = max(counts.items(), key=lambda item: item[1])
        # A single red frame in a red/green sequence must not emit red.
        if count * 2 <= len(colors):
            return "unknown"
        return winner

    def as_states(self, observations: list[TrafficLightObservation]) -> dict[str, TrafficLightState]:
        """Convert observations to the shared contract used by the rules engine."""
        return {
            observation.light_id: TrafficLightState(
                light_id=observation.light_id,
                color=observation.color,
                confidence=observation.color_confidence,
                bbox=observation.bbox,
            )
            for observation in observations
        }
