from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import BBox, Detection

# COCO ids are used when a stock Ultralytics checkpoint is selected. Custom
# checkpoints may use different ids, so names are always retained.
COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    13: "stop sign",
}
RELEVANT_NAMES = {
    "person",
    "pedestrian",
    "bicycle",
    "bike",
    "cyclist",
    "car",
    "vehicle",
    "motorcycle",
    "motorbike",
    "bus",
    "truck",
    "traffic light",
    "traffic_light",
    "stop sign",
    "stop_sign",
    "fire",
    "smoke",
    "flame",
    "obstacle",
    "debris",
    "animal",
    "cone",
    "barrier",
}


class NullDetector:
    """Safe fallback used when no local model is available."""

    def predict(self, frame: np.ndarray) -> list[Detection]:
        return []


class YoloDetector:
    """Ultralytics adapter.

    The model is intentionally never downloaded implicitly. The competition
    runs offline, so weights must be placed in the configured local path before
    submission. Set ``allow_download`` only for local development.
    """

    _model_cache: dict[tuple[str, str], Any] = {}

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = str(config.get("model", "yolo11n.pt"))
        self.confidence = float(config.get("confidence", 0.25))
        self.iou = float(config.get("iou", 0.5))
        self.imgsz = int(config.get("imgsz", 640))
        self.device = str(config.get("device", ""))
        self.allow_download = bool(config.get("allow_download", False))
        self._model: Any | None = None
        self._names: dict[int, str] = {}
        # Load eagerly so build_detector() can fall back cleanly before the
        # first video frame is processed.
        self._load()

    def _load(self) -> Any:
        model_path = Path(self.model_name).expanduser()
        if not model_path.exists() and not model_path.is_absolute():
            for candidate in (
                Path.cwd() / model_path,
                Path(__file__).resolve().parents[2] / "weights" / model_path,
            ):
                if candidate.exists():
                    model_path = candidate
                    break
        # Ultralytics accepts a URL/path. We only permit a download when the
        # caller explicitly opts in, because official runs have no internet.
        if not model_path.exists() and not self.allow_download and "://" not in self.model_name:
            raise FileNotFoundError(
                f"YOLO weights not found: {self.model_name}. "
                "Place the checkpoint locally or set allow_download=true for development."
            )
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed; install requirements-part-a.txt "
                "or configure a different detector"
            ) from exc
        key = (str(model_path.resolve()) if model_path.exists() else self.model_name, self.device)
        if key not in self._model_cache:
            model_ref = self.model_name if "://" in self.model_name else str(model_path)
            self._model_cache[key] = YOLO(model_ref)
        self._model = self._model_cache[key]
        names = getattr(self._model, "names", COCO_NAMES)
        if isinstance(names, dict):
            self._names = {int(k): str(v) for k, v in names.items()}
        else:
            self._names = {i: str(v) for i, v in enumerate(names)}
        return self._model

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        model = self._load()
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = model.predict(frame, **kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy = self._to_numpy(boxes.xyxy)
        scores = self._to_numpy(boxes.conf)
        classes = self._to_numpy(boxes.cls).astype(int)
        detections: list[Detection] = []
        for coords, score, class_id in zip(xyxy, scores, classes):
            name = self._names.get(int(class_id), COCO_NAMES.get(int(class_id), str(class_id)))
            normalized_name = name.lower().replace("-", "_").replace(" ", "_")
            if normalized_name not in RELEVANT_NAMES:
                continue
            x1, y1, x2, y2 = [float(v) for v in coords[:4]]
            detections.append(
                Detection(
                    bbox=(max(0.0, x1), max(0.0, y1), max(x1, x2), max(y1, y2)),
                    score=float(score),
                    class_id=int(class_id),
                    class_name=name,
                )
            )
        return detections


class MotionDetector:
    """Optional classical fallback for development and smoke tests.

    It is not intended to replace a road-user detector in the final system.
    It can be enabled with detector.fallback_motion=true and is useful when a
    model checkpoint is not yet available.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.min_area = float(config.get("motion_min_area", 250.0))

    def predict(self, frame: np.ndarray) -> list[Detection]:
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output: list[Detection] = []
        h, w = frame.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 8 or bh < 8 or bw * bh < self.min_area:
                continue
            # Motion fallback cannot reliably distinguish object categories.
            # Use a neutral class and let the tracker maintain the track.
            output.append(
                Detection(
                    bbox=(float(x), float(y), float(min(w, x + bw)), float(min(h, y + bh))),
                    score=0.35,
                    class_id=-1,
                    class_name="vehicle",
                )
            )
        return output


def build_detector(config: dict[str, Any]) -> Any:
    backend = str(config.get("backend", "auto")).lower()
    if backend == "null":
        return NullDetector()
    if backend == "motion":
        return MotionDetector(config)
    if backend == "onnx":
        try:
            from .onnx_detector import OnnxDetector

            return OnnxDetector(config)
        except Exception as exc:
            warnings.warn(f"ONNX detector unavailable ({exc}); falling back to no detections")
            return NullDetector()
    try:
        return YoloDetector(config)
    except Exception as exc:
        if bool(config.get("fallback_motion", False)):
            warnings.warn(f"YOLO unavailable ({exc}); using motion fallback")
            return MotionDetector(config)
        warnings.warn(f"YOLO unavailable ({exc}); Part A will run without detections")
        return NullDetector()
