from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _configure_cuda_dlls() -> None:
    """Expose pip-installed NVIDIA DLL directories to ONNX Runtime on Windows."""
    import glob
    import os
    import site

    candidates: list[str] = []
    for root in {site.getsitepackages()[0] if site.getsitepackages() else "", site.getusersitepackages() if hasattr(site, "getusersitepackages") else ""}:
        if root:
            candidates.extend(glob.glob(os.path.join(root, "nvidia", "*", "bin")))
    for directory in candidates:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")

from ..contracts import Detection
from .detector import COCO_NAMES, RELEVANT_NAMES


class OnnxDetector:
    """YOLO ONNX detector using CUDAExecutionProvider when available."""

    _sessions: dict[tuple[str, int], Any] = {}

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_path = self._resolve_model(str(config.get("model", "yolo11n.onnx")))
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.imgsz = int(config.get("imgsz", 640))
        self.confidence = float(config.get("confidence", 0.25))
        self.iou = float(config.get("iou", 0.5))
        self.device = str(config.get("device", "0"))
        self.class_names = self._configured_names()
        self.session = self._get_session()

    @staticmethod
    def _resolve_model(name: str) -> Path:
        """Find the checkpoint relative to the project, not the cwd.

        A relative path silently resolved against the current working directory
        turns a missing checkpoint into a NullDetector and therefore into zero
        events for every video, while the run still passes the format check and
        scores zero with no error anywhere. The organizers do run from the
        repository root, but a demo, a notebook or an absolute-path invocation
        should not depend on that.
        """
        path = Path(name).expanduser()
        if path.is_absolute() or path.exists():
            return path
        for candidate in (
            Path.cwd() / path,
            Path(__file__).resolve().parents[2] / path,
            Path(__file__).resolve().parents[2] / "weights" / path.name,
        ):
            if candidate.exists():
                return candidate
        return path

    def _configured_names(self) -> dict[int, str]:
        raw = self.config.get("class_names")
        if isinstance(raw, list):
            return {i: str(name) for i, name in enumerate(raw)}
        return dict(COCO_NAMES)

    def _get_session(self) -> Any:
        _configure_cuda_dlls()
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.device not in {"cpu", "-1"}:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        key = (str(self.model_path.resolve()), self.imgsz)
        if key not in self._sessions:
            session = None
            if providers[0] != "CPUExecutionProvider":
                # A CUDA provider that is listed but cannot actually load its
                # shared libraries is a common failure on fresh hosts. Falling
                # back to CPU keeps the submission alive instead of raising.
                try:
                    session = ort.InferenceSession(str(self.model_path), providers=providers)
                except Exception as exc:  # pragma: no cover - host dependent
                    warnings.warn(f"CUDAExecutionProvider unavailable ({exc}); using CPU")
            if session is None:
                session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            self._sessions[key] = session
        session = self._sessions[key]
        self.active_providers = session.get_providers()
        meta = session.get_modelmeta().custom_metadata_map or {}
        raw_names = meta.get("names")
        if raw_names and not self.config.get("class_names"):
            try:
                parsed = json.loads(raw_names)
                if isinstance(parsed, dict):
                    self.class_names = {int(k): str(v) for k, v in parsed.items()}
                elif isinstance(parsed, list):
                    self.class_names = {i: str(v) for i, v in enumerate(parsed)}
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        return session

    @staticmethod
    def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = round(w * scale), round(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        left, top = (size - new_w) // 2, (size - new_h) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas, scale, float(left), float(top)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        padded, scale, pad_x, pad_y = self._letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, 0)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        predictions = np.asarray(output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        # YOLO11 export is (84, 8400); custom exports may be (8400, 84).
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            return []
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
        keep = scores >= self.confidence
        boxes, class_ids, scores = boxes[keep], class_ids[keep], scores[keep]
        detections: list[Detection] = []
        for box, class_id, score in zip(boxes, class_ids, scores):
            name = self.class_names.get(int(class_id), str(int(class_id)))
            normalized = name.lower().replace("-", "_").replace(" ", "_")
            if normalized not in RELEVANT_NAMES:
                continue
            cx, cy, bw, bh = [float(v) for v in box]
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(frame.shape[1]), x2), min(float(frame.shape[0]), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection((x1, y1, x2, y2), float(score), int(class_id), name)
            )
        return self._nms(detections)

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []
        grouped: dict[int, list[Detection]] = {}
        for detection in detections:
            grouped.setdefault(detection.class_id, []).append(detection)
        kept: list[Detection] = []
        for group in grouped.values():
            boxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in group]
            scores = [d.score for d in group]
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou)
            if isinstance(indices, tuple):
                indices = indices[0]
            for index in np.asarray(indices).reshape(-1).tolist():
                kept.append(group[int(index)])
        return sorted(kept, key=lambda d: d.score, reverse=True)
