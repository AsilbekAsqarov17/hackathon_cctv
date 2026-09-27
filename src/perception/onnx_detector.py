from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _configure_cuda_dlls(ort: Any) -> None:
    """Locate the pip-installed NVIDIA CUDA/cuDNN DLLs before creating a session.

    Two things are needed on Windows. First, the `nvidia\\*\\bin` directories from
    the `nvidia-*-cu12` wheels must be on the DLL search path; without this,
    cuBLAS emits "Could not locate nvrtc64_120_0.dll" while probing for an
    optional JIT path. Second, ONNX Runtime >= 1.21 exposes `preload_dlls`,
    which resolves the CUDA/cuDNN/MSVC runtime dependencies.

    Note `site.getsitepackages()` returns every candidate root, and the first
    entry is the interpreter prefix rather than `Lib\\site-packages`, so all
    entries must be searched.
    """
    import glob
    import os
    import site

    roots = set(site.getsitepackages())
    usersite = getattr(site, "getusersitepackages", None)
    if usersite is not None:
        roots.add(usersite())
    for root in roots:
        for directory in sorted(glob.glob(os.path.join(root, "nvidia", "*", "bin"))):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(directory)
                except OSError:
                    pass
            if directory not in os.environ.get("PATH", ""):
                os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]

    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload()


from ..contracts import Detection
from .detector import COCO_NAMES, RELEVANT_NAMES


class OnnxDetector:
    """YOLO ONNX detector using CUDAExecutionProvider when available."""

    _sessions: dict[tuple[str, int, str], Any] = {}

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_path = Path(str(config.get("model", "yolo11n.onnx")))
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.imgsz = int(config.get("imgsz", 640))
        self.confidence = float(config.get("confidence", 0.25))
        self.iou = float(config.get("iou", 0.5))
        self.device = str(config.get("device", "0"))
        self.class_names = self._configured_names()
        self.session = self._get_session()

    def _configured_names(self) -> dict[int, str]:
        raw = self.config.get("class_names")
        if isinstance(raw, list):
            return {i: str(name) for i, name in enumerate(raw)}
        return dict(COCO_NAMES)

    def _get_session(self) -> Any:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.device not in {"cpu", "-1"}:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                device_id = int(self.device) if self.device.isdigit() else 0
                providers = [
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                    "CPUExecutionProvider",
                ]
        # Must happen before InferenceSession so the loader finds cuDNN/cuBLAS.
        _configure_cuda_dlls(ort)
        key = (str(self.model_path.resolve()), self.imgsz, self.device)
        if key not in self._sessions:
            self._sessions[key] = ort.InferenceSession(
                str(self.model_path), providers=providers
            )
        session = self._sessions[key]
        self.active_providers = session.get_providers()
        if self.device not in {"cpu", "-1"} and "CUDAExecutionProvider" not in self.active_providers:
            raise RuntimeError(
                f"CUDA device requested but ONNX Runtime providers are {self.active_providers}"
            )
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
