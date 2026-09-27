"""Verify that ONNX Runtime actually initialises CUDA on this machine.

Run this on a single still frame before any video processing. It fails loudly
if ONNX Runtime silently falls back to CPU.

    python scripts/onnx_gpu_smoke.py
    python scripts/onnx_gpu_smoke.py --clip data/inspection/clip_5s.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def load_frame(image: str | None, clip: str | None, clip_seconds: float) -> tuple[np.ndarray, str]:
    if clip:
        capture = cv2.VideoCapture(clip)
        if not capture.isOpened():
            raise SystemExit(f"could not open clip {clip}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(int(fps * 0.5), max(0, total - 1)))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise SystemExit(f"could not read a frame from {clip}")
        return frame, f"{clip} @ frame {int(fps * 0.5)}"
    picture = cv2.imread(image or "")
    if picture is None:
        raise SystemExit(f"could not read image {image}")
    return picture, image or ""


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)[None] / 255.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/yolo11n.onnx")
    parser.add_argument("--image", default="data/inspection/frame_5000.jpg")
    parser.add_argument("--clip", default=None)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    import onnxruntime as ort

    print(f"onnxruntime {ort.__version__}")
    print("available_providers:", ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("CUDAExecutionProvider is not compiled into this build")

    # Same DLL setup the detector adapter uses.
    from src.perception.onnx_detector import _configure_cuda_dlls

    _configure_cuda_dlls(ort)
    print("dll search path: configured")
    if hasattr(ort, "print_debug_info"):
        ort.print_debug_info()

    providers: list = [("CUDAExecutionProvider", {"device_id": args.device_id}), "CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    active = session.get_providers()
    print("session_providers:", active)

    if active[0] != "CUDAExecutionProvider":
        print("FAIL: ONNX Runtime fell back to CPU; refusing to continue.")
        return 1

    image, origin = load_frame(args.image, args.clip, args.clip_seconds)
    tensor = letterbox(image, 640)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    session.run([out.name], {inp.name: tensor})  # warm-up
    start = time.perf_counter()
    runs = 5
    for _ in range(runs):
        result = session.run([out.name], {inp.name: tensor})[0]
    elapsed = (time.perf_counter() - start) / runs

    print(
        {
            "status": "PASS",
            "device": active[0],
            "device_id": args.device_id,
            "source": origin,
            "output_shape": list(result.shape),
            "mean_latency_ms": round(elapsed * 1000, 2),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
