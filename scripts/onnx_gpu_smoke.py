"""Verify ONNX Runtime CUDA provider on one image or a short clip."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/yolo11n.onnx")
    parser.add_argument("--image", default="data/inspection/frame_5000.jpg")
    parser.add_argument("--clip", default=None, help="optional short .mp4; only a few seconds are allowed")
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    args = parser.parse_args()
    available = ort.get_available_providers()
    print("available_providers:", available)
    if "CUDAExecutionProvider" not in available:
        raise SystemExit("CUDAExecutionProvider is not available")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    active = session.get_providers()
    print("session_providers:", active)
    if active[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CUDA provider fell back: {active}")
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read {args.image}")
    if args.clip:
        cap = cv2.VideoCapture(args.clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(fps * 0.5), max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)))
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read clip")
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)[None] / 255.0
    start = time.perf_counter()
    result = session.run([out.name], {inp.name: tensor})[0]
    elapsed = time.perf_counter() - start
    print({"input": str(Path(args.image if not args.clip else args.clip)), "output_shape": list(result.shape), "seconds": round(elapsed, 4), "device": active[0]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
