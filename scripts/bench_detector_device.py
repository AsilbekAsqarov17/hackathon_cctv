"""Compare GPU and CPU inference throughput for the base and fine-tuned detectors."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

IMGSZ = 640
WARMUP = 3
RUNS = 40

MODELS = {
    "base (yolo11n.pt)": "weights/yolo11n.pt",
    "finetuned (best.pt)": "runs/detect/runs/detect/traffic_part_a_gpu/weights/best.pt",
}


def bench(model_path: str, device: str, frames: list[np.ndarray]) -> float:
    model = YOLO(model_path)
    for _ in range(WARMUP):
        model.predict(frames[0], imgsz=IMGSZ, device=device, verbose=False)
    start = time.perf_counter()
    for frame in frames:
        model.predict(frame, imgsz=IMGSZ, device=device, verbose=False)
    elapsed = time.perf_counter() - start
    return elapsed / len(frames) * 1000.0


def main() -> int:
    import cv2

    images = sorted(Path("data/traffic_coco/images/val").glob("*.jpg"))[:RUNS]
    frames = [cv2.imread(str(p)) for p in images]
    print(f"benchmark: {len(frames)} frames, imgsz={IMGSZ}\n")
    header = f"{'model':<24}{'CPU ms/img':>14}{'GPU ms/img':>14}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for label, path in MODELS.items():
        cpu_ms = bench(path, "cpu", frames)
        gpu_ms = bench(path, "0", frames)
        print(f"{label:<24}{cpu_ms:>14.1f}{gpu_ms:>14.1f}{cpu_ms / gpu_ms:>9.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
