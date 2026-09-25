"""Run one YOLO frame on CUDA device 0."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/inspection/frame_5000.jpg")
    parser.add_argument("--model", default="weights/yolo11n.pt")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run scripts/setup_cuda.ps1 first")
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read {args.image}")
    model = YOLO(args.model)
    result = model.predict(image, device=args.device, imgsz=640, conf=0.2, verbose=False)[0]
    print({"device": torch.cuda.get_device_name(0), "detections": len(result.boxes) if result.boxes is not None else 0, "image": str(Path(args.image))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
