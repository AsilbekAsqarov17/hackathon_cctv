"""Evaluate detector outputs on sampled development frames."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/traffic_coco/images/development")
    parser.add_argument("--model", default="weights/yolo11n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--output", default="data/inspection/detector_eval.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit > 0:
        images = images[: args.limit]
    model = YOLO(args.model)
    counts = Counter()
    confidence_sum = Counter()
    per_frame = []
    for image in images:
        result = model.predict(str(image), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        frame_counts = Counter()
        if boxes is not None:
            classes = boxes.cls.cpu().numpy().astype(int)
            scores = boxes.conf.cpu().numpy()
            for class_id, score in zip(classes, scores):
                name = model.names.get(int(class_id), str(class_id))
                counts[name] += 1
                confidence_sum[name] += float(score)
                frame_counts[name] += 1
        per_frame.append({"image": image.name, "counts": dict(frame_counts)})
    report = {
        "model": args.model,
        "images": len(images),
        "counts": dict(counts),
        "mean_confidence": {k: confidence_sum[k] / counts[k] for k in counts},
        "per_frame": per_frame,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "images", "counts", "mean_confidence")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
