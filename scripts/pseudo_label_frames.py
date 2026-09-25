"""Generate initial YOLO pseudo-labels for development frames.

These labels are a starting point for review, not final ground truth. Manually
correct or delete labels before using them for a reported fine-tuning result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Keep only road-user/event-relevant COCO classes and remap to contiguous IDs.
CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 9: 6, 13: 7}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/traffic_coco/images/development")
    parser.add_argument("--output", default="data/traffic_coco/labels/development")
    parser.add_argument("--model", default="weights/yolo11n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    image_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    model = YOLO(args.model)
    total = 0
    for index, image in enumerate(images, 1):
        result = model.predict(str(image), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        lines = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            h, w = result.orig_shape
            for coords, score, class_id in zip(xyxy, confs, classes):
                original_id = int(class_id)
                if original_id not in CLASS_MAP:
                    continue
                x1, y1, x2, y2 = [float(v) for v in coords]
                mapped_id = CLASS_MAP[original_id]
                line = f"{mapped_id} {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                lines.append(line)
                total += 1
        (output_dir / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if index % 10 == 0 or index == len(images):
            print(f"{index}/{len(images)} frames, {total} boxes")
    print(f"labels written to {output_dir}; pseudo-label count={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
