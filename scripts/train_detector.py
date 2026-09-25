"""Fine-tune the optional Part A detector on a local dataset.

This script is intentionally separate from inference. It is not called by the
submission harness and does not affect Part B.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="traffic_part_a")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-part-a.txt before fine-tuning") from exc

    model = YOLO(args.model)
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        kwargs["device"] = args.device
    model.train(**kwargs)
    print(f"weights saved under {Path(args.project) / args.name / 'weights'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
