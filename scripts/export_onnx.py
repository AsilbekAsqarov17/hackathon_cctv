"""Export a trained Ultralytics checkpoint to ONNX for GPU inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/yolo11n.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    model = YOLO(args.model)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, device="cpu")
    if args.output:
        source = Path(exported)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
        print(target)
    else:
        print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
