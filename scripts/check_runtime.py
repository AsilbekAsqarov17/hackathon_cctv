"""Check the Part A runtime and report missing optional components."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    model = os.getenv("TRAFFIC_YOLO_MODEL", "yolo11n.pt")
    model_path = Path(model)
    if not model_path.is_absolute():
        candidates = [Path.cwd() / model_path, root / model_path, root / "weights" / model_path]
        model_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    print(f"project: {root}")
    print(f"numpy: {bool(importlib.util.find_spec('numpy'))}")
    print(f"opencv: {bool(importlib.util.find_spec('cv2'))}")
    print(f"ultralytics: {bool(importlib.util.find_spec('ultralytics'))}")
    print(f"torch: {bool(importlib.util.find_spec('torch'))}")
    print(f"scipy: {bool(importlib.util.find_spec('scipy'))}")
    print(f"lap: {bool(importlib.util.find_spec('lap'))}")
    print(f"cython_bbox: {bool(importlib.util.find_spec('cython_bbox'))}")
    print(f"model: {model_path} ({'found' if model_path.exists() else 'missing'})")
    print(f"scene default: {(root / 'configs/scenes/default.json').exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
