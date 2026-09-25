"""Compute a homography for a fixed-camera scene JSON file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.geometry import compute_homography


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--image-points", required=True, help="JSON list of [x,y] image points")
    parser.add_argument("--world-points", required=True, help="JSON list of [x,y] world points in metres")
    args = parser.parse_args()
    path = Path(args.scene)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    data["homography"] = compute_homography(
        json.loads(args.image_points), json.loads(args.world_points)
    )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
