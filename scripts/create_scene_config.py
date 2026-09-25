"""Create a blank camera scene configuration for manual calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--scene-id", default="camera_01")
    parser.add_argument("--normalized", action="store_true")
    args = parser.parse_args()
    data = {
        "scene_id": args.scene_id,
        "normalized": args.normalized,
        "road_polygon": [],
        "lanes": [],
        "stop_lines": [],
        "solid_lines": [],
        "crosswalks": [],
        "traffic_lights": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
