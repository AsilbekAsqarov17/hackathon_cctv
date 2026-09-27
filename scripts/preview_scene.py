"""Draw a scene configuration over a representative video frame."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import load_scene_config


def _arrow(out, start, end, color, thickness=5, head=26):
    """Draw a direction arrow so lane heading is visible in the overlay."""
    (x1, y1), (x2, y2) = start, end
    cv2.arrowedLine(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, tipLength=0.35)


def draw(frame, config):
    out = frame.copy()

    def pts(polygon):
        return np.asarray([(int(x), int(y)) for x, y in polygon], dtype=np.int32)

    for road in config.road_polygons:
        cv2.polylines(out, [pts(road)], True, (0, 180, 255), 3)
    for crossing in config.crossings:
        cv2.polylines(out, [pts(crossing)], True, (0, 255, 255), 5)
    for line in config.stop_lines:
        cv2.line(out, pts(line.segment)[0], pts(line.segment)[1], (0, 0, 255), 6)
    for line in config.solid_lines:
        cv2.line(out, pts(line.segment)[0], pts(line.segment)[1], (255, 0, 0), 4)
    for light in config.traffic_lights:
        x1, y1, x2, y2 = [int(v) for v in light.roi]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(out, light.light_id, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for lane in config.lanes:
        polygon = pts(lane.polygon)
        cv2.polylines(out, [polygon], True, (255, 0, 255), 5)
        x, y = polygon[0]
        cv2.putText(out, f"L{lane.lane_id} {lane.name}", (x + 8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 3)
        # Direction arrow along the lane's own heading, anchored mid-lane.
        arr = np.asarray(lane.direction, dtype=np.float64)
        norm = float(np.hypot(arr[0], arr[1])) or 1.0
        ux, uy = arr[0] / norm, arr[1] / norm
        centre = polygon.reshape(-1, 2).mean(axis=0).astype(np.float64)
        half = 150.0
        _arrow(out, (centre[0] - ux * half, centre[1] - uy * half),
               (centre[0] + ux * half, centre[1] + uy * half), (255, 0, 255))
        if lane.approach_point is not None:
            ax, ay = lane.approach_point
            cv2.circle(out, (int(ax), int(ay)), 16, (0, 0, 255), -1)
            cv2.circle(out, (int(ax), int(ay)), 16, (255, 255, 255), 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("scene")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", default="scene_preview.jpg")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    h, w = frame.shape[:2]
    config = load_scene_config(args.scene, w, h)
    output = draw(frame, config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
