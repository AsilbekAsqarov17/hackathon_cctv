"""Render the directional lane geometry over real frames for visual checking.

Draws each calibrated lane as a translucent band with its id, name and
direction vector, plus the stop line and signal ROIs, so the recalibrated
geometry can be checked against the actual road rather than against numbers.

Example::

    python scripts/preview_directional_lanes.py data/data_video1.mp4 ^
        --scene configs/scenes/data_video1.json --times 5 40 120 200 290
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector  # noqa: E402

BGR = {
    0: (255, 120, 0),    # westbound outer  - blue-ish
    1: (255, 170, 60),   # westbound inner  - lighter blue
    2: (60, 190, 255),   # eastbound inner  - amber
    3: (40, 140, 255),   # eastbound outer  - deeper amber
}


def band_mask(polygon: list, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--out", default="data/authoring/directional_lanes")
    args = parser.parse_args()

    scene = SceneContext(load_scene_config(args.scene, 3840, 2160), 3840, 2160)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    wanted = {round(t * args.fps): t for t in args.times}
    written: dict[float, str] = {}
    index = 0
    while len(written) < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            canvas = frame.copy()
            shape = frame.shape[:2]
            for lane in scene.config.lanes:
                colour = BGR.get(lane.lane_id, (200, 200, 200))
                mask = band_mask(lane.polygon, shape)
                tint = canvas.copy()
                tint[mask == 1] = colour
                canvas = cv2.addWeighted(tint, 0.28, canvas, 0.72, 0)
                cv2.polylines(canvas, [np.array(lane.polygon, np.int32)], True, colour, 4)
                unit = normalized_vector(lane.direction) or (1.0, 0.0)
                cx = int(np.mean([p[0] for p in lane.polygon]))
                cy = int(np.mean([p[1] for p in lane.polygon]))
                tip = (int(cx + unit[0] * 320), int(cy + unit[1] * 320))
                cv2.arrowedLine(canvas, (cx, cy), tip, colour, 8, tipLength=0.25)
                label = f"{lane.lane_id} {lane.name}  dir=({lane.direction[0]:+.2f},{lane.direction[1]:+.2f})"
                cv2.putText(canvas, label, (cx - 210, cy - 26), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 0, 0), 6)
                cv2.putText(canvas, label, (cx - 210, cy - 26), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, colour, 2)
            for line in scene.config.stop_lines:
                (ax, ay), (bx, by) = line.segment
                cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (0, 0, 255), 6)
                cv2.putText(canvas, f"stop {line.line_id}", (int(ax) - 40, int(ay) - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            for light in scene.config.traffic_lights:
                x0, y0, x1, y1 = (int(v) for v in light.roi)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 4)
                cv2.putText(canvas, light.light_id, (x0 - 20, y0 - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(canvas, f"t={wanted[index]:.1f}s frame={index}", (24, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
            cv2.putText(canvas, f"t={wanted[index]:.1f}s frame={index}", (24, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
            small = cv2.resize(canvas, (1600, 900), interpolation=cv2.INTER_AREA)
            path = out_dir / f"lanes_{wanted[index]:07.1f}.jpg"
            cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written[wanted[index]] = str(path)
        index += 1
    capture.release()
    for t in sorted(written):
        print(f"t={t:7.1f}s  {written[t]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
