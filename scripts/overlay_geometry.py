"""Render the full scene geometry over a frame of any sample video.

Works at any resolution: the scene is authored once at 3840x2160 and mapped onto
whatever frame size the video has, so the same overlay tool checks both sample
videos and shows whether a rule's geometry actually lands on the road.

Example::

    python scripts/overlay_geometry.py data/data_video2.mp4 --times 40 120 --lane-width 3
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

LANE_COLOURS = [(255, 120, 0), (255, 170, 60), (60, 190, 255), (40, 140, 255), (200, 200, 255), (0, 255, 255)]
ROAD_COLOUR = (0, 90, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--out", default="data/authoring/geometry_overlay")
    parser.add_argument("--scale", type=float, default=1.0, help="output downscale")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--lane-width", type=int, default=2)
    parser.add_argument("--rect", nargs=4, type=int, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="crop the frame to this region before drawing")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    width = args.width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(round(width * capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / max(1, width)))
    scene = SceneContext(load_scene_config(args.scene, width, height), width, height)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {round(t * args.fps): t for t in args.times}
    written: dict[float, str] = {}
    index = 0
    lw = max(1, int(round(args.lane_width * width / 3840)))
    while len(written) < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            # Draw in full-frame coordinates, then crop, so scene geometry and
            # rule annotations never need offsetting for a zoomed view.
            canvas = frame.copy()
            overlay = canvas.copy()
            for poly in scene.config.road_polygons:
                cv2.fillPoly(overlay, [np.array(poly, np.int32)], ROAD_COLOUR)
            canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
            for poly in scene.config.road_polygons:
                cv2.polylines(canvas, [np.array(poly, np.int32)], True, (0, 160, 0), lw)
            for lane in scene.config.lanes:
                colour = LANE_COLOURS[lane.lane_id % len(LANE_COLOURS)]
                pts = np.array(lane.polygon, np.int32)
                cv2.polylines(canvas, [pts], True, colour, lw + 1)
                unit = normalized_vector(lane.direction) or (1.0, 0.0)
                cx = int(np.mean([p[0] for p in lane.polygon]))
                cy = int(np.mean([p[1] for p in lane.polygon]))
                tip = (int(cx + unit[0] * width * 0.09), int(cy + unit[1] * width * 0.09))
                cv2.arrowedLine(canvas, (cx, cy), tip, colour, lw + 2, tipLength=0.25)
                tag = f"{lane.lane_id}:{lane.name}"
                cv2.putText(canvas, tag, (cx - 40, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 + width / 3000, (0, 0, 0), lw + 3)
                cv2.putText(canvas, tag, (cx - 40, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 + width / 3000, colour, lw)
            for line in scene.config.stop_lines:
                (ax, ay), (bx, by) = line.segment
                cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (0, 0, 255), lw + 2)
                cv2.putText(canvas, f"STOP {line.line_id}", (int(ax) - 20, int(ay) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 + width / 3000, (0, 0, 255), lw + 1)
            for poly in scene.config.crossings:
                pts = np.array(poly, np.int32)
                cv2.polylines(canvas, [pts], True, (255, 255, 255), lw)
            for light in scene.config.traffic_lights:
                x0, y0, x1, y1 = (int(v) for v in light.roi)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), lw + 1)
                cv2.putText(canvas, light.light_id, (x0 - 10, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 + width / 3400, (0, 255, 255), lw)
            label = f"{Path(args.video).name} t={wanted[index]:.1f}s {width}x{height} f={index}"
            cv2.putText(canvas, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 + width / 2600, (0, 0, 0), lw + 4)
            cv2.putText(canvas, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 + width / 2600, (255, 255, 255), lw + 1)
            if args.rect:
                rx0, ry0, rx1, ry1 = args.rect
                canvas = canvas[ry0:ry1, rx0:rx1]
            if args.scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale)
            path = out_dir / f"{Path(args.video).stem}_{wanted[index]:07.1f}.jpg"
            cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written[wanted[index]] = str(path)
        index += 1
    capture.release()
    for t in sorted(written):
        print(f"t={t:7.1f}s  {written[t]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
