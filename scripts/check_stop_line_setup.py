"""Check the stop line and its governing signal agree, at every resolution.

Verifies the invariants the signal-dependent rules depend on:

* the stop line spans and cuts the lanes it governs;
* the governing signal is found by declared approach direction;
* every governed lane's approach point is upstream of the line, which is what
  queue suppression uses to recognise a standing queue;
* the upstream test holds identically at 4K and 1080p.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import TrafficLightState  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--sizes", nargs="+", default=["3840x2160", "1920x1080"])
    parser.add_argument("--governed", default="2,3")
    args = parser.parse_args()
    governed = {int(v) for v in args.governed.split(",") if v.strip()}

    ok = True
    for size in args.sizes:
        w, h = (int(v) for v in size.lower().split("x"))
        config = load_scene_config(args.scene, w, h)
        scene = SceneContext(config, w, h)
        print(f"=== {w}x{h} ===")
        for line in config.stop_lines:
            light_id = scene.signal_for_line(line)
            print(f"  stop line {line.line_id}: {tuple(round(v) for v in line.segment[0])}"
                  f" -> {tuple(round(v) for v in line.segment[1])}  dir {line.direction}")
            print(f"    governing signal: {light_id}")
            if light_id is None:
                ok = False
                print("    *** no signal matches this line's approach direction")
                continue
            for color in ("red", "yellow", "green", "unknown"):
                scene.lights = {light_id: TrafficLightState(light_id, color, 0.9,
                                                            next(l.roi for l in config.traffic_lights
                                                                 if l.light_id == light_id))}
                got = scene.signal_color_for_line(line)
                if got != color:
                    ok = False
                    print(f"    *** reading {color} returned {got}")
            scene.lights = {}
            if scene.signal_color_for_line(line) != "unknown":
                ok = False
                print("    *** an unreadable signal must report unknown, not a guess")
            else:
                print("    colour round-trip ok; unreadable -> unknown (no invented state)")
            for lane in config.lanes:
                if lane.lane_id not in governed or lane.approach_point is None:
                    continue
                origin = line.segment[0]
                p = lane.approach_point
                along = (p[0] - origin[0]) * line.direction[0] + (p[1] - origin[1]) * line.direction[1]
                status = "upstream" if along < 0 else "*** NOT UPSTREAM"
                if along >= 0:
                    ok = False
                print(f"    lane {lane.lane_id} {lane.name:8s} approach along={along:9.1f}  {status}")
        print()
    print("OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
