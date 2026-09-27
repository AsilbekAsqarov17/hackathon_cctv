"""Check that one authored scene maps consistently onto different resolutions.

The competition camera delivers the same intersection at more than one frame
size (data_video1 is 3840x2160, data_video2 is 1920x1080). A single authored
scene must therefore resolve the same geometry at both, or every geometry rule
silently breaks on the smaller videos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene.config import SceneContext, load_scene_config  # noqa: E402

# Points that were individually verified in 4K during the geometry milestone:
# (x, y, expected_lane_id_or_None, what it is)
PROBES = [
    (2336, 1206, 2, "eastbound car in E_inner"),
    (903, 760, 3, "queued bus in E_outer"),
    (2000, 674, 0, "W_outer mid-band"),
    (2000, 822, 1, "W_inner mid-band"),
    (1400, 825, 2, "E_inner mid-band"),
    (1400, 1063, 3, "E_outer mid-band"),
    (3529, 934, None, "east leg, deliberately no direction"),
    (3737, 2124, None, "lower-right junction, no direction"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--sizes", nargs="+", default=["3840x2160", "1920x1080"])
    args = parser.parse_args()

    scenes = []
    for size in args.sizes:
        w, h = (int(v) for v in size.lower().split("x"))
        config = load_scene_config(args.scene, w, h)
        scenes.append((w, h, SceneContext(config, w, h)))

    ref_w, ref_h, ref = scenes[0]
    print(f"reference {ref_w}x{ref_h}  ({args.scene})")
    print(f"{'probe (4K)':>16} {'expect':>7} " + " ".join(f"{w}x{h}".rjust(11) for w, h, _ in scenes[1:]))
    print("-" * 90)
    ok = True
    for x, y, expected, note in PROBES:
        base = ref.lane_for_point((x, y))
        got = base.lane_id if base else None
        if got != expected:
            ok = False
        cells = []
        for w, h, scene in scenes[1:]:
            sx, sy = w / ref_w, h / ref_h
            other = scene.lane_for_point((x * sx, y * sy))
            other_id = other.lane_id if other else None
            cells.append(f"{str(other_id):>11}")
            if other_id != got:
                ok = False
        flag = "" if got == expected else "  <-- UNEXPECTED at reference"
        print(f"{f'({x},{y})':>16} {str(got):>7} " + " ".join(cells) + flag + f"   {note}")

    print()
    for w, h, scene in scenes[1:]:
        line = scene.config.stop_lines[0]
        light = scene.config.traffic_lights[0]
        print(f"{w}x{h}: stop_line {line.segment[0]}..{line.segment[1]} dir {line.direction}")
        print(f"{w}x{h}: {light.light_id} roi {tuple(round(v,1) for v in light.roi)}")
    print("\nCONSISTENT" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
