"""Report whether each geometry-dependent rule is *eligible* for a scene file.

A rule is eligible when its geometry preconditions are satisfied by the scene
file, i.e. it could fire on some track. It is not a claim that it does fire: that
needs real tracks. This is the check to run after authoring scene geometry.

Example::

    python scripts/check_rule_gating.py configs/scenes/data_video1.json
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import FrameState, SceneState, TrackState  # noqa: E402
from src.rules.engine import RuleEngine  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402

WIDTH, HEIGHT = 3840, 2160


def moving_vehicle(track_id: int = 1, y: float = 800.0) -> TrackState:
    history = deque([(i * 0.1, 900.0 + i * 30, y + i * 4) for i in range(60)], maxlen=300)
    bottom = deque([(i * 0.1, 900.0 + i * 30, y + 20) for i in range(60)], maxlen=300)
    return TrackState(
        track_id=track_id,
        class_id=2,
        class_name="car",
        bbox=(850.0, y - 100, 950.0, y + 20),
        score=0.9,
        first_seen=0.0,
        last_seen=6.0,
        center=(900.0, y - 40),
        velocity=(300.0, 33.0),
        history=history,
        bottom_history=bottom,
        last_seen_frame=60,
    )


def eligibility(scene: SceneContext) -> dict[str, tuple[bool, str]]:
    config = scene.config
    lanes_with_dir = [lane for lane in config.lanes if lane.direction != (0.0, 0.0)]
    approach = [lane for lane in lanes_with_dir if lane.approach_point is not None]
    return {
        "stopped_vehicle": (
            bool(scene.has_road),
            f"needs road area; has_road={scene.has_road}",
        ),
        "wrong_way": (
            bool(scene.has_geometry and lanes_with_dir),
            f"needs geometry + lane directions; lanes={len(lanes_with_dir)}",
        ),
        "solid_line_crossing": (
            bool((scene.has_road or scene.has_geometry) and config.solid_lines),
            f"needs road/geometry + solid lines; solid_lines={len(config.solid_lines)}",
        ),
        "red_light": (
            bool(
                scene.has_geometry
                and config.stop_lines
                and config.traffic_lights
                and approach
            ),
            (
                f"needs geometry + stop_lines({len(config.stop_lines)}) + "
                f"traffic_lights({len(config.traffic_lights)}) + lane approach_point({len(approach)})"
            ),
        ),
    }


def main() -> int:
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "configs/scenes/data_video1.json"
    config = load_scene_config(scene_path, WIDTH, HEIGHT)
    scene = SceneContext(config, WIDTH, HEIGHT)
    print(f"scene file    : {scene_path}")
    print(f"scene_id      : {scene.scene_id}")
    print(f"has_geometry  : {scene.has_geometry}")
    print(f"has_road      : {scene.has_road}")
    print(
        "collections   : "
        f"lanes={len(config.lanes)} stop_lines={len(config.stop_lines)} "
        f"solid_lines={len(config.solid_lines)} crossings={len(config.crossings)} "
        f"traffic_lights={len(config.traffic_lights)} road_polygons={len(config.road_polygons)}"
    )

    # The engine must also load and evaluate without raising.
    engine = RuleEngine({"rules": {}})
    state = FrameState(
        frame_id=60,
        timestamp=6.0,
        tracks=[moving_vehicle()],
        scene=SceneState(scene.scene_id, WIDTH, HEIGHT, {}, scene),
    )
    signals = engine.evaluate(state)
    print(f"engine evaluated {len(signals)} rules without error")

    print("\nrule eligibility:")
    all_eligible = True
    for label, (ok, reason) in eligibility(scene).items():
        all_eligible = all_eligible and ok
        signal = signals.get(label)
        fired = bool(getattr(signal, "active", False))
        print(f"  {label:22} eligible={ok!s:5} fires_on_probe={fired!s:5}  ({reason})")
    print(f"\nall four target rules eligible: {all_eligible}")
    return 0 if all_eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
