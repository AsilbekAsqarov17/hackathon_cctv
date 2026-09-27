"""Prove wrong_way still fires on a real violation in the recalibrated lanes.

Zero detections on clean traffic is only reassuring if the rule is actually
capable of firing in this geometry. This builds synthetic vehicles that follow
a calibrated lane's own centreline, once in the permitted direction (control,
must stay silent) and once directly against it (must fire), and runs them
through the real RuleEngine with the real scene.

Nothing here is part of the submitted pipeline; it is a validation harness.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import FrameState, SceneState, TrafficLightState, TrackState  # noqa: E402
from src.rules.engine import RuleEngine  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.geometry import normalized_vector  # noqa: E402


def lane_centre_points(scene: SceneContext, lane_id: int, count: int = 120) -> list[tuple[float, float]]:
    """Densified mid-axis of a lane polygon, ordered left to right.

    The calibrated lanes are diagonal bands written as an upper edge left to
    right followed by the lower edge right to left, so the mid-axis is the
    midpoint of ``polygon[i]`` and ``polygon[-1 - i]``.
    """
    lane = next(l for l in scene.config.lanes if l.lane_id == lane_id)
    poly = lane.polygon
    half = len(poly) // 2
    mid = [
        ((poly[i][0] + poly[len(poly) - 1 - i][0]) / 2.0,
         (poly[i][1] + poly[len(poly) - 1 - i][1]) / 2.0)
        for i in range(half)
    ]
    mid.sort(key=lambda p: p[0])
    if len(mid) < 2:
        return mid
    out: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(mid, mid[1:]):
        steps = max(1, (count - 1) // (len(mid) - 1))
        for s in range(steps):
            f = s / steps
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    out.append(mid[-1])
    return out


def build_track(track_id: int, path: list[tuple[float, float]], fps: float) -> TrackState:
    first_t = 0.0
    last_t = (len(path) - 1) / fps
    speed_vec = (
        (path[min(1, len(path) - 1)][0] - path[0][0]) * fps,
        (path[min(1, len(path) - 1)][1] - path[0][1]) * fps,
    )
    history = deque(
        [(i / fps, p[0], p[1]) for i, p in enumerate(path)], maxlen=300
    )
    return TrackState(
        track_id=track_id,
        class_id=2,
        class_name="car",
        bbox=(path[-1][0] - 40.0, path[-1][1] - 40.0, path[-1][0] + 40.0, path[-1][1]),
        score=0.9,
        first_seen=first_t,
        last_seen=last_t,
        center=path[-1],
        velocity=speed_vec,
        bottom_history=history,
        history=deque(history, maxlen=300),
    )


def run(engine: RuleEngine, scene: SceneContext, path, fps: float, label: str) -> dict:
    state: dict[int, TrackState] = {}
    fired: list[float] = []
    for i, _ in enumerate(path):
        t = i / fps
        track = build_track(1, path[: i + 1], fps)
        state[1] = track
        lights = {
            light.light_id: TrafficLightState(light.light_id, "green", 1.0, light.roi)
            for light in scene.config.traffic_lights
        }
        frame = FrameState(
            i, t, [track], SceneState(scene.scene_id, scene.width, scene.height, lights, scene)
        )
        signal = engine.evaluate(frame)["wrong_way"]
        if signal.active:
            fired.append(t)
    if not fired:
        print(f"  {label}: no event")
        return {"fired": False}
    print(f"  {label}: FIRED at t={fired[0]:.2f}..{fired[-1]:.2f} "
          f"({len(fired)} frames, {fired[-1] - fired[0]:.2f}s span)")
    return {"fired": True, "first": fired[0], "last": fired[-1], "frames": len(fired)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="configs/scenes/data_video1.json")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--count", type=int, default=120)
    args = parser.parse_args()

    scene = SceneContext(load_scene_config(args.scene, args.width, args.height),
                         args.width, args.height)
    cfg = json.load(open(args.config, encoding="utf-8"))["rules"]["wrong_way"]
    print(f"wrong_way config: {({k: v for k, v in cfg.items() if not k.startswith('_')})}\n")

    ok = True
    for lane in scene.config.lanes:
        path = lane_centre_points(scene, lane.lane_id, args.count)
        lane_obj = next(l for l in scene.config.lanes if l.lane_id == lane.lane_id)
        unit = normalized_vector(lane_obj.direction)
        # Permitted direction: walk the centreline along the lane direction.
        forward = path if unit[0] >= 0 else list(reversed(path))
        # Against the direction: same corridor, opposite sense.
        backward = list(reversed(forward))
        print(f"lane {lane.lane_id} {lane.name} dir={lane_obj.direction}")
        # Sanity: both paths must actually be inside the lane.
        for name, p in (("forward", forward[len(forward) // 2]), ("backward", backward[len(backward) // 2])):
            got = scene.lane_for_point(p)
            inside = got is not None and got.lane_id == lane.lane_id
            print(f"    {name} mid {p[0]:.0f},{p[1]:.0f} -> lane "
                  f"{got.lane_id if got else None} ({'inside' if inside else 'OUTSIDE'})")
            if not inside:
                ok = False
        a = run(RuleEngine({"wrong_way": cfg}), scene, forward, args.fps, "with-flow (must be silent)")
        b = run(RuleEngine({"wrong_way": cfg}), scene, backward, args.fps, "against-flow (must fire)")
        if a["fired"] or not b["fired"]:
            ok = False
            print("    *** UNEXPECTED ***")
        print()

    print("ALL LANES BEHAVE CORRECTLY" if ok else "SOME LANES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
