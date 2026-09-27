"""Run the stopped_vehicle + wrong_way milestone over a video.

Detector and tracker are unchanged from the perception milestone. This script
adds the two rules, the signal-state channel the rules need for queue
suppression, a debug renderer and development JSONL output.

Only ``stopped_vehicle`` and ``wrong_way`` are enabled (see the config); every
other rule is explicitly disabled and is not evaluated for events.

Example::

    python scripts/run_rules.py --video data/data_video1.mp4 ^
        --config configs/data_video1_rules_dev.json ^
        --out-video debug/rules/rules_debug.mp4 ^
        --out-jsonl debug/rules/events.jsonl ^
        --out-events debug/rules/events.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.contracts import FrameState, SceneState  # noqa: E402
from src.perception.detector import build_detector  # noqa: E402
from src.perception.tracker import build_tracker  # noqa: E402
from src.perception.traffic_lights import TrafficLightPerception, is_traffic_light  # noqa: E402
from src.rules.engine import RuleEngine  # noqa: E402
from src.rules.motion import is_stationary, window_heading, window_speed  # noqa: E402
from src.scene.config import SceneContext, load_scene_config  # noqa: E402
from src.scene.signals import TrafficLightReader  # noqa: E402
from src.temporal.segmenter import TemporalSegmenter  # noqa: E402
from src.tracking.track_manager import TrackManager  # noqa: E402

# Debug palette (BGR).
COLOR_STOPPED = (0, 140, 255)      # amber: event is firing
COLOR_WRONG = (0, 0, 255)          # red: wrong way
COLOR_TRACK = (200, 200, 200)      # grey: plain track
COLOR_QUEUED = (190, 160, 120)     # tan: stationary but suppressed as a signal queue
COLOR_EXPECTED = (0, 255, 255)     # cyan arrow: lane direction
COLOR_OBSERVED = (0, 255, 0)       # green arrow: measured direction
COLOR_MARK = (0, 255, 255)         # event start/end marker
TRAIL = 40


def arrow(canvas, origin, direction, length, color, thickness=4) -> None:
    ox, oy = origin
    dx, dy = direction
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return
    ux, uy = dx / norm, dy / norm
    tip = (int(ox + ux * length), int(oy + uy * length))
    cv2.arrowedLine(canvas, (int(ox), int(oy)), tip, color, thickness, tipLength=0.3)


def draw_debug(
    canvas,
    states,
    scene: SceneContext,
    per_track: dict[int, dict[str, Any]],
    timestamp: float,
    scale: float,
    lights: dict[str, Any],
    markers: list[dict[str, Any]] | None = None,
) -> None:
    def sx(v: float) -> int:
        return int(round(v * scale))

    def sy(v: float) -> int:
        return int(round(v * scale))

    for state in states:
        info = per_track.get(state.track_id, {})
        stopped = info.get("stopped_active", False)
        wrong = info.get("wrong_active", False)
        queued = info.get("queued", False)
        if stopped:
            color = COLOR_STOPPED
        elif wrong:
            color = COLOR_WRONG
        elif queued:
            color = COLOR_QUEUED
        else:
            color = COLOR_TRACK
        x1, y1, x2, y2 = (sx(v) for v in state.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(1, int(3 * scale)))

        trail = list(state.bottom_history)[-TRAIL:]
        for i in range(1, len(trail)):
            p0 = (sx(trail[i - 1][1]), sy(trail[i - 1][2]))
            p1 = (sx(trail[i][1]), sy(trail[i][2]))
            fade = 0.35 + 0.65 * (i / max(1, len(trail) - 1))
            cv2.line(canvas, p0, p1, tuple(int(c * fade) for c in color), max(1, int(3 * scale)))

        bx, by = sx(state.bottom_center[0]), sy(state.bottom_center[1])
        radius = max(3, int(6 * scale))
        cv2.circle(canvas, (bx, by), radius, (0, 255, 255), -1)

        lane = scene.lane_for_point(state.bottom_center)
        lane_id = lane.lane_id if lane else None
        speed = info.get("speed", 0.0)
        label = f"#{state.track_id} {state.class_name} L{lane_id} {speed:.0f}px/s"
        ty = max(int(22 * scale), y1 - int(10 * scale))
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 0), 5)
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, color, 2)

        observed = info.get("observed")
        if observed:
            arrow(canvas, (bx, by), observed, int(70 * scale), COLOR_OBSERVED, max(1, int(3 * scale)))
        if lane is not None:
            arrow(canvas, (bx, by), lane.direction, int(70 * scale), COLOR_EXPECTED, max(1, int(2 * scale)))

        line_y = ty + int(18 * scale)
        since = info.get("stopped_since")
        if since is not None:
            held = timestamp - since
            text = f"STOPPED {held:5.1f}s" + ("  EVENT" if stopped else "")
            cv2.putText(canvas, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (0, 0, 0), 5)
            cv2.putText(canvas, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, COLOR_STOPPED, 2)
            line_y += int(18 * scale)
        elif queued:
            held = info.get("queued_for", 0.0)
            text = f"QUEUE {held:5.1f}s (suppressed)"
            cv2.putText(canvas, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (0, 0, 0), 5)
            cv2.putText(canvas, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, COLOR_QUEUED, 2)
            line_y += int(18 * scale)
        if wrong:
            text = f"WRONG WAY dot={info.get('dot', 0.0):.2f}"
            cv2.putText(canvas, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (0, 0, 0), 5)
            cv2.putText(canvas, text, (x1, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, COLOR_WRONG, 2)

        # event start / end marker on the box corner
        for marker in markers or []:
            if marker.get("track_id") != state.track_id:
                continue
            if abs(timestamp - marker["t"]) > 1.2:
                continue
            tag = "START" if marker["kind"] == "start" else "END"
            mx, my = x2, y2
            cv2.circle(canvas, (mx, my), int(16 * scale), COLOR_MARK, -1)
            cv2.putText(canvas, tag, (mx - int(34 * scale), my - int(8 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 0, 0), 5)
            cv2.putText(canvas, tag, (mx - int(34 * scale), my - int(8 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, COLOR_MARK, 2)

    for light_id, light in lights.items():
        x1, y1, x2, y2 = (sx(v) for v in light.bbox)
        col = {"red": COLOR_WRONG, "yellow": COLOR_STOPPED, "green": (0, 200, 0)}.get(light.color, (160, 160, 160))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, max(1, int(2 * scale)))
        cv2.putText(canvas, f"{light_id}:{light.color}", (x1, max(int(18 * scale), y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, col, 2)

    header = (
        f"t={timestamp:7.2f}s  tracks={len(states)}  "
        f"stopped={sum(1 for v in per_track.values() if v.get('stopped_active'))}  "
        f"queued={sum(1 for v in per_track.values() if v.get('queued'))}  "
        f"wrong={sum(1 for v in per_track.values() if v.get('wrong_active'))}  "
        f"light={next(iter(lights.values())).color if lights else 'n/a'}"
    )
    cv2.putText(canvas, header, (sx(20), sy(44)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 0), 6)
    cv2.putText(canvas, header, (sx(20), sy(44)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (255, 255, 255), 2)


def render_timeline(events: list[dict[str, Any]], duration: float, path: str) -> None:
    """Draw the final event intervals on a time axis."""
    rows = ["stopped_vehicle", "wrong_way"]
    colors = {"stopped_vehicle": COLOR_STOPPED, "wrong_way": COLOR_WRONG}
    width, row_h, pad = 1600, 90, 60
    height = pad + row_h * len(rows) + 40
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    for index, label in enumerate(rows):
        y = pad + index * row_h
        cv2.line(canvas, (pad, y), (width - 20, y), (90, 90, 90), 2)
        count = sum(1 for e in events if e["label"] == label)
        cv2.putText(canvas, f"{label} ({count} events)", (10, y - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    for event in events:
        y = pad + rows.index(event["label"]) * row_h
        x1 = int(pad + (width - pad - 20) * (event["start_sec"] / max(duration, 1e-6)))
        x2 = int(pad + (width - pad - 20) * (event["end_sec"] / max(duration, 1e-6)))
        cv2.rectangle(canvas, (x1, y - 16), (max(x2, x1 + 3), y + 16), colors[event["label"]], -1)
        cv2.putText(canvas, f"#{event['track_id']}", (x1, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[event["label"]], 1)
    for second in range(0, int(duration) + 1, 30):
        x = int(pad + (width - pad - 20) * (second / max(duration, 1e-6)))
        cv2.line(canvas, (x, pad - 8), (x, pad + row_h * len(rows)), (70, 70, 70), 1)
        cv2.putText(canvas, f"{second}s", (x - 8, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (170, 170, 170), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/data_video1.mp4")
    parser.add_argument("--config", default="configs/data_video1_rules_dev.json")
    parser.add_argument("--out-video", default="debug/rules/rules_debug.mp4")
    parser.add_argument("--out-jsonl", default="debug/rules/events.jsonl")
    parser.add_argument("--out-events", default="debug/rules/events.json")
    parser.add_argument("--out-timeline", default="debug/rules/timeline.jpg")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out-width", type=int, default=1920)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = build_detector(config["detector"])
    tracker = build_tracker(config["tracker"])
    manager = TrackManager(
        max_age_seconds=float(config["track_manager"]["max_age_seconds"]),
        history_seconds=float(config["track_manager"]["history_seconds"]),
    )
    # Signal state must come from the CALIBRATED ROIs, not from detector boxes:
    # on this video the base checkpoint's "traffic light" output is a false
    # positive (a blue pedestrian-crossing sign), so detector-derived state would
    # be meaningless and queue suppression would never see a red light.
    stop_cfg = config["rules"]["stopped_vehicle"]
    stop_window = float(stop_cfg.get("window", 2.5))
    stop_net = float(stop_cfg.get("net_max_px", 25.0))
    stop_path = float(stop_cfg.get("path_max_px", 250.0))
    wrong_cfg = config["rules"]["wrong_way"]
    wrong_window = float(wrong_cfg.get("window", 2.5))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    scene_config = load_scene_config(config["scene"]["path"], width, height)
    scene = SceneContext(scene_config, width, height)
    light_reader = TrafficLightReader(scene, debounce_frames=3)
    engine = RuleEngine(config["rules"])
    segmenter = TemporalSegmenter(config["temporal"], duration=total / max(fps, 1e-6))
    print(f"video={args.video} {width}x{height} fps={fps:.3f} frames={total}")
    print(f"detector={type(detector).__name__} tracker={type(tracker).__name__} scene={scene.scene_id}")
    print(f"has_geometry={scene.has_geometry} has_road={scene.has_road} "
          f"lanes={[lane.lane_id for lane in scene_config.lanes]}")

    writer = None
    out_scale = 1.0
    if args.out_video and not args.no_video:
        Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
        out_w = args.out_width if args.out_width > 0 else width
        out_h = int(round(height * out_w / width))
        out_scale = out_w / width
        writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(1, args.stride), (out_w, out_h))
        print(f"writing debug video -> {args.out_video} ({out_w}x{out_h})")

    jsonl_file = None
    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = Path(args.out_jsonl).open("w", encoding="utf-8")

    stopped_timers: dict[int, float] = {}
    queue_timers: dict[int, float] = {}
    active_events: dict[int, float] = {}
    event_markers: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    light_states: dict[str, Any] = {}
    frame_index = -1
    processed = 0
    started = time.perf_counter()
    analyze = getattr(detector, "analyze", None)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.stride:
            continue
        if args.max_frames and processed >= args.max_frames:
            break
        processed += 1
        timestamp = frame_index / fps

        if analyze is not None:
            detections, _ = analyze(frame, frame_id=frame_index)
        else:
            detections = detector.predict(frame)
        trackable = [d for d in detections if not is_traffic_light(d.class_name)]
        observations = tracker.update(trackable, frame_index, timestamp, frame=frame)
        states = manager.update(observations, timestamp, frame_index, scene=scene)

        # Signal state from the calibrated ROIs (drives queue suppression).
        light_states = light_reader.update(frame, detections=[])
        scene.lights = light_states

        frame_state = FrameState(frame_index, timestamp, states,
                                 SceneState(scene.scene_id, width, height, light_states, scene))
        stopped = engine._stopped_vehicle_candidates(frame_state)
        wrong = engine._wrong_way_candidates(frame_state)

        # Live event start/end tracking, purely for the debug markers.
        active_now = {int(s.evidence["track_id"]) for s in stopped + wrong}
        for tid in list(active_events):
            if tid not in active_now:
                event_markers.append({"track_id": tid, "kind": "end", "t": round(timestamp, 3)})
                del active_events[tid]
        for tid in sorted(active_now):
            if tid not in active_events:
                event_markers.append({"track_id": tid, "kind": "start", "t": round(timestamp, 3)})
                active_events[tid] = timestamp

        per_track: dict[int, dict[str, Any]] = {}
        for track in states:
            heading = window_heading(track.bottom_history, wrong_window, timestamp)
            stationary = is_stationary(track.bottom_history, stop_window, stop_net, stop_path, timestamp)
            queued = bool(stationary and engine._queued_at_signal(track, frame_state))
            since = engine._condition_start.get(("stopped_vehicle", track.track_id))
            if queued and since is None:
                # Track when the suppressed queue started, for the overlay.
                queue_timers.setdefault(track.track_id, round(timestamp, 3))
            if not queued:
                queue_timers.pop(track.track_id, None)
            per_track[track.track_id] = {
                "observed": heading[0] if heading else None,
                "speed": window_speed(track.bottom_history, 2.5, timestamp),
                "stopped_since": since,
                "stopped_active": False,
                "queued": queued,
                "queued_for": (timestamp - queue_timers[track.track_id]) if track.track_id in queue_timers else 0.0,
                "wrong_active": False,
            }
        for signal in stopped:
            tid = int(signal.evidence["track_id"])
            per_track.setdefault(tid, {})
            per_track[tid]["stopped_since"] = signal.evidence["stopped_since"]
            per_track[tid]["stopped_active"] = True
        for signal in wrong:
            tid = int(signal.evidence["track_id"])
            per_track.setdefault(tid, {})
            per_track[tid]["wrong_active"] = True
            per_track[tid]["dot"] = signal.evidence["dot"]

        # feed the segmenter (harness contract) and log per-frame evidence
        engine_signals = {}
        engine_signals["stopped_vehicle"] = stopped[0] if stopped else engine._inactive("stopped_vehicle")
        engine_signals["wrong_way"] = wrong[0] if wrong else engine._inactive("wrong_way")
        segmenter.add(timestamp, engine_signals)

        if jsonl_file is not None:
            for signal in stopped + wrong:
                ev = signal.evidence
                record = {
                    "frame": frame_index,
                    "timestamp": round(timestamp, 3),
                    "track_id": ev["track_id"],
                    "label": signal.label,
                    "start_sec": round(float(signal.start_hint or timestamp), 3),
                    "lane_id": ev.get("lane_id"),
                    "evidence": ev,
                }
                events.append(record)
                jsonl_file.write(json.dumps(record) + "\n")

        if writer is not None:
            canvas = frame
            if out_scale != 1.0:
                canvas = cv2.resize(frame, (int(width * out_scale), int(height * out_scale)))
            draw_debug(canvas, states, scene, per_track, timestamp, out_scale, light_states,
                       markers=event_markers)
            writer.write(canvas)

    cap.release()
    if writer is not None:
        writer.release()
    if jsonl_file is not None:
        jsonl_file.close()
    elapsed = time.perf_counter() - started

    segments = segmenter.finalize(total / max(fps, 1e-6))
    final_events = []
    for start, end, label in segments:
        related = [e for e in events if e["label"] == label and e["start_sec"] <= start + 1e-6]
        tid = int(related[-1]["track_id"]) if related else -1
        lane = related[-1]["lane_id"] if related else None
        ev = related[-1]["evidence"] if related else {}
        final_events.append({
            "track_id": tid,
            "label": label,
            "start_sec": round(float(start), 3),
            "end_sec": round(float(end), 3),
            "duration_sec": round(float(end) - float(start), 3),
            "lane_id": lane,
            "evidence": ev,
        })

    if args.out_events:
        Path(args.out_events).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_events).write_text(json.dumps(final_events, indent=2), encoding="utf-8")
    if args.out_timeline:
        render_timeline(final_events, total / max(fps, 1e-6), args.out_timeline)

    by_label: dict[str, int] = defaultdict(int)
    for event in final_events:
        by_label[event["label"]] += 1
    print(f"\nprocessed {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.2f} fps)")
    print(f"events: {dict(by_label)}  total={len(final_events)}")
    for event in final_events:
        print(f"  {event['label']:16} track#{event['track_id']:<5} lane={event['lane_id']} "
              f"{event['start_sec']:7.2f} -> {event['end_sec']:7.2f} ({event['duration_sec']:6.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
