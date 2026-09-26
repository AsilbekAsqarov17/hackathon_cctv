from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import apply_environment_overrides, load_config
from .contracts import FrameState, SceneState
from .perception.detector import build_detector
from .perception.tracker import build_tracker
from .rules.engine import OFFICIAL_LABELS, RuleEngine
from .scene.config import SceneContext, load_scene_config
from .scene.signals import TrafficLightReader
from .temporal.segmenter import TemporalSegmenter
from .tracking.analytics import TrafficAnalytics
from .tracking.track_manager import TrackManager
from .video import VideoReader

ROOT = Path(__file__).resolve().parents[1]


class PartAPipeline:
    """Complete offline Part A pipeline for one video."""

    def __init__(self, config_path: str | Path | None = None, deadline_seconds: float | None = None):
        selected = config_path or os.getenv("TRAFFIC_CONFIG")
        if not selected:
            default = ROOT / "configs" / "default.json"
            selected = default if default.exists() else None
        self.config: dict[str, Any] = apply_environment_overrides(load_config(selected))
        self.deadline_seconds = deadline_seconds
        self.detector_config = self.config.get("detector", {})
        self.tracker_config = self.config.get("tracker", {})
        active = self.config.get("active_classes", OFFICIAL_LABELS)
        if isinstance(active, str):
            active = [item.strip() for item in active.split(",") if item.strip()]
        self.active_classes = set(active)
        self.rules = RuleEngine(self.config.get("rules", {}))
        self.debug_enabled = bool(self.config.get("debug", {}).get("enabled", False))
        self.write_debug_video = bool(self.config.get("debug", {}).get("write_video", True))
        self.dump_jsonl = bool(self.config.get("debug", {}).get("dump_jsonl", False))
        self.debug_dir = Path(self.config.get("debug", {}).get("output_dir", ROOT / "debug"))
        if not self.debug_dir.is_absolute():
            self.debug_dir = ROOT / self.debug_dir

    def _scene_for_video(self, video_path: str, width: int, height: int) -> SceneContext:
        configured = str(self.config.get("scene", {}).get("path", ""))
        candidates: list[Path] = []
        if configured:
            configured_path = Path(configured)
            candidates.append(configured_path if configured_path.is_absolute() else ROOT / configured_path)
        env_path = os.getenv("TRAFFIC_SCENE_CONFIG")
        if env_path:
            env_configured = Path(env_path)
            candidates.append(env_configured if env_configured.is_absolute() else ROOT / env_configured)
        candidates.extend(
            [
                ROOT / "configs" / "scenes" / f"{Path(video_path).stem}.json",
                ROOT / "configs" / "scenes" / "default.json",
            ]
        )
        scene_path: Path | None = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                scene_path = candidate
                break
        if scene_path is None:
            return SceneContext(None, width, height)
        try:
            return SceneContext(load_scene_config(scene_path, width, height), width, height)
        except Exception as exc:
            warnings.warn(f"could not load scene config {scene_path}: {exc}")
            return SceneContext(None, width, height)

    @staticmethod
    def _draw_debug(
        frame: np.ndarray,
        state: FrameState,
        signals: dict[str, Any],
    ) -> np.ndarray:
        output = frame.copy()
        for track in state.tracks:
            if track.missed_frames > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = (0, 220, 0) if track.class_name.lower() == "person" else (0, 180, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{track.track_id}:{track.class_name}"
            cv2.putText(output, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        active = [label for label, signal in signals.items() if signal.active]
        if active:
            cv2.putText(output, ", ".join(active), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    def run(self, video_path: str) -> list[list[float | str]]:
        started = time.perf_counter()
        reader = VideoReader(video_path)
        info = reader.info
        # OpenCV occasionally reports zero for streamed files. Keep a usable
        # duration so temporal output is not rejected by the harness.
        duration = info.duration
        scene_context = self._scene_for_video(video_path, info.width, info.height)
        light_reader = TrafficLightReader(scene_context)
        detector = build_detector(self.detector_config)
        tracker_config = dict(self.tracker_config)
        tracker_config.setdefault("frame_rate", info.fps)
        tracker = build_tracker(tracker_config)
        manager = TrackManager(max_age_seconds=2.0)
        analytics = TrafficAnalytics()
        segmenter = TemporalSegmenter(self.config.get("temporal", {}), duration=duration)
        base_stride = max(1, int(self.detector_config.get("stride", 3)))
        max_stride = max(base_stride, int(self.config.get("runtime", {}).get("max_stride", 12)))
        stride = base_stride

        # The harness allows three times the video duration for Part A and
        # Part B together, and a video that overruns is scored as empty --
        # worse than predicting nothing, because the work is thrown away. This
        # pipeline cannot see the harness budget, so it sets its own from the
        # duration and keeps the sampling rate honest against the clock.
        budget = self.deadline_seconds
        if budget is None:
            budget = float(duration) * float(
                self.config.get("runtime", {}).get("part_a_budget_factor", 1.3)
            )
        deadline = time.perf_counter() + max(5.0, budget)
        writer: cv2.VideoWriter | None = None
        flags_file = None
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            if self.write_debug_video:
                output_path = self.debug_dir / f"{Path(video_path).stem}_part_a.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # The loop only yields every `stride`-th frame, but the writer is
                # told the source frame rate, so the rendered clip plays back
                # `stride` times too fast -- which is actively misleading when
                # the point of the render is to judge whether a detection is
                # right. Declare the rate the frames are actually written at.
                render_fps = max(1.0, info.fps / max(1, stride))
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, render_fps, (info.width, info.height)
                )
            if self.dump_jsonl:
                flags_file = (self.debug_dir / f"{Path(video_path).stem}_flags.jsonl").open("w", encoding="utf-8")
        processed = 0
        last_timestamp = 0.0
        perception_warned = False
        next_deadline_check = 200
        try:
            for frame_id, timestamp, frame in reader.frames(stride=stride):
                last_timestamp = timestamp
                if frame_id >= next_deadline_check:
                    next_deadline_check = frame_id + 200
                    remaining_frames = max(1, int(info.frame_count) - frame_id)
                    remaining_time = deadline - time.perf_counter()
                    # Seconds still affordable per remaining second of video.
                    # Falling below 1.0 means we will overrun at the current
                    # rate, so widen the sampling rather than run out of clock.
                    affordable = remaining_time / max(1e-6, remaining_frames / max(1e-6, info.fps))
                    # A rendered debug clip is for inspection, not for the timed
                    # submission path, and its declared frame rate assumes a
                    # constant stride. Freezing it keeps the render honest.
                    if affordable < 1.0 and stride < max_stride and not self.debug_enabled:
                        stride = min(max_stride, max(stride + 1, int(stride * (1.0 / max(0.05, affordable)))))
                        warnings.warn(
                            f"Part A behind schedule; detector stride widened to {stride}"
                        )
                try:
                    detections = detector.predict(frame)
                except Exception as exc:
                    if not perception_warned:
                        warnings.warn(f"detector failed; continuing without detections: {exc}")
                        perception_warned = True
                    detections = []
                try:
                    observations = tracker.update(detections, frame_id, timestamp, frame)
                except Exception as exc:
                    if not perception_warned:
                        warnings.warn(f"tracker failed; continuing without tracks: {exc}")
                        perception_warned = True
                    observations = []
                tracks = manager.update(observations, timestamp, frame_id, scene_context)
                traffic_stats = analytics.update(tracks, scene_context)
                lights = light_reader.update(frame, detections)
                scene_state = SceneState(
                    scene_id=scene_context.scene_id,
                    width=info.width,
                    height=info.height,
                    lights=lights,
                    context=scene_context,
                )
                state = FrameState(frame_id, timestamp, tracks, scene_state)
                signals = self.rules.evaluate(state)
                filtered = {
                    label: signal
                    for label, signal in signals.items()
                    if label in self.active_classes
                }
                segmenter.add(timestamp, filtered)
                if flags_file is not None:
                    flags_file.write(
                        json.dumps(
                            {
                                "timestamp": round(timestamp, 4),
                                "active": [label for label, signal in filtered.items() if signal.active],
                                "tracks": manager.snapshot(),
                                "traffic": traffic_stats,
                            }
                        )
                        + "\n"
                    )
                if writer is not None:
                    writer.write(self._draw_debug(frame, state, filtered))
                processed += 1
        finally:
            reader.close()
            if writer is not None:
                writer.release()
            if flags_file is not None:
                flags_file.close()
        if duration <= 0.0 and last_timestamp > 0.0:
            duration = last_timestamp + 1.0 / max(1.0, info.fps)
        events = segmenter.finalize(duration)
        elapsed = time.perf_counter() - started
        print(
            f"[Part A] {Path(video_path).name}: {processed} sampled frames, "
            f"{len(events)} events, {elapsed:.1f}s"
        )
        return events


def run_part_a(video_path: str, config_path: str | Path | None = None) -> list[list[float | str]]:
    """Run Part A and fail soft so the official harness can process other videos."""
    try:
        return PartAPipeline(config_path).run(video_path)
    except Exception as exc:
        warnings.warn(f"Part A failed for {video_path}: {exc}")
        return []
