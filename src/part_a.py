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
from .part_b import clear_risk_features, publish_risk_features
from .perception.detector import build_detector
from .perception.tracker import build_tracker
from .risk.features import extract_risk_features
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

    def __init__(self, config_path: str | Path | None = None):
        selected = config_path or os.getenv("TRAFFIC_CONFIG")
        if not selected:
            default = ROOT / "configs" / "default.json"
            selected = default if default.exists() else None
        self.config: dict[str, Any] = apply_environment_overrides(load_config(selected))
        self.detector_config = self.config.get("detector", {})
        self.tracker_config = self.config.get("tracker", {})
        active = self.config.get("active_classes", OFFICIAL_LABELS)
        if isinstance(active, str):
            active = [item.strip() for item in active.split(",") if item.strip()]
        self.active_classes = set(active)
        self.rules = RuleEngine(self.config.get("rules", {}))
        self.debug_enabled = bool(self.config.get("debug", {}).get("enabled", False))
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
        clear_risk_features(video_path)
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
        stride = max(1, int(self.detector_config.get("stride", 3)))
        writer: cv2.VideoWriter | None = None
        flags_file = None
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.debug_dir / f"{Path(video_path).stem}_part_a.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, max(1.0, info.fps), (info.width, info.height))
            if self.dump_jsonl:
                flags_file = (self.debug_dir / f"{Path(video_path).stem}_flags.jsonl").open("w", encoding="utf-8")
        processed = 0
        risk_samples: list[dict[str, Any]] = []
        last_timestamp = 0.0
        try:
            for frame_id, timestamp, frame in reader.frames(stride=stride):
                last_timestamp = timestamp
                detections = detector.predict(frame)
                observations = tracker.update(detections, frame_id, timestamp, frame)
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
                risk_samples.append(
                    extract_risk_features(
                        state,
                        signals,
                        near_distance=float(self.config.get("risk", {}).get("near_distance_px", 180.0)),
                        pedestrian_distance=float(self.config.get("risk", {}).get("pedestrian_distance_px", 140.0)),
                    ).to_dict()
                )
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
        publish_risk_features(video_path, risk_samples)
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
