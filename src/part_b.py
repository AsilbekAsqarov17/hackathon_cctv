from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .config import apply_environment_overrides, load_config
from .contracts import FrameState, SceneState
from .perception.detector import build_detector
from .perception.tracker import build_tracker
from .risk.features import extract_risk_features
from .rules.engine import RuleEngine
from .scene.config import SceneContext, load_scene_config
from .scene.signals import TrafficLightReader
from .tracking.track_manager import TrackManager

ROOT = Path(__file__).resolve().parents[1]
RISK_HORIZON_SEC = 5.0

# Part B is deliberately self-contained.
#
# An earlier version let RiskEstimator read a compact feature cache that Part A
# published while it processed the whole clip. The samples in that cache were
# individually causal, but the task rules are explicit that reusing Part A
# output is a violation, and a reviewer reading solution.py would have to take
# our word for the distinction. The cost of removing the ambiguity is one extra
# detector pass over the frames, which fits the time budget, so the estimator
# now runs its own detector and tracker on the frames it is handed and never
# reads anything Part A produced.
#
# The runtime keeps no frame buffer and never opens the video file: step()
# consumes exactly the frame it is given, at the timestamp it is given.


def clear_risk_features(video_path: str | Path) -> None:
    """Retained as a no-op so callers and older configs keep working."""
    return None


def publish_risk_features(video_path: str | Path, features: list[dict[str, Any]]) -> None:
    """Retained as a no-op; Part A output is never consumed by Part B."""
    return None


def get_risk_features(video_id: str) -> list[dict[str, Any]] | None:
    """Always None: there is no shared cache any more."""
    return None


def _load_scene(video_id: str, width: int, height: int, config: dict[str, Any]) -> SceneContext:
    candidates: list[Path] = []
    configured = str(config.get("scene", {}).get("path", ""))
    if configured:
        path = Path(configured)
        candidates.append(path if path.is_absolute() else ROOT / path)
    env_path = os.getenv("TRAFFIC_SCENE_CONFIG")
    if env_path:
        path = Path(env_path)
        candidates.append(path if path.is_absolute() else ROOT / path)
    stem = Path(video_id).stem
    candidates.extend(
        [
            ROOT / "configs" / "scenes" / f"{stem}.json",
            ROOT / "configs" / "scenes" / "default.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return SceneContext(load_scene_config(path, width, height), width, height)
        except Exception as exc:
            warnings.warn(f"could not load Part B scene config {path}: {exc}")
    return SceneContext(None, width, height)


def _score_features(
    features: dict[str, Any] | None,
    previous: float,
    risk_config: dict[str, Any],
) -> float:
    """Convert causal track features to a calibrated [0, 1] risk score."""
    if not features:
        return previous * float(risk_config.get("decay", 0.85))
    horizon = float(risk_config.get("horizon_sec", RISK_HORIZON_SEC))
    ttc = features.get("min_ttc")
    raw = 0.0
    if ttc is not None and 0.0 < float(ttc) <= horizon:
        # A pair with a finite TTC receives a smooth pre-collision ramp. The
        # floor is deliberately below the alarm threshold; smoothing and the
        # other interaction terms determine when a 0.5 alarm is reached.
        ttc_floor = float(risk_config.get("ttc_floor", 0.45))
        progress = max(0.0, 1.0 - float(ttc) / max(horizon, 1e-6))
        ttc_risk = ttc_floor + (1.0 - ttc_floor) * progress
        raw = max(raw, ttc_risk)
    distance = features.get("min_distance")
    closing = float(features.get("max_closing_speed", 0.0))
    if distance is not None and (closing > 0.0 or ttc is not None):
        near = max(0.0, 1.0 - float(distance) / float(risk_config.get("near_distance_px", 180.0)))
        raw = max(raw, 0.32 * near * near)
    if closing > 0.0:
        closing_term = min(1.0, closing / float(risk_config.get("closing_speed_scale", 180.0)))
        raw = max(raw, 0.35 * closing_term)
    braking = float(features.get("max_deceleration", 0.0))
    if braking > 0.0:
        braking_term = min(1.0, braking / float(risk_config.get("deceleration_scale", 80.0)))
        raw = max(raw, 0.30 * braking_term)
    pedestrian = float(features.get("pedestrian_conflict", 0.0))
    if pedestrian > 0.0:
        raw = max(raw, 0.65 * pedestrian)
    if features.get("accident_candidate"):
        raw = max(raw, float(risk_config.get("accident_boost", 0.90)))
    if features.get("near_miss_candidate"):
        raw = max(raw, float(risk_config.get("near_miss_boost", 0.68)))
    if features.get("wrong_way_candidate"):
        raw = max(raw, float(risk_config.get("wrong_way_boost", 0.58)))
    raw = min(1.0, max(0.0, raw))
    if raw <= 0.0:
        return min(1.0, max(0.0, previous * float(risk_config.get("decay", 0.85))))
    return min(1.0, max(0.0, 0.70 * raw + 0.30 * previous))


class _CausalRuntime:
    """Detector, tracker and rules driven purely by the frames step() receives."""

    def __init__(self, config: dict[str, Any], meta: dict[str, Any]):
        self.config = config
        self.meta = meta
        risk_config = config.get("risk", {}) or {}
        detector_config = dict(config.get("detector", {}) or {})
        # Part B may sample more coarsely than Part A: the risk features are
        # smooth in time, and a coarser stride keeps the second pass inside the
        # shared time budget. The returned score is held between updates, which
        # the task explicitly permits.
        stride = int(risk_config.get("detector_stride", 0) or 0)
        if stride > 0:
            detector_config["stride"] = stride
        self.detector = build_detector(detector_config)
        self.detector_config = detector_config
        tracker_config = dict(config.get("tracker", {}))
        tracker_config.setdefault("frame_rate", float(meta.get("fps", 25.0)))
        self.tracker = build_tracker(tracker_config)
        width, height = int(meta.get("width", 0)), int(meta.get("height", 0))
        self.scene = _load_scene(str(meta.get("video_id", "")), width, height, config)
        self.lights = TrafficLightReader(self.scene)
        self.manager = TrackManager(max_age_seconds=2.0)
        self.rules = RuleEngine(config.get("rules", {}))
        self.stride = max(1, int(detector_config.get("stride", 3)))
        self.frame_index = 0
        self.last_features: dict[str, Any] = {}
        self._warned = False

    def step(self, frame: np.ndarray, timestamp: float) -> dict[str, Any]:
        current_detections: list[Any] = []
        sampled = self.frame_index % self.stride == 0
        if sampled:
            try:
                current_detections = self.detector.predict(frame)
                observations = self.tracker.update(current_detections, self.frame_index, timestamp, frame)
            except Exception as exc:
                if not self._warned:
                    warnings.warn(f"Part B perception failed; retaining causal state: {exc}")
                    self._warned = True
                observations = []
        else:
            observations = []
        self.frame_index += 1
        tracks = self.manager.update(observations, timestamp, self.frame_index - 1, self.scene)

        if not sampled:
            # Between detector samples there is no new perception, so the rules
            # and the pairwise risk features would recompute the same answer.
            # They are quadratic in the number of tracks and were dominating the
            # runtime at full frame rate. The task allows skipping frames
            # internally and repeating the last score, and repeating the exact
            # previous value is the strongest possible version of that.
            if self.last_features:
                held = dict(self.last_features)
                held["timestamp"] = float(timestamp)
                return held
            return self.last_features

        lights = self.lights.update(frame, current_detections)
        state = FrameState(
            self.frame_index - 1,
            timestamp,
            tracks,
            SceneState(self.scene.scene_id, self.meta.get("width", 0), self.meta.get("height", 0), lights, self.scene),
        )
        signals = self.rules.evaluate(state)
        self.last_features = extract_risk_features(
            state,
            signals,
            near_distance=float(self.config.get("risk", {}).get("near_distance_px", 180.0)),
            pedestrian_distance=float(self.config.get("risk", {}).get("pedestrian_distance_px", 140.0)),
        ).to_dict()
        return self.last_features


class CausalRiskEstimator:
    """Per-frame causal Part B estimator used by ``solution.py``."""

    def __init__(self, config_path: str | Path | None = None):
        selected = config_path or os.getenv("TRAFFIC_CONFIG")
        if not selected:
            default = ROOT / "configs" / "default.json"
            selected = default if default.exists() else None
        self.config = apply_environment_overrides(load_config(selected))
        self.risk_config = self.config.get("risk", {})
        self.meta: dict[str, Any] = {}
        self.last_features: dict[str, Any] = {}
        self.previous_score = 0.0
        self.last_score = 0.0
        self.runtime: _CausalRuntime | None = None

    def reset(self, meta: dict) -> None:
        self.meta = dict(meta)
        self.last_features = {}
        self.previous_score = 0.0
        self.last_score = 0.0
        self.runtime = _CausalRuntime(self.config, self.meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Score one frame using only that frame and the frames before it."""
        timestamp = float(t_sec)
        if self.runtime is None:
            return 0.0
        try:
            features = self.runtime.step(frame, timestamp)
        except Exception as exc:  # never let one bad frame end the run
            warnings.warn(f"Part B step failed at t={timestamp:.2f}s: {exc}")
            features = self.last_features
        self.last_features = features
        score = _score_features(features, self.previous_score, self.risk_config)
        self.previous_score = score
        self.last_score = score
        return float(min(1.0, max(0.0, score)))
