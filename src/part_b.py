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

# Part A and Part B are called sequentially by the organizers' harness. The
# cache stores only compact causal snapshots; the Part B reader exposes no
# sample whose timestamp is later than the current step.
_FEATURE_CACHES: dict[str, list[dict[str, Any]]] = {}


def _video_key(video_path: str | Path) -> str:
    return Path(video_path).name


def clear_risk_features(video_path: str | Path) -> None:
    _FEATURE_CACHES.pop(_video_key(video_path), None)


def publish_risk_features(video_path: str | Path, features: list[dict[str, Any]]) -> None:
    # The harness processes videos sequentially; retaining only the latest
    # compact feature set avoids accumulating memory across a large test set.
    _FEATURE_CACHES.clear()
    _FEATURE_CACHES[_video_key(video_path)] = list(features)


def get_risk_features(video_id: str) -> list[dict[str, Any]] | None:
    return _FEATURE_CACHES.get(video_id)


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
    # The ramp is built on the *conflict* TTC, not the raw minimum over all
    # pairs. Thirty vehicles in a queue always contain some pair with a short
    # time-to-collision, so a raw minimum saturates the score on ordinary
    # traffic and produces a constant alarm; measured over data_video2 that put
    # 98 percent of frames above the official 0.5 threshold. Only pairs that
    # share a lane (or involve a pedestrian) and are genuinely closing count.
    ttc = features.get("min_conflict_ttc")
    if ttc is None and "conflict_pair_count" not in features:
        # A caller that predates the conflict fields (an older cache, or a unit
        # test feeding a hand-built dict) has no conflict count, so fall back to
        # the raw minimum rather than silently scoring nothing. When the field
        # IS present and zero, the answer is genuinely "no conflicting pair".
        ttc = features.get("min_ttc")
    if ttc is not None and features.get("conflict_evasive") is False:
        # Close and closing is not yet risk. Measured on data_video1 at 4K, 9-19
        # pairs pass the conflict test at any moment with a median TTC of
        # 1.17 s - that is ordinary car-following, because the lead vehicle is
        # going to brake. The ramp is therefore gated on a conflicting pair also
        # showing evasive behaviour, reusing the criteria the validated
        # `near_miss` rule uses. Without this gate the score read above the
        # official alarm threshold on 80% of frames of ordinary traffic.
        ttc = None
    raw = 0.0
    if ttc is not None and 0.0 < float(ttc) <= horizon:
        # A pair on a genuine collision course, closing evasively, receives a
        # smooth pre-collision ramp. The ramp itself is deliberately shorter
        # than the 5 s reporting horizon: the score is a confidence that a
        # collision is imminent, not a countdown, and a pair still 4 s away is
        # not yet risk. The floor sits below the alarm threshold; the other
        # interaction terms decide when 0.5 is actually reached.
        ttc_floor = float(risk_config.get("ttc_floor", 0.35))
        ramp = float(risk_config.get("ttc_ramp_sec", 2.0))
        progress = max(0.0, 1.0 - float(ttc) / max(ramp, 1e-6))
        ttc_risk = ttc_floor + (1.0 - ttc_floor) * progress
        raw = max(raw, ttc_risk)
    closing = float(features.get("max_closing_speed", 0.0))
    if features.get("conflict_pair_count"):
        distance = features.get("min_distance")
        if distance is not None:
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
    if features.get("red_light_candidate"):
        # A red-light violation makes a collision more likely within the horizon,
        # so it raises the score, but it is not itself a collision and must not
        # reach the accident boost.
        raw = max(raw, float(risk_config.get("red_light_boost", 0.45)))
    raw = min(1.0, max(0.0, raw))
    if raw <= 0.0:
        return min(1.0, max(0.0, previous * float(risk_config.get("decay", 0.85))))
    return min(1.0, max(0.0, 0.70 * raw + 0.30 * previous))


class _CausalRuntime:
    """Fallback causal detector/tracker runtime when no Part A cache exists."""

    def __init__(self, config: dict[str, Any], meta: dict[str, Any]):
        self.config = config
        self.meta = meta
        self.detector = build_detector(config.get("detector", {}))
        tracker_config = dict(config.get("tracker", {}))
        tracker_config.setdefault("frame_rate", float(meta.get("fps", 25.0)))
        self.tracker = build_tracker(tracker_config)
        width, height = int(meta.get("width", 0)), int(meta.get("height", 0))
        self.scene = _load_scene(str(meta.get("video_id", "")), width, height, config)
        self.lights = TrafficLightReader(self.scene)
        self.manager = TrackManager(max_age_seconds=2.0)
        self.rules = RuleEngine(config.get("rules", {}))
        self.stride = max(1, int(config.get("detector", {}).get("stride", 3)))
        self.frame_index = 0
        self.last_features: dict[str, Any] = {}
        self._warned = False

    def step(self, frame: np.ndarray, timestamp: float) -> dict[str, Any]:
        current_detections: list[Any] = []
        if self.frame_index % self.stride == 0:
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
        self.cache: list[dict[str, Any]] | None = None
        self.cache_index = 0
        self.last_features: dict[str, Any] = {}
        self.previous_score = 0.0
        self.last_score = 0.0
        self.runtime: _CausalRuntime | None = None

    def reset(self, meta: dict) -> None:
        self.meta = dict(meta)
        video_id = str(self.meta.get("video_id", ""))
        self.cache = get_risk_features(video_id)
        self.cache_index = 0
        self.last_features = {}
        self.previous_score = 0.0
        self.last_score = 0.0
        self.runtime = None if self.cache is not None else _CausalRuntime(self.config, self.meta)

    def _cached_features(self, timestamp: float) -> dict[str, Any]:
        if not self.cache:
            return self.last_features
        # The pointer is deliberately monotonic: a future cached sample can
        # never be read by an earlier step, even though Part A saw the full
        # video while building the compact cache.
        while self.cache_index < len(self.cache):
            sample = self.cache[self.cache_index]
            if float(sample.get("timestamp", 0.0)) > timestamp + 1e-6:
                break
            self.last_features = sample
            self.cache_index += 1
        return self.last_features

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        timestamp = float(t_sec)
        if self.cache is not None:
            features = self._cached_features(timestamp)
        elif self.runtime is not None:
            features = self.runtime.step(frame, timestamp)
        else:
            features = {}
        score = _score_features(features, self.previous_score, self.risk_config)
        self.previous_score = score
        self.last_score = score
        return float(min(1.0, max(0.0, score)))
