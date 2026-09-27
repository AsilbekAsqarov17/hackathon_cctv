from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

# The default is deliberately conservative. Geometry-dependent rules stay off
# until a scene configuration is supplied for the actual camera.
DEFAULT_CONFIG: dict[str, Any] = {
    "detector": {
        "backend": "auto",  # auto, yolo, motion, or null
        "model": "yolo11n.pt",
        "confidence": 0.25,
        "iou": 0.5,
        "imgsz": 640,
        "device": "",
        "stride": 3,
        "fallback_motion": False,
        "allow_download": False,
    },
    "tracker": {
        "backend": "official",  # official or simple_bytetrack
        "high_threshold": 0.5,
        "low_threshold": 0.15,
        "match_threshold": 0.7,
        "max_age": 30,
        "min_hits": 2,
    },
    "scene": {
        "path": "",
        "default_path": "configs/scenes/default.json",
    },
    "rules": {
        # Trajectory-window thresholds. net_max_px / path_max_px were measured
        # from the data_video1 track history: stationary windows sit at <=20 px
        # net while moving windows start at ~31 px, so 25 px separates them.
        "wrong_way": {
            "enabled": True,
            "window": 2.5,
            "direction_dot": -0.45,
            "min_travel_px": 45.0,
            "min_speed": 6.0,
            "confirm_sec": 1.5,
            # Only calibrated lanes may raise wrong_way. Lanes absent from the
            # scene file (east leg, lower intersection) have no direction and so
            # can never match, and this allowlist enforces that explicitly.
            "lane_allowlist": [0, 1, 2, 3],
        },
        "stopped_vehicle": {
            "enabled": True,
            "window": 2.5,
            "net_max_px": 25.0,
            "path_max_px": 250.0,
            "duration": 10.0,
            "tolerance_gap": 1.5,
            "queue_radius_px": 220.0,
            "speed": 3.0,
        },
        "congestion": {"enabled": True, "min_vehicles": 4, "speed": 8.0, "duration": 5.0},
        "stop_line": {"enabled": True, "hold": 0.6},
        "red_light": {"enabled": True, "hold": 0.8},
        "solid_line_crossing": {"enabled": True, "hold": 0.5},
        "illegal_turn": {"enabled": True, "hold": 0.8},
        "illegal_u_turn": {"enabled": True, "hold": 0.8, "angle": 2.35},
        "jaywalking": {"enabled": True, "duration": 1.0},
        "failure_to_yield": {"enabled": True, "hold": 0.8},
        "near_miss": {"enabled": True, "ttc": 2.5, "distance": 120.0, "hold": 0.5},
        "accident": {"enabled": True, "distance": 55.0, "hold": 0.8},
        "road_obstacle": {"enabled": True, "duration": 1.0},
        "fire_smoke": {"enabled": True, "duration": 0.5},
    },
    "risk": {
        "horizon_sec": 5.0,
        "ttc_floor": 0.45,
        "near_distance_px": 180.0,
        "pedestrian_distance_px": 140.0,
        "closing_speed_scale": 180.0,
        "deceleration_scale": 80.0,
        "decay": 0.85,
        "accident_boost": 0.90,
        "near_miss_boost": 0.68,
        "wrong_way_boost": 0.58,
    },
    "temporal": {
        "merge_gap": 1.0,
        "min_duration": {
            "accident": 0.3,
            "near_miss": 0.3,
            "red_light": 0.3,
            "wrong_way": 1.0,
            "illegal_u_turn": 0.5,
            "stopped_vehicle": 10.0,
            "jaywalking": 0.5,
            "failure_to_yield": 0.5,
            "illegal_turn": 0.5,
            "solid_line_crossing": 0.3,
            "stop_line": 0.3,
            "congestion": 5.0,
            "road_obstacle": 0.5,
            "fire_smoke": 0.3,
        },
    },
    "debug": {"enabled": False, "write_video": True, "dump_jsonl": False, "output_dir": "debug"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load JSON configuration, with optional PyYAML support.

    JSON is the default because it is part of the Python standard library and
    therefore works in the offline harness without another dependency.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        return config
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        project_candidate = Path(__file__).resolve().parents[1] / path
        if project_candidate.exists():
            path = project_candidate
    if not path.exists():
        raise FileNotFoundError(f"configuration not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON instead") from exc
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be an object")
    return _deep_merge(config, loaded)


def apply_environment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply small runtime overrides useful for the competition harness."""
    out = copy.deepcopy(config)
    mapping = {
        "TRAFFIC_YOLO_MODEL": ("detector", "model"),
        "TRAFFIC_DEVICE": ("detector", "device"),
        "TRAFFIC_IMGSZ": ("detector", "imgsz"),
        "TRAFFIC_STRIDE": ("detector", "stride"),
        "TRAFFIC_DETECTOR_BACKEND": ("detector", "backend"),
        "TRAFFIC_TRACKER_BACKEND": ("tracker", "backend"),
        "TRAFFIC_SCENE_CONFIG": ("scene", "path"),
        "TRAFFIC_ACTIVE_CLASSES": None,
    }
    for env_name, path in mapping.items():
        value = os.getenv(env_name)
        if not value:
            continue
        if path is None:
            out["active_classes"] = [x.strip() for x in value.split(",") if x.strip()]
            continue
        section, key = path
        out.setdefault(section, {})[key] = value
    return out
