"""Regenerate configs/default.json from the validated development config.

The organizers' harness calls ``solution.detect_events`` with no configuration
argument, so whatever lives in ``configs/default.json`` *is* the submission. It
used to be the starter file with placeholder thresholds and no scene, which
would have silently shipped untuned rules.

Generating it from the development config means the two cannot drift: the
development config is the one every threshold in it was measured against, and it
is what ``scripts/replay_rules.py`` and the unit tests exercise. Only the debug
block differs, because the submission must not write video.

Run after changing any threshold in the development config::

    python scripts/build_default_config.py
"""
from __future__ import annotations

import json
from pathlib import Path

DEV = Path("configs/data_video1_rules_dev.json")
OUT = Path("configs/default.json")

BANNER = [
    "GENERATED FILE - do not edit by hand.",
    "",
    "Produced by scripts/build_default_config.py from",
    "configs/data_video1_rules_dev.json, which is the configuration every",
    "threshold was measured against. The organizers' harness calls",
    "solution.detect_events(video_path) with no config argument, so this file",
    "is what actually runs in the submission.",
    "",
    "Regenerate with:  python scripts/build_default_config.py",
]


def main() -> int:
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    config: dict = {
        "_generated_by": "scripts/build_default_config.py",
        "_source": str(DEV).replace("\\", "/"),
        "_banner": BANNER,
        "detector": dev["detector"],
        "tracker": dev["tracker"],
        "track_manager": dev["track_manager"],
        "scene": dev["scene"],
        "rules": dev["rules"],
        "risk": dev["risk"],
        "temporal": dev["temporal"],
        # The submission must not spend its time budget encoding a debug video.
        "debug": {"enabled": False, "write_video": False,
                  "dump_jsonl": False, "output_dir": "debug"},
    }
    OUT.write_text(json.dumps(config, indent=2), encoding="utf-8")
    enabled = sorted(k for k, v in config["rules"].items() if v.get("enabled", True))
    print(f"wrote {OUT}")
    print(f"scene:    {config['scene']['path']}")
    print(f"detector: {config['detector'].get('backend')} "
          f"{config['detector'].get('primary', {}).get('model', config['detector'].get('model'))} "
          f"stride {config['detector'].get('stride')}")
    print(f"rules enabled ({len(enabled)}): {', '.join(enabled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
