"""Rebuild the previous milestone's 4-coarse-band scene for before/after runs.

The repository's HEAD predates that calibration, so the old lane set is
reconstructed here from the coordinates recorded in camera.md and in the
scene file's own history. Everything except `lanes` is byte-identical to the
current scene, so a bounded timing run isolates the effect of the geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

OLD_LANES = [
    {
        "id": 0,
        "name": "A_far",
        "polygon": [[0, 225], [900, 355], [1800, 490], [2500, 588],
                    [2500, 685], [1800, 585], [900, 455], [0, 328]],
        "direction": [1.0, 0.11],
        "allowed_moves": ["straight"],
    },
    {
        "id": 1,
        "name": "A_near",
        "polygon": [[0, 318], [900, 450], [1800, 585], [2500, 685],
                    [2500, 790], [1800, 690], [900, 560], [0, 430]],
        "direction": [1.0, 0.11],
        "allowed_moves": ["straight"],
    },
    {
        "id": 2,
        "name": "B_far",
        "polygon": [[0, 430], [900, 560], [1800, 690], [2500, 790],
                    [2500, 948], [1800, 890], [900, 805], [0, 710]],
        "direction": [1.0, 0.11],
        "allowed_moves": ["straight"],
        "approach_point": [1600, 780],
    },
    {
        "id": 3,
        "name": "B_near",
        "polygon": [[0, 710], [900, 805], [1800, 890], [2500, 948],
                    [2500, 1105], [1800, 1090], [900, 1050], [0, 990]],
        "direction": [1.0, 0.11],
        "allowed_moves": ["straight"],
        "approach_point": [1600, 1000],
    },
]


def main() -> int:
    current = json.loads(
        Path("configs/scenes/data_video1.json").read_text(encoding="utf-8")
    )
    old = dict(current)
    old["scene_id"] = "data_video1_coarse4"
    old["lanes"] = OLD_LANES
    old["_comment"] = [
        "Reconstruction of the previous milestone's 4 coarse lane bands, all",
        "declared [1.0, 0.11] over y~225-1105. Used only for before/after",
        "timing and opposing-share comparisons; see camera.md.",
    ]
    Path("configs/scenes/_old_lanes.json").write_text(
        json.dumps(old, indent=2), encoding="utf-8"
    )

    config = json.loads(
        Path("configs/data_video1_rules_dev.json").read_text(encoding="utf-8")
    )
    config["scene"]["path"] = "configs/scenes/_old_lanes.json"
    # The previous milestone ran with wrong_way gated off entirely.
    config["rules"]["wrong_way"]["lane_allowlist"] = []
    Path("configs/_old_lanes_rules.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print("wrote configs/scenes/_old_lanes.json and configs/_old_lanes_rules.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
