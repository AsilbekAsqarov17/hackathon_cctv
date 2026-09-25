"""Camera calibration helpers used by scene configuration tooling."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .geometry import compute_homography


def validate_homography(
    matrix: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
) -> dict[str, float]:
    h = np.asarray(matrix, dtype=np.float64)
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
    projected = cv2.perspectiveTransform(src, h).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    return {
        "mean_error": float(errors.mean()) if len(errors) else 0.0,
        "max_error": float(errors.max()) if len(errors) else 0.0,
        "condition_number": float(np.linalg.cond(h)),
    }


__all__ = ["compute_homography", "validate_homography"]
