from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..contracts import TrackState
from ..scene.config import SceneContext
from .track_manager import TrackManager


class TrafficAnalytics:
    """Small lane-count/speed adapter inspired by Smart-Traffic's analytics.

    It is deliberately separate from the rule engine: the same statistics can
    be used for congestion rules, debugging, or a future reporting API.
    """

    def __init__(self) -> None:
        self._counts: dict[Any, int] = defaultdict(int)
        self._speeds: dict[Any, list[float]] = defaultdict(list)

    def update(self, tracks: list[TrackState], scene: SceneContext | None = None) -> dict[str, Any]:
        self._counts.clear()
        self._speeds.clear()
        for track in tracks:
            if not TrackManager.is_vehicle(track):
                continue
            lane_id = track.lane_id
            if lane_id is None and scene is not None:
                lane = scene.lane_for_point(track.center)
                lane_id = lane.lane_id if lane is not None else "unassigned"
            key = lane_id if lane_id is not None else "unassigned"
            self._counts[key] += 1
            self._speeds[key].append(track.speed)
        return {
            "counts": dict(self._counts),
            "average_speed": {
                key: (sum(values) / len(values) if values else 0.0)
                for key, values in self._speeds.items()
            },
        }
