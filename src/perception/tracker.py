from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..contracts import BBox, Detection, TrackObservation
from ..scene.geometry import bbox_iou, normalized_vector, vector_dot


@dataclass
class _Track:
    track_id: int
    bbox: BBox  # (x1, y1, x2, y2)
    score: float
    class_id: int
    class_name: str
    velocity: tuple[float, float]  # pixels per frame (vx, vy) for center
    last_frame: int
    hits: int = 1
    missed: int = 0

    def predicted_bbox(self, frame_id: int) -> BBox:
        dt = max(0, frame_id - self.last_frame)
        if dt == 0 or (self.velocity[0] == 0.0 and self.velocity[1] == 0.0):
            return self.bbox
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        return (
            self.bbox[0] + dx,
            self.bbox[1] + dy,
            self.bbox[2] + dx,
            self.bbox[3] + dy,
        )


class SimpleByteTracker:
    """Small dependency-free two-stage IoU tracker.

    It follows ByteTrack's useful high-confidence-first association idea and is
    used as a safe fallback. The adapter also supports loading the official
    ByteTrack implementation when it is placed under third_party/byte_track.
    """

    def __init__(self, config: dict[str, Any]):
        self.high_threshold = float(config.get("high_threshold", 0.5))
        self.low_threshold = float(config.get("low_threshold", 0.15))
        self.match_threshold = float(config.get("match_threshold", 0.3))
        self.max_age = int(config.get("max_age", 30))
        self.min_hits = int(config.get("min_hits", 2))
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def _center(self, bbox: BBox) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _cost(self, track: _Track, detection: Detection, frame_id: int, frame_shape: tuple[int, ...]) -> float:
        pred_bbox = track.predicted_bbox(frame_id)
        iou_pred = bbox_iou(pred_bbox, detection.bbox)
        iou_last = bbox_iou(track.bbox, detection.bbox)
        iou = max(iou_pred, iou_last)

        tc = self._center(pred_bbox)
        dc = self._center(detection.bbox)
        h, w = frame_shape[:2]
        diag = max(1.0, math.hypot(w, h))
        distance = math.hypot(tc[0] - dc[0], tc[1] - dc[1]) / diag

        # Class matching penalty if classes disagree
        class_penalty = 0.0 if (track.class_id == detection.class_id or track.class_name == detection.class_name) else 0.25

        cost = (1.0 - 0.7 * iou - 0.3 * (1.0 - min(1.0, distance * 5.0))) + class_penalty
        return max(0.0, cost)

    def _match(
        self,
        tracks: list[_Track],
        detections: list[Detection],
        frame_id: int,
        frame_shape: tuple[int, ...],
        match_thresh: float | None = None,
    ) -> tuple[list[tuple[_Track, Detection]], list[_Track], list[Detection]]:
        if not tracks or not detections:
            return [], tracks, detections

        threshold = self.match_threshold if match_thresh is None else match_thresh

        num_tracks = len(tracks)
        num_dets = len(detections)
        cost_matrix = np.full((num_tracks, num_dets), 1e5, dtype=np.float32)

        for ti, track in enumerate(tracks):
            pred_bbox = track.predicted_bbox(frame_id)
            for di, detection in enumerate(detections):
                iou_pred = bbox_iou(pred_bbox, detection.bbox)
                iou_last = bbox_iou(track.bbox, detection.bbox)
                best_iou = max(iou_pred, iou_last)
                
                # Check center distance normalized by box size
                tc = self._center(pred_bbox)
                dc = self._center(detection.bbox)
                tw = max(1.0, pred_bbox[2] - pred_bbox[0])
                th = max(1.0, pred_bbox[3] - pred_bbox[1])
                dist_normalized = math.hypot((tc[0] - dc[0]) / tw, (tc[1] - dc[1]) / th)

                # Gate association: must have some overlap or be very close
                if best_iou >= threshold or dist_normalized <= 1.2:
                    cost_matrix[ti, di] = self._cost(track, detection, frame_id, frame_shape)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: list[tuple[_Track, Detection]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        # Maximum acceptable cost for a valid assignment
        max_cost = 1.0 - (0.5 * threshold)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_cost:
                matches.append((tracks[r], detections[c]))
                used_tracks.add(r)
                used_detections.add(c)

        remaining_tracks = [t for i, t in enumerate(tracks) if i not in used_tracks]
        remaining_detections = [d for i, d in enumerate(detections) if i not in used_detections]
        return matches, remaining_tracks, remaining_detections

    @staticmethod
    def _update_track(track: _Track, detection: Detection, frame_id: int) -> _Track:
        old_center = ((track.bbox[0] + track.bbox[2]) / 2.0, (track.bbox[1] + track.bbox[3]) / 2.0)
        new_center = ((detection.bbox[0] + detection.bbox[2]) / 2.0, (detection.bbox[1] + detection.bbox[3]) / 2.0)
        dt = max(1, frame_id - track.last_frame)
        raw_velocity = ((new_center[0] - old_center[0]) / dt, (new_center[1] - old_center[1]) / dt)
        
        # Adaptive velocity smoothing: if previously stationary, take new velocity faster
        alpha = 0.5 if track.hits > 1 else 0.8
        velocity = (
            alpha * raw_velocity[0] + (1.0 - alpha) * track.velocity[0],
            alpha * raw_velocity[1] + (1.0 - alpha) * track.velocity[1],
        )
        track.bbox = detection.bbox
        track.score = detection.score
        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.velocity = velocity
        track.last_frame = frame_id
        track.hits += 1
        track.missed = 0
        return track

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackObservation]:
        high = [d for d in detections if d.score >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.score < self.high_threshold]
        active = list(self._tracks.values())
        shape = frame.shape if frame is not None else (1, 1, 3)
        matches, remaining_tracks, remaining_high = self._match(active, high, frame_id, shape)
        for track, detection in matches:
            self._update_track(track, detection, frame_id)
        # A second pass associates low-confidence detections with tracks that
        # missed the high-confidence pass, as in ByteTrack's low-score stage.
        if low and remaining_tracks:
            # Slightly lower match threshold for second stage association
            low_matches, still_remaining, _ = self._match(
                remaining_tracks, low, frame_id, shape, match_thresh=max(0.15, self.match_threshold * 0.7)
            )
            for track, detection in low_matches:
                self._update_track(track, detection, frame_id)
            remaining_tracks = still_remaining
        for track in remaining_tracks:
            track.missed += 1
        for detection in remaining_high:
            track = _Track(
                track_id=self._next_id,
                bbox=detection.bbox,
                score=detection.score,
                class_id=detection.class_id,
                class_name=detection.class_name,
                velocity=(0.0, 0.0),
                last_frame=frame_id,
            )
            self._next_id += 1
            self._tracks[track.track_id] = track
        for track_id, track in list(self._tracks.items()):
            if track.missed > self.max_age:
                del self._tracks[track_id]
        return [
            TrackObservation(
                track_id=track.track_id,
                bbox=track.bbox,
                score=track.score,
                class_id=track.class_id,
                class_name=track.class_name,
            )
            for track in self._tracks.values()
            if track.last_frame == frame_id and track.hits >= self.min_hits
        ]


class OfficialByteTrackerAdapter:
    """Adapter for the upstream ByteTrack repository when it is available.

    The upstream project has historically exposed slightly different import
    paths, so the adapter accepts both common layouts. It is intentionally lazy
    and never required for the dependency-free fallback.
    """

    def __init__(self, config: dict[str, Any]):
        root = Path(__file__).resolve().parents[2] / "third_party" / "byte_track"
        if root.exists():
            sys.path.insert(0, str(root))
        self._impl: Any | None = None
        self._tracker: Any | None = None
        self._config = config
        self._next_fallback_id = 1
        self._class_by_id: dict[int, tuple[int, str]] = {}
        try:
            from yolox.tracker.byte_tracker import BYTETracker  # type: ignore

            self._impl = BYTETracker
        except Exception:
            try:
                from tracker.byte_tracker import BYTETracker  # type: ignore

                self._impl = BYTETracker
            except Exception:
                try:
                    from bytetrack import BYTETracker  # type: ignore

                    self._impl = BYTETracker
                except Exception as exc:
                    raise RuntimeError("official ByteTrack source is not importable") from exc

        class Args:
            track_high_thresh = float(config.get("high_threshold", 0.5))
            track_low_thresh = float(config.get("low_threshold", 0.15))
            track_thresh = float(config.get("high_threshold", 0.5))
            track_buffer = int(config.get("max_age", 30))
            match_thresh = float(config.get("match_threshold", 0.3))
            mot20 = False
            min_box_area = 10

        try:
            self._tracker = self._impl(Args(), frame_rate=float(config.get("frame_rate", 30.0)))
        except TypeError:
            self._tracker = self._impl(Args())

    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[TrackObservation]:
        if frame is None:
            raise ValueError("official ByteTrack requires a frame")
        if not detections:
            # The upstream API is stateful; call it with an empty array so it
            # can age tracks consistently.
            array = np.empty((0, 5), dtype=np.float32)
        else:
            array = np.asarray(
                [[*d.bbox, d.score] for d in detections], dtype=np.float32
            )
        height, width = frame.shape[:2]
        result = self._tracker.update(array, (height, width), (height, width))
        if not result:
            return []

        output: list[TrackObservation] = []
        # Official ByteTrack releases have returned both
        # (tlwh, ids, classes) and a list of STrack objects. Support both.
        if len(result) >= 2 and isinstance(result[0], (list, tuple, np.ndarray)):
            tlwh, ids = result[0], result[1]
            classes = result[2] if len(result) > 2 else None
            rows = []
            for i, track_id in enumerate(ids):
                if track_id <= 0:
                    continue
                rows.append((int(track_id), np.asarray(tlwh[i][:4], dtype=float), classes, i))
        else:
            rows = []
            for track in result:
                track_id = int(getattr(track, "track_id", 0))
                if track_id <= 0:
                    continue
                rows.append((track_id, np.asarray(getattr(track, "tlwh"), dtype=float), None, 0))

        for track_id, tlwh, classes, index in rows:
            x, y, w, h = [float(v) for v in tlwh[:4]]
            class_id = int(classes[index]) if classes is not None and index < len(classes) else -1
            class_name = str(class_id)
            # The upstream tracker returns boxes/IDs but not class IDs in this
            # API. Recover the class from the closest current detection and
            # retain it for unmatched frames.
            best_detection: Detection | None = None
            best_iou = 0.0
            for detection in detections:
                value = bbox_iou((x, y, x + w, y + h), detection.bbox)
                if value > best_iou:
                    best_iou, best_detection = value, detection
            if best_detection is not None:
                class_id = best_detection.class_id
                class_name = best_detection.class_name
                self._class_by_id[track_id] = (class_id, class_name)
            elif track_id in self._class_by_id:
                class_id, class_name = self._class_by_id[track_id]
            output.append(
                TrackObservation(
                    track_id=track_id,
                    bbox=(x, y, x + w, y + h),
                    score=1.0,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return output


def build_tracker(config: dict[str, Any]) -> Any:
    backend = str(config.get("backend", "simple_bytetrack")).lower()
    if backend in {"official", "bytetrack"}:
        try:
            return OfficialByteTrackerAdapter(config)
        except Exception as exc:
            # Keep the submission runnable if a developer forgot to add the
            # optional source tree.
            import warnings

            warnings.warn(f"official ByteTrack unavailable ({exc}); using fallback")
    return SimpleByteTracker(config)
