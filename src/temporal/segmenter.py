from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..contracts import RuleSignal


@dataclass
class _ActiveSample:
    timestamp: float
    confidence: float
    start_hint: float | None
    evidence: dict[str, Any]


class TemporalSegmenter:
    """Converts sampled rule flags into valid, non-overlapping intervals."""

    def __init__(self, config: dict[str, Any] | None = None, duration: float = 0.0):
        config = config or {}
        self.duration = max(0.0, float(duration))
        self.merge_gap = float(config.get("merge_gap", 1.0))
        durations = config.get("min_duration", {}) or {}
        self.min_duration = {str(k): float(v) for k, v in durations.items()}
        self._samples: dict[str, list[_ActiveSample]] = defaultdict(list)
        self._last_timestamp = 0.0
        self._sample_dt = 0.0

    def add(self, timestamp: float, signals: dict[str, RuleSignal] | dict[str, list[RuleSignal]]) -> None:
        """Record one sampled frame.

        Accepts either a single signal per label or a *list* of signals per
        label. The list form matters: several objects can satisfy one rule at
        once, and collapsing them into one flag would merge two events into one
        segment and lose one of them entirely.
        """
        timestamp = float(timestamp)
        if self._last_timestamp:
            self._sample_dt = max(self._sample_dt, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp
        for label, value in signals.items():
            entries = value if isinstance(value, list) else [value]
            for signal in entries:
                if signal.active and signal.confidence > 0.0:
                    self._samples[label].append(
                        _ActiveSample(
                            timestamp=timestamp,
                            confidence=float(signal.confidence),
                            start_hint=signal.start_hint,
                            evidence=dict(signal.evidence),
                        )
                    )

    @staticmethod
    def _instance_key(sample: _ActiveSample) -> tuple:
        """Identity of the object a sample refers to, for grouping.

        Two pedestrians jaywalking at once are two events, so samples are
        grouped per object rather than per class. The identity comes from the
        rule's own evidence, which every rule populates with the track ids it is
        about. A rule that reports a *group* condition (congestion) has no single
        track, and falls back to one bucket keyed by the direction it reports.
        """
        evidence = sample.evidence or {}
        keys: list[int] = []
        for field in ("track_id", "person_id", "vehicle_id", "first_id", "second_id"):
            value = evidence.get(field)
            if isinstance(value, int):
                keys.append(value)
        if keys:
            return tuple(sorted(set(keys)))
        direction = evidence.get("direction")
        if direction is not None:
            return ("group", str(direction))
        return ()

    def _groups(self, label: str) -> list[list[_ActiveSample]]:
        """Group a label's samples into events, one group per object.

        Samples are first bucketed by object identity and then split on temporal
        gaps within each bucket, so two objects active at the same time produce
        two groups rather than one merged group.
        """
        samples = sorted(self._samples.get(label, []), key=lambda item: item.timestamp)
        if not samples:
            return []
        buckets: dict[tuple, list[_ActiveSample]] = {}
        for sample in samples:
            buckets.setdefault(self._instance_key(sample), []).append(sample)
        groups: list[list[_ActiveSample]] = []
        for bucket in buckets.values():
            bucket.sort(key=lambda item: item.timestamp)
            current = [bucket[0]]
            for sample in bucket[1:]:
                if sample.timestamp - current[-1].timestamp <= self.merge_gap + 1e-6:
                    current.append(sample)
                else:
                    groups.append(current)
                    current = [sample]
            groups.append(current)
        groups.sort(key=lambda group: group[0].timestamp)
        return groups

    def finalize(self, duration: float | None = None) -> list[list[float | str]]:
        total_duration = self.duration if duration is None else float(duration)
        events: list[list[float | str]] = []
        for label, groups in self._groups_all():
            minimum = self.min_duration.get(label, 0.5)
            for group in groups:
                hinted_starts = [item.start_hint for item in group if item.start_hint is not None]
                start = min(hinted_starts) if hinted_starts else group[0].timestamp
                end = group[-1].timestamp + max(self._sample_dt, 1.0 / 25.0)
                start = max(0.0, min(start, total_duration))
                end = max(start, min(end, total_duration))
                if end <= start or end - start + 1e-6 < minimum:
                    continue
                events.append([round(start, 3), round(end, 3), label])
        # The harness rejects same-class overlaps, so segments of one class are
        # merged whenever they touch. Merging is preferred over dropping: two
        # objects violating the same rule at the same time cannot be reported
        # as two overlapping segments, and dropping one would lose an event
        # outright.
        events.sort(key=lambda event: (event[2], event[0], event[1]))
        merged: list[list[float | str]] = []
        for event in events:
            label = str(event[2])
            start, end = float(event[0]), float(event[1])
            if merged and str(merged[-1][2]) == label and start <= float(merged[-1][1]) + 1e-6:
                if end > float(merged[-1][1]):
                    merged[-1][1] = round(end, 3)
                continue
            merged.append([round(start, 3), round(end, 3), label])
        merged.sort(key=lambda event: (float(event[0]), str(event[2])))
        return merged

    def _groups_all(self) -> list[tuple[str, list[list[_ActiveSample]]]]:
        return [(label, self._groups(label)) for label in sorted(self._samples)]
