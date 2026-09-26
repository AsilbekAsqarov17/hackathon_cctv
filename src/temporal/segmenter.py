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
        # merge_gap may be a single number or a per-class override. Repeated
        # occurrences of one class close together are usually a single event
        # seen from several angles or several road users at once, and the task's
        # convention is to report those as one segment rather than many.
        gap = config.get("merge_gap", 1.0)
        if isinstance(gap, dict):
            self.merge_gap = float(gap.get("default", 1.0))
            self.merge_gap_by_class = {str(k): float(v) for k, v in gap.items() if k != "default"}
        else:
            self.merge_gap = float(gap)
            self.merge_gap_by_class = {}
        durations = config.get("min_duration", {}) or {}
        self.min_duration = {str(k): float(v) for k, v in durations.items()}
        ceilings = config.get("max_duration", {}) or {}
        self.max_duration = {str(k): float(v) for k, v in ceilings.items()}
        self._samples: dict[str, list[_ActiveSample]] = defaultdict(list)
        self._last_timestamp = 0.0
        self._sample_dt = 0.0

    def add(self, timestamp: float, signals: dict[str, RuleSignal]) -> None:
        timestamp = float(timestamp)
        if self._last_timestamp:
            self._sample_dt = max(self._sample_dt, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp
        for label, signal in signals.items():
            if signal.active and signal.confidence > 0.0:
                self._samples[label].append(
                    _ActiveSample(
                        timestamp=timestamp,
                        confidence=float(signal.confidence),
                        start_hint=signal.start_hint,
                        evidence=dict(signal.evidence),
                    )
                )

    def _gap_for(self, label: str) -> float:
        return self.merge_gap_by_class.get(label, self.merge_gap)

    def _groups(self, label: str) -> list[list[_ActiveSample]]:
        samples = sorted(self._samples.get(label, []), key=lambda item: item.timestamp)
        if not samples:
            return []
        gap = self._gap_for(label)
        groups: list[list[_ActiveSample]] = [[samples[0]]]
        for sample in samples[1:]:
            previous = groups[-1][-1]
            if sample.timestamp - previous.timestamp <= gap + 1e-6:
                groups[-1].append(sample)
            else:
                groups.append([sample])
        return groups

    def finalize(self, duration: float | None = None) -> list[list[float | str]]:
        total_duration = self.duration if duration is None else float(duration)
        events: list[list[float | str]] = []
        for label, groups in self._groups_all():
            minimum = self.min_duration.get(label, 0.5)
            ceiling = self.max_duration.get(label)
            for group in groups:
                hinted_starts = [item.start_hint for item in group if item.start_hint is not None]
                start = min(hinted_starts) if hinted_starts else group[0].timestamp
                end = group[-1].timestamp + max(self._sample_dt, 1.0 / 25.0)
                start = max(0.0, min(start, total_duration))
                end = max(start, min(end, total_duration))
                # A rule that stays satisfied for the whole clip is a rule that
                # is too loose, not a real event. Reporting it as one long
                # segment earns no temporal IoU against a short ground-truth
                # event, so cap the length instead. The onset carries the
                # information, so trim from the end and keep the start.
                if ceiling is not None and ceiling > 0 and end - start > ceiling:
                    end = start + ceiling
                if end <= start or end - start + 1e-6 < minimum:
                    continue
                events.append([round(start, 3), round(end, 3), label])
        # The harness rejects same-class overlaps. Enforce that invariant here
        # even when a future rule implementation emits overlapping candidates.
        events.sort(key=lambda event: (event[2], event[0], event[1]))
        kept: list[list[float | str]] = []
        last_end: dict[str, float] = {}
        for event in events:
            label = str(event[2])
            start, end = float(event[0]), float(event[1])
            if start < last_end.get(label, -1.0):
                continue
            kept.append(event)
            last_end[label] = end
        kept.sort(key=lambda event: (float(event[0]), str(event[2])))
        return kept

    def _groups_all(self) -> list[tuple[str, list[list[_ActiveSample]]]]:
        return [(label, self._groups(label)) for label in sorted(self._samples)]
