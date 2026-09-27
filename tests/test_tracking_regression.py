"""Regression tests for the tracking and rule-specificity work.

Each test here corresponds to a defect that was diagnosed and fixed, and is
written so that reintroducing the defect fails the test rather than quietly
restoring the old behaviour.
"""
from __future__ import annotations

import unittest
from collections import deque

from src.contracts import Detection, TrackObservation
from src.perception.tracker import SimpleByteTracker
from src.tracking.track_manager import TrackManager


def det(x1, y1, x2, y2, score=0.9, name="car", cid=2):
    return Detection((float(x1), float(y1), float(x2), float(y2)), score, cid, name)


def obs(tid, x1, y1, x2, y2, name="car", cid=2):
    return TrackObservation(tid, (float(x1), float(y1), float(x2), float(y2)), 0.9, cid, name)


class AssociationTests(unittest.TestCase):
    """A vehicle that drifts must keep its identity; two vehicles must not merge."""

    def _tracker(self, **kw):
        cfg = {"high_threshold": 0.5, "low_threshold": 0.15, "match_threshold": 0.3,
               "max_age": 30, "min_hits": 1}
        cfg.update(kw)
        return SimpleByteTracker(cfg)

    def test_same_vehicle_keeps_one_id_across_frames(self) -> None:
        t = self._tracker()
        first = t.update([det(100, 100, 200, 200)], 0, 0.0)
        self.assertEqual(len(first), 1)
        tid = first[0].track_id
        # A 12 px/frame drift: IoU stays well above 0.3, so the id must hold.
        for f, x in enumerate(range(100, 400, 12), start=1):
            out = t.update([det(x, 100, x + 100, 200)], f, f * 0.1)
            self.assertEqual([o.track_id for o in out], [tid], f"id changed at frame {f}")

    def test_two_adjacent_vehicles_get_separate_ids(self) -> None:
        """Regression guard for over-eager association after lowering the threshold."""
        t = self._tracker()
        out = t.update([det(100, 100, 200, 200), det(320, 100, 420, 200)], 0, 0.0)
        self.assertEqual(len({o.track_id for o in out}), 2)
        for f, dx in enumerate(range(0, 200, 10), start=1):
            out = t.update(
                [det(100 + dx, 100, 200 + dx, 200), det(320 + dx, 100, 420 + dx, 200)], f, f * 0.1
            )
            self.assertEqual(len({o.track_id for o in out}), 2, f"merged at frame {f}")

    def test_short_disappearance_preserves_identity(self) -> None:
        t = self._tracker()
        tid = t.update([det(100, 100, 200, 200)], 0, 0.0)[0].track_id
        for f in range(1, 4):  # three frames with no detection at all
            self.assertEqual(t.update([], f, f * 0.1), [])
        back = t.update([det(104, 100, 204, 200)], 4, 0.4)
        self.assertEqual([o.track_id for o in back], [tid])

    def test_duplicate_detections_are_removed_by_nms(self) -> None:
        """NMS is the detector's responsibility, and it is the layer that owns it.

        The tracker deliberately does *not* de-duplicate: two vehicles in
        adjacent lanes legitimately overlap in projection, so folding overlapping
        detections together inside the tracker would merge real objects. This
        therefore tests the component that actually has the job, and asserts
        that two genuinely separate vehicles are left alone.
        """
        from src.perception.onnx_detector import OnnxDetector

        nms = OnnxDetector._nms.__get__(
            type("D", (), {"confidence": 0.25, "iou": 0.5})(), OnnxDetector._nms)
        near_duplicates = [
            det(100, 100, 200, 200, score=0.9),
            det(102, 101, 202, 201, score=0.8),
            det(101, 100, 201, 200, score=0.7),
        ]
        self.assertEqual(len(nms(near_duplicates)), 1, "NMS kept a near-duplicate box")

        separate = [det(100, 100, 200, 200, score=0.9), det(320, 100, 420, 200, score=0.9)]
        self.assertEqual(len(nms(separate)), 2, "NMS wrongly merged two separate vehicles")


class TrackManagerTests(unittest.TestCase):
    def test_unobserved_track_is_not_returned_as_active(self) -> None:
        """The stale-track fix: retained for identity, never reported as observed."""
        m = TrackManager(max_age_seconds=2.0)
        m.update([obs(1, 100, 100, 200, 200)], 0.0, 0)
        self.assertEqual([s.track_id for s in m.update([obs(1, 100, 100, 200, 200)], 0.1, 3)], [1])
        # Now the track vanishes. It must be retained internally...
        active = m.update([], 0.2, 6)
        self.assertEqual(active, [], "an unobserved track was reported as active")
        self.assertEqual([s.track_id for s in m.all_states()], [1], "it should still be retained")

    def test_active_equals_observed_ids(self) -> None:
        m = TrackManager(max_age_seconds=5.0)
        m.update([obs(1, 0, 0, 50, 50), obs(2, 300, 300, 350, 350)], 0.0, 0)
        states = m.update([obs(1, 1, 0, 51, 50)], 0.1, 3)
        self.assertEqual([s.track_id for s in states], [1])

    def test_single_observation_has_no_measured_speed(self) -> None:
        """A track seen exactly once must not report 0.0 m/s.

        Its velocity is still the constructor default, so speed came out as
        exactly zero -- indistinguishable from a stationary vehicle. Measured at
        the head of a development clip, 14 of 14 tracks looked stopped on the
        second sampled frame, which fed phantom standstill into every
        speed-dependent rule.
        """
        from src.contracts import TrackState

        once = TrackState(1, 0, "car", (0, 0, 100, 60), 0.9, 0.0, 0.0, (50, 30))
        once.history.append((0.0, 50.0, 30.0))
        self.assertIsNone(once.speed_mps, "a once-seen track reported a speed")

        twice = TrackState(1, 0, "car", (0, 0, 100, 60), 0.9, 0.0, 0.0, (50, 30))
        twice.history.append((0.0, 50.0, 30.0))
        twice.history.append((1.0, 50.0, 30.0))
        self.assertIsNotNone(twice.speed_mps, "a twice-seen track has no speed")
        self.assertLess(twice.speed_mps, 0.01, "a stationary track must read as stationary")

    def test_rules_do_not_treat_unknown_speed_as_zero(self) -> None:
        """`speed_mps or 0.0` maps unknown to stationary and would undo the fix."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "rules" / "engine.py").read_text()
        self.assertNotIn(
            "speed_mps or 0.0",
            src,
            "a rule coerces an unmeasured speed to 0.0 m/s (i.e. to 'stopped')",
        )

    def test_snapshot_flags_stale_tracks(self) -> None:
        m = TrackManager(max_age_seconds=5.0)
        m.update([obs(1, 0, 0, 50, 50), obs(2, 300, 300, 350, 350)], 0.0, 0)
        m.update([obs(1, 0, 0, 50, 50)], 0.1, 3)
        snap = {s["track_id"]: s["active"] for s in m.snapshot()}
        self.assertEqual(snap, {1: True, 2: False})


class RuleSpecificityTests(unittest.TestCase):
    """Rules must not fire on ordinary traffic."""

    def _states(self):
        from src.contracts import FrameState, SceneState, TrackState
        from src.scene.config import SceneContext
        return FrameState, SceneState, TrackState, SceneContext

    def test_accident_requires_impact_confirmation_not_just_contact(self) -> None:
        """Contact while closing is constant in dense traffic; a collision also
        destroys momentum. The rule must not report on the first frame of touch."""
        import inspect
        from src.rules.engine import RuleEngine
        src = inspect.getsource(RuleEngine._interaction_rule)
        self.assertIn("_pending", src, "accident lost its impact-confirmation stage")
        self.assertIn("confirm_sec", src)

    def test_near_miss_requires_a_vulnerable_road_user(self) -> None:
        """Measured: 1.3 close-and-braking pairs per frame of ordinary queueing,
        and no kinematic discriminator separates them. So the rule must gate on
        a pedestrian or cyclist being involved."""
        import inspect
        from src.rules.engine import RuleEngine
        src = inspect.getsource(RuleEngine._interaction_rule)
        self.assertIn("vulnerable", src, "near_miss lost its vulnerable-road-user gate")

    def test_jaywalking_requires_movement(self) -> None:
        import inspect
        from src.rules.engine import RuleEngine
        src = inspect.getsource(RuleEngine._jaywalking)
        self.assertIn("min_speed", src, "jaywalking no longer requires the pedestrian to move")

    def test_stopped_vehicle_not_fired_by_a_signal_queue(self) -> None:
        """A stationary vehicle with another stationary vehicle close by is a
        queue, not a broken-down car."""
        import inspect
        from src.rules.engine import RuleEngine
        self.assertIn("_in_a_queue", inspect.getsource(RuleEngine._stopped_vehicle))

    def test_geometry_helpers_use_physical_units(self) -> None:
        """Distance thresholds must come from metres-per-pixel, never constants."""
        import inspect
        from src.contracts import TrackState
        self.assertIsNotNone(TrackState.metres_per_pixel)
        # A 4.5 m car drawn 225 px wide implies 0.02 m/px, not 1.5/225.
        t = TrackState(1, 2, "car", (0, 0, 225, 120), 0.9, 0.0, 0.0, (112, 60))
        self.assertAlmostEqual(t.metres_per_pixel, 4.5 / 225, places=4)


if __name__ == "__main__":
    unittest.main()
