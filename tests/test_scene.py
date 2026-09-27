"""Tests for the calibrated scene: resolution mapping, stop line, signal matching.

These are the invariants every geometry-dependent rule silently depends on. A
violation of any of them does not raise: it just makes a rule stop firing, or
start firing on the wrong thing, which is far harder to notice. The same checks
are available standalone as ``scripts/check_scene_resolution.py``,
``scripts/check_line_span.py`` and ``scripts/check_stop_line_setup.py``; here
they run as part of the normal test suite so a regression cannot be committed.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.contracts import TrafficLightState
from src.scene.config import SceneContext, load_scene_config
from src.scene.geometry import normalized_vector, point_in_polygon

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "configs" / "scenes" / "data_video1.json"
AUTHORING = (3840, 2160)
SIZES = [(3840, 2160), (1920, 1080)]
# Lanes the single signal-controlled stop line governs (the eastbound carriageway).
GOVERNED_LANES = {2, 3}


class SceneResolutionTests(unittest.TestCase):
    def test_one_scene_serves_every_resolution(self) -> None:
        """The same authored scene must resolve to the same geometry at any size.

        The competition delivers this intersection at 3840x2160 and 1920x1080.
        Without `reference_size` the coordinates would be used verbatim and be
        2x wrong on the smaller video, which silently disables every
        geometry-dependent rule rather than raising.
        """
        scenes = {
            size: SceneContext(load_scene_config(SCENE, *size), *size) for size in SIZES
        }
        reference_w, reference_h = AUTHORING
        ref = scenes[AUTHORING]
        # Points individually verified at 4K during the geometry milestone.
        probes = [
            (2336, 1206, 2), (903, 760, 3), (2000, 674, 0), (2000, 822, 1),
            (1400, 825, 2), (1400, 1063, 3), (3529, 934, None), (3737, 2124, None),
        ]
        for x, y, expected in probes:
            base = ref.lane_for_point((x, y))
            base_id = base.lane_id if base else None
            self.assertEqual(base_id, expected, f"lane at {(x, y)} at 4K")
            for (w, h), scene in scenes.items():
                if (w, h) == AUTHORING:
                    continue
                scaled = (x * w / reference_w, y * h / reference_h)
                other = scene.lane_for_point(scaled)
                other_id = other.lane_id if other else None
                self.assertEqual(
                    other_id, base_id,
                    f"lane at {scaled} in {w}x{h} disagrees with 4K",
                )

    def test_reference_size_is_declared(self) -> None:
        import json

        data = json.loads(SCENE.read_text(encoding="utf-8-sig"))
        self.assertEqual(
            [int(v) for v in data["reference_size"]], list(AUTHORING),
            "the scene must declare the resolution its coordinates are written in",
        )

    def test_signal_rois_scale_with_the_frame(self) -> None:
        """A 4K ROI placed verbatim on a 1080p frame reads empty pixels.

        This is not hypothetical: it is why the earlier signal timeline for
        data_video2 was almost entirely "unknown", which would have left every
        signal-dependent rule silent on the half-resolution video.
        """
        big = load_scene_config(SCENE, *AUTHORING)
        small = load_scene_config(SCENE, 1920, 1080)
        for large, reduced in zip(big.traffic_lights, small.traffic_lights):
            for lo, hi, name in ((0, 2, "x"), (1, 3, "y")):
                expected = large.roi[lo] * 0.5
                self.assertAlmostEqual(reduced.roi[lo], expected, places=3,
                                       msg=f"{large.light_id} {name}0")
                self.assertAlmostEqual(reduced.roi[hi], large.roi[hi] * 0.5, places=3,
                                       msg=f"{large.light_id} {name}1")

    def test_scale_is_identity_at_the_authoring_resolution(self) -> None:
        """Mapping to the authoring size must be a no-op, or geometry drifts."""
        config = load_scene_config(SCENE, *AUTHORING)
        for lane in config.lanes:
            for x, y in lane.polygon:
                self.assertTrue(0.0 <= x <= AUTHORING[0])
                self.assertTrue(0.0 <= y <= AUTHORING[1])
        self.assertAlmostEqual(config.stop_lines[0].segment[0][0], 1824.0, places=3)


class StopLineGeometryTests(unittest.TestCase):
    def test_stop_line_spans_the_carriageway_it_governs(self) -> None:
        """A line covering only part of its lane cannot see a crossing on the rest.

        Measured before this was fixed: the previous segment covered the far 20%
        of the eastbound carriageway, so vehicles tracking through the inner part
        of the lane passed south of its lower end and never intersected it.

        The test is done in the line's own frame. The carriageway is diagonal, so
        a vertical slice would give a misleading answer: it intersects a
        slanted stop line at a shallow angle and reports low coverage even for a
        line that spans the lane completely.
        """
        for width, height in SIZES:
            config = load_scene_config(SCENE, width, height)
            line = config.stop_lines[0]
            (ax, ay), (bx, by) = line.segment
            length = math.hypot(bx - ax, by - ay)
            self.assertGreater(length, 0.0)
            u = normalized_vector((bx - ax, by - ay))
            assert u is not None
            v = (-u[1], u[0])
            for lane in config.lanes:
                if lane.lane_id not in GOVERNED_LANES:
                    continue
                # Extent of this lane projected onto the line (s) and onto its
                # normal (d). The line must cover the whole s extent, and the
                # lane must have points on both sides of it.
                along: list[float] = []
                across: list[float] = []
                xs = [p[0] for p in lane.polygon]
                ys = [p[1] for p in lane.polygon]
                for x in range(int(min(xs)) - 300, int(max(xs)) + 300, 8):
                    for y in range(int(min(ys)) - 300, int(max(ys)) + 300, 8):
                        if not point_in_polygon((x, y), lane.polygon):
                            continue
                        dx, dy = x - ax, y - ay
                        along.append(dx * u[0] + dy * u[1])
                        across.append(dx * v[0] + dy * v[1])
                self.assertTrue(along, f"lane {lane.lane_id} must meet the line")
                s0, s1 = min(along), max(along)
                covered = max(0.0, min(s1, length) - max(s0, 0.0))
                share = covered / (s1 - s0) if s1 - s0 > 1e-6 else 1.0
                self.assertGreater(
                    share, 0.95,
                    f"{width}x{height} lane {lane.lane_id}: line covers only {share:.0%}"
                    f" of the lane width (s {s0:.0f}..{s1:.0f}, line 0..{length:.0f})",
                )
                self.assertLess(min(across), 0.0)
                self.assertGreater(max(across), 0.0)

    def test_line_direction_points_the_way_traffic_travels(self) -> None:
        """`direction` decides which side is the approach, so queue suppression
        depends on it agreeing with the lanes it governs."""
        for width, height in SIZES:
            config = load_scene_config(SCENE, width, height)
            line = config.stop_lines[0]
            self.assertIsNotNone(line.direction)
            unit = normalized_vector(line.direction)
            assert unit is not None
            for lane in config.lanes:
                if lane.lane_id not in GOVERNED_LANES or lane.approach_point is None:
                    continue
                lane_unit = normalized_vector(lane.direction)
                assert lane_unit is not None
                dot = lane_unit[0] * unit[0] + lane_unit[1] * unit[1]
                self.assertGreater(dot, 0.9,
                                   f"line and lane {lane.lane_id} disagree on travel direction")

    def test_approach_points_are_upstream(self) -> None:
        """Queue suppression classifies a vehicle as queued only when it is
        upstream of the line, so a governed lane's approach point must be too."""
        for width, height in SIZES:
            config = load_scene_config(SCENE, width, height)
            line = config.stop_lines[0]
            origin = line.segment[0]
            direction = line.direction
            assert direction is not None
            for lane in config.lanes:
                if lane.lane_id not in GOVERNED_LANES or lane.approach_point is None:
                    continue
                p = lane.approach_point
                along = ((p[0] - origin[0]) * direction[0]
                         + (p[1] - origin[1]) * direction[1])
                self.assertLess(along, 0.0,
                                f"{width}x{height} lane {lane.lane_id} is not upstream")


class SignalMatchingTests(unittest.TestCase):
    def test_governing_signal_is_found_by_direction(self) -> None:
        """A head must be matched to a line by approach direction, not position.

        A signal head is mounted high, so its ROI projects above the road point
        it governs and can look upstream in the image; a positional test cannot
        work and would pair the wrong head with the line.
        """
        for width, height in SIZES:
            scene = SceneContext(load_scene_config(SCENE, width, height), width, height)
            line = scene.config.stop_lines[0]
            self.assertEqual(scene.signal_for_line(line), "signal_eastbound")

    def test_opposite_facing_head_is_never_used(self) -> None:
        """`signal_left` faces the other way, so it must not judge this line."""
        for width, height in SIZES:
            scene = SceneContext(load_scene_config(SCENE, width, height), width, height)
            line = scene.config.stop_lines[0]
            roi = next(l.roi for l in scene.config.traffic_lights
                       if l.light_id == "signal_left")
            scene.lights = {
                "signal_left": TrafficLightState("signal_left", "red", 1.0, roi),
                "signal_eastbound": TrafficLightState("signal_eastbound", "green", 1.0,
                                                       scene.config.traffic_lights[0].roi),
            }
            self.assertEqual(scene.signal_color_for_line(line), "green")

    def test_unreadable_signal_reports_unknown(self) -> None:
        """An unknown state must never be promoted to red.

        The official definition of `red_light` is a crossing on red. If an
        unreadable head returned anything else, the rule would invent a signal
        state and could report a violation that never happened.
        """
        for width, height in SIZES:
            scene = SceneContext(load_scene_config(SCENE, width, height), width, height)
            line = scene.config.stop_lines[0]
            scene.lights = {}
            self.assertEqual(scene.signal_color_for_line(line), "unknown")
            roi = next(l.roi for l in scene.config.traffic_lights
                       if l.light_id == "signal_eastbound")
            scene.lights = {
                "signal_eastbound": TrafficLightState("signal_eastbound", "unknown", 0.0, roi)
            }
            self.assertEqual(scene.signal_color_for_line(line), "unknown")


class CarriagewayTests(unittest.TestCase):
    def test_carriageway_only_is_strictly_narrower(self) -> None:
        """`jaywalking` depends on this distinction, and it is not obvious.

        Measured on data_video2: with the road polygons, 53% of all pedestrian
        bottom-centre samples counted as "on the road", because those polygons
        include the verges and footways around the junction. Restricting to the
        lane polygons is what makes the rule usable, and it must hold that
        carriageway implies road while the converse does not.
        """
        config = load_scene_config(SCENE, 1920, 1080)
        scene = SceneContext(config, 1920, 1080)
        # Every carriageway point is road.
        for lane in config.lanes:
            cx = sum(p[0] for p in lane.polygon) / len(lane.polygon)
            cy = sum(p[1] for p in lane.polygon) / len(lane.polygon)
            self.assertTrue(scene.is_road_point((cx, cy), carriageway_only=True),
                            f"lane {lane.lane_id} centre must be carriageway")
            self.assertTrue(scene.is_road_point((cx, cy)))
        # And there exist road points that are not carriageway, which is the
        # whole reason the distinction exists.
        found = None
        for y in range(0, 1080, 4):
            for x in range(0, 1920, 8):
                if (scene.is_road_point((x, y))
                        and not scene.is_road_point((x, y), carriageway_only=True)):
                    found = (x, y)
                    break
            if found:
                break
        self.assertIsNotNone(
            found,
            "expected the road polygons to cover more than the lanes at this camera; "
            "if this now fails, jaywalking no longer needs carriageway_only",
        )

    def test_pavement_point_is_not_carriageway(self) -> None:
        config = load_scene_config(SCENE, 1920, 1080)
        scene = SceneContext(config, 1920, 1080)
        for point in ((400.0, 60.0), (1500.0, 120.0), (300.0, 1000.0)):
            self.assertFalse(
                scene.is_road_point(point, carriageway_only=True),
                f"{point} should not be carriageway",
            )


if __name__ == "__main__":
    unittest.main()
