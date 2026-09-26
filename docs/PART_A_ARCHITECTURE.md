# Part A architecture

## Runtime

```text
VideoReader
  -> OnnxDetector (YOLO11n via ONNX Runtime)
  -> SimpleByteTracker (two-stage IoU association)
  -> TrackManager
  -> SceneContext
  -> RuleEngine
  -> TemporalSegmenter
  -> [[start_sec, end_sec, label], ...]
```

## Canonical state

`TrackManager` is the only source of track state. It maintains bounding boxes,
centres, smoothed velocity and acceleration, lane history, ground-contact
history and persistence. The default tracker is the dependency-free two-stage
IoU association in `src/perception/tracker.py`; the official ByteTrack adapter
in `third_party/byte_track/` is available but needs `lap`, `cython_bbox` and
`scipy`, none of which ship manylinux wheels, so it is not the default. That is
a deliberate trade: a failed install on the evaluation host would score the run
zero, which is far worse than slightly weaker track identity.

An imported tracker must convert its output to `TrackObservation` and must not
create a second independent track history.

## Scale

`TrackState.metres_per_pixel` derives image scale from the apparent box height
against a nominal real-world height for the class. Every distance and speed
threshold in `configs/default.json` is therefore in metres or m/s, not pixels.
This is not a refinement: the first version of these thresholds was written in
pixels, and at 3840x2160 a car at 30 km/h moves roughly 700 px/s, so
`stopped_vehicle` at 3 px/s and `congestion` at 8 px/s could never fire while
`near_miss` and `accident` fired almost continuously. Callers must treat `None`
from `speed_mps` / `metres_per_pixel` as "scale unknown" rather than substituting
a pixel constant, which is the bug this replaced.

## Rules

Rules receive a `FrameState` and return `RuleSignal(active, confidence,
evidence, start_hint)`. They never open a video or run a model. `start_hint`
carries the onset of a condition so the segmenter can start the event where the
behaviour began rather than where the sample grid happened to fall.

`accident` and `near_miss` are transient by definition, so they test contact and
deceleration rather than sustained proximity. `accident` uses the edge-to-edge
gap between boxes in metres, because box *centres* are a poor proxy for contact:
two cars in adjacent lanes have close centres while their bodies are a metre
apart.

## Temporal post-processing

`TemporalSegmenter` converts sampled active flags into valid intervals, applies
per-class `min_duration` and `max_duration`, merges gaps up to `merge_gap`, and
enforces the no-same-class-overlap invariant. The maximum duration matters: a
rule that never deactivates is a rule that is too loose, and one 60-second
segment earns no temporal IoU against a short ground-truth event.

## Scene configuration

Geometry is external to the rule code. A scene JSON file defines lanes,
directions, allowed turns, stop lines, solid markings, crosswalks, road
polygons, traffic-light ROIs, and an optional homography. All coordinates are
normalized, so one file serves any resolution of the same camera.

Test videos are named `test_001.mp4`, so the per-video lookup
`configs/scenes/<stem>.json` never matches and every test video resolves to
`configs/scenes/default.json`. The calibrated camera file must therefore *be*
the default, or the geometry-dependent rules silently disable themselves.

Use `scripts/preview_scene_fill.py` rather than an outline-only preview when
checking a road polygon: an outline that grazes a pavement looks identical to one
that covers it, and filling the region is the only way to see the difference.

## Relationship to Part B

Part A and Part B share code, not results. `RiskEstimator` runs its own
detector, tracker and rules on the frames it is handed and reads nothing Part A
produced. See `docs/PART_B_ARCHITECTURE.md`.
