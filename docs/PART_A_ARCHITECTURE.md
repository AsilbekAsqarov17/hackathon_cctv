# Part A architecture

## Runtime

```text
VideoReader
  -> DetectorAdapter
  -> ByteTrackAdapter
  -> TrackManager
  -> SceneContext
  -> RuleEngine
  -> TemporalSegmenter
  -> [[start_sec, end_sec, label], ...]
```

## Canonical state

`TrackManager` is the only source of track state. It maintains bounding boxes,
centers, velocity, acceleration, lane history, and persistence. The default
configuration requests the official ByteTrack adapter when its optional
dependencies are installed; a dependency-free two-stage fallback is used when
they are unavailable. Imported tracker code must convert its output to
`TrackObservation` and must not create a second independent track history.

## Rules

Rules receive a `FrameState` and return `RuleSignal(active, confidence)`. They
never open a video or run a model. `TemporalSegmenter` converts sampled active
flags into valid intervals, applies minimum durations and merge gaps, and
enforces the no-same-class-overlap invariant.

## Scene configuration

Geometry is external to the rule code. A scene JSON file can define lanes,
directions, allowed turns, stop lines, solid markings, crosswalks, the road
polygon, traffic-light ROIs, and an optional homography. Without a scene file,
road and signal rules are disabled rather than guessed.

## Part B boundary

`RiskEstimator` remains a placeholder in `solution.py`. Part A is complete and
testable independently with the harness option `--no-risk`.
