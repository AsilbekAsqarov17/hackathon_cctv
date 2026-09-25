# Repository study notes

These notes record how the referenced projects feed the Part A design. They do
not imply that code is copied.

## Smart-Traffic-Analysis-With-Yolo

Source: https://github.com/bahakizil/Smart-Traffic-Analysis-With-Yolo

The project has a `LaneDetector`/`LaneVehicleProcessor` structure and combines
object detection, ByteTrack, lane assignment, vehicle counts, and speed/flow
statistics. We keep the reference `main.py` snapshot under
`third_party/smart_traffic/` and use its lane/trajectory ideas through our
`SceneContext`, `TrackManager`, and `TrafficAnalytics` adapter.

## SignalWatch

Source: https://github.com/som3a-web/signalwatch-traffic-detection

The useful concepts are calibration, stop lines, traffic-light state, geometric
rules, side-of-line crossing checks, heading-direction checks, and temporal
debounce/fired states for red-light/wrong-way events. No source is copied until
the repository license is clarified. Our scene schema, signal reader, and rule
state machines are an independent implementation of the concepts.

## YOLO-Traffic-Analytics

Source: https://github.com/vini-mon/YOLO-Traffic-Analytics

Use its trajectory, direction, density, and stationary-vehicle ideas as
references. The competition runtime uses our own tracker adapter and feature
pipeline.

## Official ByteTrack

Source: https://github.com/FoundationVision/ByteTrack

A selected tracker-core snapshot is under `third_party/byte_track/` with its
MIT license. The dependency-free tracker is the default until the official
tracker's optional dependencies are available in the offline environment.
