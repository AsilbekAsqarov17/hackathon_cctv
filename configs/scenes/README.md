# Scene configuration

Copy `default.json` to a camera-specific file such as `intersection_01.json`.
Coordinates may be pixels, or set `"normalized": true` and use values in the
range 0..1. A scene file is optional; without one, line/signal/turn rules
remain disabled while a conservative lower-frame road fallback supports basic
stopped-vehicle and congestion analysis.

Recommended fields:

- `lanes`: polygons, traffic direction vectors, allowed movements, and an optional `approach_point` before a stop line
- `stop_lines`: line segments and optional direction vectors
- `solid_lines`: lane markings that must not be crossed
- `crosswalks`: pedestrian crossing polygons
- `road_polygon`: optional road region; lanes are used if omitted
- `traffic_lights`: ROIs used by the HSV signal reader
- `auto_road_fallback`: set false to disable the lower-frame road assumption
