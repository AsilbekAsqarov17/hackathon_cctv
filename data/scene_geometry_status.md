# Scene geometry status — data_video1

**STATUS: CALIBRATED, LANES NOW DIRECTIONAL.** Superseded in part — see
`camera.md` and `configs/scenes/data_video1.json` for the authoritative
description.

- `configs/scenes/data_video1.json` — authored geometry: 4 road polygons,
  4 **directional** lanes, 1 stop line, 6 solid lines, 4 crossings,
  2 verified traffic-light ROIs. `has_geometry = true`, `has_road = true`.
- `configs/scenes/data_video1_uncalibrated.json` — retained as the safe
  fallback; still reports `has_geometry = false` so every geometry rule is
  disabled.
- `camera.md` — human-readable scene description, method, verification and
  known ambiguity.

## Current state (supersedes the sections below)

Everything the "What is still needed (human task)" list asked for has since been
done: `camera.md` exists and is authoritative, both signal ROIs were confirmed
by cropping the lamp through red→green cycles, and the lane polygons were
validated against `debug/perception/data_video1_tracks.jsonl` rather than by eye.

The lanes were then **recalibrated per direction**, which is the change that
unblocked `wrong_way`:

| | before | after |
|---|---|---|
| lane directions | all `(1.0, 0.11)` | `(-1.0,-0.30)` westbound, `(1.0, 0.47)` eastbound |
| opposing moving tracks | 545 of 1012 (54 %) | 10 of 750 (1.3 %) |
| `wrong_way` events over 10 200 frames | rule disabled (empty allowlist) | 0, with the rule verified to fire on synthetic violations |
| `stopped_vehicle` events | 7 | 7 (unchanged) |

Method, the 2D evidence that showed the old horizontal bands were the problem,
and the two open items (no readable signal on the east leg; the stop line spans
only 20 % of the eastbound carriageway) are documented in `camera.md`.

The findings below are retained as the historical record of how the geometry was
originally assessed. The traffic-light finding is still relevant because it
constrains the runtime pipeline.

## Historical record


- `configs/scenes/data_video1.json` — authored geometry: 4 road polygons,
  4 lanes with direction vectors, 1 stop line, 6 solid lines, 4 crossings,
  2 verified traffic-light ROIs. `has_geometry = true`, `has_road = true`.
- `configs/scenes/data_video1_uncalibrated.json` — retained as the safe
  fallback; still reports `has_geometry = false` so every geometry rule is
  disabled.
- `camera.md` — human-readable scene description, method, verification and
  known ambiguity.

The findings below that concerned *missing* geometry have been resolved. The
traffic-light finding is retained because it still constrains the runtime
pipeline.

## Is `camera.md` available?

**At the time of writing, no.** There was no `camera.md` anywhere in the
repository. The only
camera-adjacent documents are:

- `data/scene_calibration.md` — records calibration *decisions*, not coordinates
- `configs/scenes/README.md` — describes the scene file schema
- `configs/scenes/data_video1_base.json` — road polygons + crosswalks only
- `configs/scenes/data_video1.json` — **unverified draft** with lanes, stop
  lines, solid lines and signal ROIs

## Interface coverage

`src/scene/config.py` already supports every element the milestone requires, so
no new interface was needed:

| Requirement            | Field                          | Status |
|------------------------|--------------------------------|--------|
| lanes                  | `lanes[].polygon`              | present |
| lane directions        | `lanes[].direction`            | present |
| stop lines             | `stop_lines[].segment`         | present |
| pedestrian crossings   | `crosswalks[].polygon`         | present |
| solid lines            | `solid_lines[].segment`        | present |
| traffic-light region   | `traffic_lights[].roi`         | present |

## The draft geometry is not usable

`configs/scenes/data_video1.json` was rendered over frame 7200 with
`scripts/preview_scene.py` (output: `data/inspection/scene_draft_overlay.jpg`).
It does not match the scene:

- lane polygons (magenta) cut diagonally across buildings, sidewalks, grass
  verges and pedestrian areas rather than following the carriageway;
- stop lines (red) and solid lines (yellow) are long diagonals crossing
  unrelated parts of the image, including pavement and a traffic island;
- road polygons (orange) cover the full frame width including non-road area;
- `ne_signal` lands on a grass/tree verge with no signal head.

Enabling rules on this draft would generate large volumes of false
`wrong_way` / `solid_line_crossing` / `red_light` events. This matches the
existing note in `data/scene_calibration.md`, which deliberately ships no stop
lines or signal ROIs for the first real-data run.

`configs/scenes/data_video1_uncalibrated.json` is therefore the honest state:
all geometry collections empty and `auto_road_fallback: false`, so
`has_geometry` and `has_road` are both `False` and every geometry-dependent rule
stays inactive (verified by `scripts/check_rule_gating.py`).

## Traffic lights: present in the scene, but not detected

This is the most important finding for `red_light`.

**Real, cycling signal heads do exist** in `data_video1.mp4`. Two were located by
eye and confirmed over time:
- a head near the central island at roughly 4K `(2245..2345, 695..825)` — green
  at t≈30 s and t≈210 s, red at t≈90 s, 150 s, 240 s, 300 s
  (`data/inspection/real_signal_zoom.jpg`);
- a head on the left approach at roughly 4K `(515..595, 950..1070)` — green at
  t≈30 s, red at t≈150 s, 240 s, 300 s (`data/inspection/left_signal_zoom.jpg`).

**Neither detector finds them.** Meanwhile the base checkpoint's
`traffic light` output on this video is entirely false positives:

| Base-model "traffic light" box (4K) | What it actually is                     |
|---------------------------------------|-----------------------------------------|
| `(2418, 702, 2485, 785)` persistent    | blue pedestrian-crossing sign (static)  |
| `(429, 1035, 483, 1225)` at t≈210 s     | a pedestrian behind a railing           |
| `(1438, 475, 1499, 581)` at t≈300 s     | the back of a street-sign post          |

Evidence: `data/inspection/tl_zoom_montage.jpg` (the sign never changes colour;
the saturated-pixel count in that box stays ~2550 for the whole 320 s, which a
functioning signal head cannot do) and `data/inspection/tl_other_boxes.jpg`.

Consequences:

1. The hybrid traffic-light branch is **not currently functional** on this
   video. It localizes objects, but none of them are signals.
2. Colour classification is technically viable — the real heads show clean,
   unambiguous red/green — but only once the correct ROIs exist. The HSV reader
   in `src/scene/signals.py` is adequate for the resolution involved; the
   earlier `unknown`/spurious-`green` readings came from cropping the wrong
   object.
3. **Automated lamp localization is not yet safe.** A saturated-colour search
   over candidate windows also locks onto red-painted kerbs and red/white
   striped bollards, so measured candidates could not be separated from road
   furniture automatically.

## What is still needed (human task)

1. Author `camera.md` (or an equivalent calibrated scene file) for the
   competition camera, giving lanes, lane directions, stop lines, crossings,
   solid lines and signal ROIs.
2. Confirm each signal ROI by cropping it and checking the lamp cycles
   red→green, as done in the two inspection images above.
3. Validate lane polygons against stable bottom-center track positions in
   `debug/perception/data_video1_tracks.jsonl` rather than by eye.
4. Only then enable `stopped_vehicle`, `wrong_way`, `solid_line_crossing` and
   `red_light`.

Until then the correct behaviour is to emit no geometry-dependent events.
