# Evidence frames

Images cited by the report, with the command that regenerates each one.
Regenerate all of them with `python scripts/collect_evidence.py`.

## `stop_line_placed_at_queue.jpg`

The recalibrated eastbound stop line, drawn in red, sits exactly where the queue's lead vehicle parks. data_video2.mp4 at t=161s: the lead car's nose is ~16px upstream of the line, matching the -17px measured by scripts/fit_stop_line_sweep.py over seven queue episodes.

```
python scripts/overlay_geometry.py data/data_video2.mp4 --times 161 --rect 700 300 1250 750 --scale 2.6
```

## `red_light_track203.jpg`

red_light firing at t=94.5s on data_video2. Track #203's front bumper has just crossed the stop line (front_along = +11.1px) while the governing head signal_eastbound reads red, and the vehicles queued behind it are stationary. This is the official definition: a vehicle crossing the stop line on red.

```
python scripts/inspect_rule_frame.py data/data_video2.mp4 --jsonl debug/perception/data_video2_tracks.jsonl --time 94.5 --rule red_light
```

## `eastbound_approach_lane_marking.jpg`

The eastbound approach at t=244.5s. The yellow line visible is a lane marking running parallel to travel, not a stop line; this camera's eastbound approach has no transverse painted stop line, so the line position is derived from where the queue's lead vehicles actually stop.

```
python scripts/crop_region.py data/data_video2.mp4 244.5 --rect 850 350 1200 600 --scale 3.4
```

## `queue_boxes_overlap.jpg`

A track pair with box IoU above 0.9 in ordinary queued traffic. Because the camera looks obliquely down the carriageway, the lead vehicle's box covers the one behind it, so image overlap is not contact evidence. This is why accident and near_miss judge separation in the road frame instead.

```
python scripts/crop_region.py data/data_video2.mp4 81.5 --rect 950 400 1450 800 --scale 3.0
```

## `scene_geometry_1080p.jpg`

The full scene geometry resolved at 1920x1080. The scene file is authored once at 3840x2160 and declares reference_size, so the same lanes, stop line, crossings and signal ROIs are mapped onto whatever frame size the video has.

```
python scripts/overlay_geometry.py data/data_video2.mp4 --times 40 --rect 1000 250 1400 800 --scale 3.0
```
