# Camera / scene description — `data_video1`

Human-readable companion to `configs/scenes/data_video1.json`. All coordinates
are **source pixels** in the 3840×2160 frame (`normalized: false`).

## Camera

| Property | Value |
|----------|-------|
| Source | `data/data_video1.mp4` |
| Resolution | 3840 × 2160 (4K) |
| FPS | 29.97002997 |
| Duration / frames | 340.34 s / 10 200 |
| Viewpoint | Fixed, elevated CCTV, oblique downward view of a large urban intersection |

Because the challenge states the hidden test videos use the same
camera/viewpoint, these coordinates may be hard-coded.

## How this geometry was produced

1. **Read markings off labelled pixel grids.** `scripts/authoring_grid.py`
   renders any frame as overview / quadrant tile / arbitrary zoom with a labelled
   coordinate grid, so coordinates are read off the actual 4K pixels rather than
   estimated from a downscaled preview.
2. **Confirmed travel direction visually**, not from tracks. Each carriageway was
   checked at high zoom for windscreen, driver, wipers, mirrors, headlights,
   grille and plate position (e.g. `data/authoring/z_farbus.jpg`,
   `z_nearbus.jpg`, `z_blackcar.jpg`).
3. **Cross-checked with paint-mask measurement.** `scripts/measure_markings.py`
   isolates white/yellow paint and fits oriented rectangles. Used only to
   *propose* candidates; every accepted element was then confirmed by eye.
4. **Verified with overlays** on frames 0, 2400, 4618, 7200 and 9050 using
   `scripts/preview_scene.py` (`data/authoring/verify_*.jpg`).

Explicitly **not** done: geometry was not inferred from object detections, and
traffic-light ROIs were not found by automatic saturated-colour search. The
previous draft `data_video1.json` was discarded, not reused — its lane polygons
cut through buildings and verges.

## Road layout and directions

The frame contains a two-way main carriageway falling from upper-left to
lower-right, which opens into a wide intersection box at lower-centre/right,
plus uncalibrated approach legs.

| Element | Description |
|---------|-------------|
| Far carriageway | Upper strip, **westbound** (decreasing image *x*); the green buses use it |
| Near carriageway | Lower strip of the same road, **eastbound**; carries the signalised queue |
| Median | Kerbed island with sign gantry between them; ends near x ≈ 2100 |
| Junction box | Large open asphalt, lower-centre/right, no direction asserted |
| East leg | Continues east past the box; one-way westbound above, one-way eastbound below |

**Direction convention.** Lanes are **directional** and are no longer all
eastbound. The main carriageway is a *two-way* road: a far/westbound carriageway
whose travel direction is decreasing image *x* and a near/eastbound carriageway
whose travel direction is increasing image *x*. Each vector is the measured
descent of its own band, not a single shared slope:

| Carriageway | Direction vector | Angle | Meaning |
|-------------|------------------|-------|---------|
| Far (upper) | `(-1.0, -0.30)` | −17.4° | westbound, image-left and up |
| Near (lower) | `(1.0, 0.47)` | +25.2° | eastbound, image-right and down |

The east leg, the lower intersection and everything east of x = 2600 assert no
direction at all (see Ambiguity).

## Lanes (4, directional)

The bands are least-squares fits to measured stream percentiles, not hand-drawn.
`scripts/measure_stream_bands.py` sampled 11 x-stations over all 10 200 frames
with turning traffic excluded (`|vx| > 0.8·|vy|` and ≥ 150 px of travel over
4 s). Measured westbound `p05..p95` then eastbound `p05..p95`:

| x | 668 | 914 | 1405 | 2141 | 2386 | 2632 | 2877 |
|---|-----|-----|------|------|------|------|------|
| westbound | 241-380 | 282-444 | 439-713 | 673-885 | 716-991 | 807-1115 | 845-1156 |
| eastbound | 432-856 | 493-945 | 726-1103 | 1221-1570 | 1187-1662 | 1242-1760 | 1346-1974 |

Fitted edges (`y = a + b·x`), all in source pixels:

| Edge | Fit | Role |
|------|-----|------|
| `W_lo` | `40 + 0.280x` | far edge of the westbound band |
| `W_mid` | `85.5 + 0.331x` | divides lane 0 from lane 1 |
| `W_hi` | `131 + 0.382x` | near edge of the westbound band |
| `E_lo` | `171 + 0.382x` | far edge of the eastbound band |
| `E_mid` | `305.5 + 0.456x` | divides lane 2 from lane 3 |
| `E_hi` | `440 + 0.530x` | near edge of the eastbound band |

`W_hi` and `E_lo` are parallel and 40 px apart, so a constant-width median
separates the streams and `lane_for_point` is never ambiguous. All six lane
pairs were verified **disjoint** on a 10 px grid.

| ID | Name | x extent | Direction | approach_point |
|----|------|----------|-----------|----------------|
| 0 | `W_outer` | 200 → 3400 | `(-1.0, -0.30)` | — |
| 1 | `W_inner` | 200 → 3400 | `(-1.0, -0.30)` | — |
| 2 | `E_inner` | 200 → **2600** | `(1.0, 0.47)` | `(1600, 900)` |
| 3 | `E_outer` | 200 → **2600** | `(1.0, 0.47)` | `(1600, 1200)` |

Lane polygons are the edge polylines above, sampled at
`x = 200, 800, 1400, 2000, 2600` (eastbound) and `…, 3000, 3400` (westbound).

**Why the extents are clipped.** East of x ≈ 2600 the layout is a junction, not a
two-way carriageway. Measurement there shows one-way legs and a *different*
two-way road whose streams are separated along **x**, not y:

| region | measured dominant flow |
|--------|------------------------|
| y 720-1200, x 2640-3360 | ~100 % westbound (one-way leg) |
| y 1200-1920, x 2160-3600 | ~100 % eastbound (one-way leg) |
| y 1680-2160, x 0-1920 | ~100 % westbound (lower road) |

Directional reasoning is therefore asserted only where the two streams were
*measured* to be cleanly separated. The westbound band narrows after x = 2600
(356 → 270 → 208 px wide) to follow the measured width. Extending the eastbound
lanes to x = 2500 instead of 2600 changed the opposing share by 0.1 points
(3.5 % → 3.6 %), so 2600 was kept for the extra real road it covers.

Lanes 2 and 3 carry the approach point used by `red_light` to decide which side
of the stop line a vehicle came from. Both are verified upstream of the stop
line (`along = -498` and `-465`).

## Road polygons (4)

| Polygon | Extent |
|---------|--------|
| A band | far kerb `(0,225)…(2500,588)` down to median `(0,430)…(2500,790)`, x ≤ 2500 |
| B band | median down to near kerb `(0,990)…(2500,1105)`, x ≤ 2500 |
| East leg | `(2000,790) (2600,800) (3200,780) (3840,740) … (3840,1180) … (2000,1120)` |
| Lower intersection | `(900,1300) (1800,1200) (2700,1150) (3840,1180) … (900,2160)` |

## Stop line (1)

| ID | Segment | Direction | Governed by |
|----|---------|-----------|-------------|
| `b_eastbound_stop` | `(1824, 804) → (1574, 1334)` | `(1.0, 0.47)` | `signal_eastbound` |

**Re-derived from measurement.** The previous segment `(2120,700)→(2175,1110)`
was wrong in two independent ways, and both had to be fixed before `red_light`
could be implemented at all. The derivation is reproducible:

```bash
python scripts/fit_stop_line_sweep.py debug/perception/data_video2_tracks.jsonl \
    --width 1920 --height 1080
```

**Problem 1 — the line was anchored to the wrong crossing.** A signal head is
mounted several metres up, so its ROI projects well above the road point it
governs and appears *upstream* in the image. An "is the head downstream of the
line" test therefore cannot be used to pair a head with a line. Heads are
matched by **declared approach direction** instead
(`SceneContext.signal_for_line`, each ROI carries the approach it faces).
Given the governing head at `(1160, 384)` (1080p), the crossing traffic queues
for is the nearest one, `crossing[1]` at `(1040, 276)`, 161 px away — **not**
`crossing[0]` at 207 px. The old line was anchored to `crossing[0]`, which put
it *inside the junction*, where ordinary through traffic crossed it.

**Problem 2 — the line covered only the top of the carriageway.** It spanned
y 700-1110 while the recalibrated eastbound carriageway spans y ≈ 991-1578 at
that x, i.e. the far 20 %. A vehicle tracking through the inner part of the lane
passed south of the line's lower end and never intersected it. The new segment
is extended perpendicular to travel until it leaves the governed carriageway on
both sides, and `scripts/check_line_span.py` asserts it spans and cuts lanes 2
and 3 (100 % coverage at both 4K and 1080p).

**Position verified against the queue, not by eye.** The line is swept upstream
from the crossing and scored on where the vehicles that *held* the queue parked.
Only leads that stopped upstream of the crossing count; a vehicle that stopped
past it was blocked by the junction, not by the signal, and including it is what
dragged the earlier calibration into the middle of the intersection. At an 80 px
setback the queue holders sit at **−19 px and −17 px** from the line, and every
other setback is worse:

| Setback (px) | median &#124;offset&#124; |
|---|---|
| 0 | 99 |
| 40 | 59 |
| **80** | **19** |
| 120 | 23 |
| 160 | 63 |
| 200 | 103 |

Note this approach has **no transverse painted stop line**; the yellow line
visible on the approach is a lane marking running parallel to travel
(`docs/evidence/eastbound_approach_lane_marking.jpg`). The calibrated position
is therefore the functional equivalent: where drivers actually stop.

`direction` is now the measured eastbound travel direction `(1.0, 0.47)`, which
is both physically correct and what queue suppression needs to decide which side
is the approach. Both lane approach points remain upstream of it
(−179 px and −38 px at 4K), verified by `scripts/check_stop_line_setup.py`.

There is deliberately **no** stop line for the westbound carriageway: its signal
head has not been identified, so one would be a guess.

## Solid lines (6)

| ID | Segment | Notes |
|----|---------|-------|
| `a_far_edge` | `(0,205) (900,340) (1800,480) (2500,580)` | Carriageway A far kerb line |
| `a_median_edge` | `(0,430) (900,560) (1800,690) (2500,790)` | Median kerb |
| `b_near_edge` | `(0,990) (900,1050) (1800,1090) (2500,1105)` | Carriageway B near edge |
| `island_center_far` | `(1024,1365) → (1493,1493)` | Kerb of the upper traffic island |
| `island_center_near` | `(1365,1707) → (2240,1685)` | Kerb of the middle traffic island |
| `island_lower_far` | `(533,1877) → (960,2027)` | Kerb of the lower traffic island |

The three island lines were each confirmed to lie on a brick-filled traffic
island, not on open road.

## Pedestrian crossings (4)

| ID | Polygon |
|----|---------|
| `xwalk_main` | `(2190,990) (3080,920) (3080,1080) (2190,1150)` |
| `xwalk_upper` | `(1813,500) (2347,460) (2347,600) (1813,645)` |
| `xwalk_center` | `(700,1184) (1920,1024) (1920,1280) (789,1323)` |
| `xwalk_lower` | `(480,1340) (1880,2060) (1620,2160) (400,1660)` |

`xwalk_lower` is the large white/**yellow** striped crossing on the lower
approach. `xwalk_upper` sits in deep shadow on some frames but its stripes were
confirmed at high zoom (`data/authoring/z_xwalk_upper.jpg`).

## Traffic signals (2)

Both were found **by eye** at high zoom and then confirmed by state change.
Crops: `data/authoring/signal_signal_eastbound_states.jpg` and
`signal_signal_left_states.jpg`.

| ID | ROI `(x1,y1,x2,y2)` | Governs |
|----|---------------------|---------|
| `signal_eastbound` | `(2293,708,2345,828)` | Eastbound queue on carriageway B |
| `signal_left` | `(513,953,574,1054)` | Left/near approach head |

`signal_eastbound` is the right-facing head of a twin head on one pole; the
left-facing head serves cross traffic and is not configured. The ROI covers the
whole 3-lamp housing, so the unlit lamps and the dark visor area dominate the
crop and the reported confidence is low (≈0.07–0.26) even though the colour is
correct. `red_light` keys on `state.color == "red"` only, so confidence does not
gate the decision.

### State component

`src/perception/traffic_lights.py` → `classify_traffic_light_state(frame, bbox)`
reuses the existing HSV reader in `src/scene/signals.py` and returns
`red | yellow | green | unknown`. It is deliberately kept separate from
localization and from the `red_light` event.

Verified on the two ROIs (matches visual inspection 6/6 on each):

| t | `signal_eastbound` | `signal_left` |
|---|--------------------|---------------|
| 30 s | green | green |
| 90 s | red | red |
| 150 s | red | red |
| 210 s | green | green |
| 240 s | red | red |
| 300 s | red | red |

**Negative controls — must never read red.** Blue pedestrian-crossing sign,
open asphalt, plain road and grass all returned `unknown` or `green`, never
`red`. This matters because the base checkpoint's `traffic light` detections on
this video are *false positives* (a blue crossing sign, a pedestrian behind a
railing, a sign post). Those detections are not used for state; only the two
configured, visually verified ROIs are.

The three separations are preserved end to end:

```
localization  (where a head is)  !=  state (what colour it shows)  !=  red_light event
```

`red_light` additionally requires the vehicle to cross the stop line and to be
in a lane whose direction points into the intersection.

## Assumptions

1. Hidden test videos share this camera, so pixel coordinates transfer directly.
2. The fixed camera means no registration drift; a single scene file suffices.
3. Directional lanes stop at x = 2600 (eastbound) and narrow after x = 2600
   (westbound) because that is where the layout becomes a junction whose
   movements are not separable; lanes are not extended across the box.
4. `normalized: false`; all values are source pixels.
5. Lane direction is expressed as an image-space vector, matching
   `src/scene/config.py::Lane.direction`.
6. Lane polygons are straight-edged bands through the fitted edge lines rather
   than curves; over the 3.2 km-equivalent span the worst deviation from the
   measured percentile envelope is about 60 px, well inside the 40 px median
   plus per-lane half-width.

## Known ambiguity / not yet calibrated

1. **The junction is bidirectional and has crossing approaches.** At high zoom a
   sedan near `(3300, 1080)` clearly shows its **rear** (two taillights and a
   plate) and travels up-left, while adjacent vehicles face right. Therefore
   **no lane direction is asserted for the east leg or the lower intersection** —
   they are `road_polygons` only. This is deliberate: a single direction there
   would be a guess and would generate false `wrong_way` events. `wrong_way`
   is restricted to lanes 0-3, which stop at x = 2600 for exactly this reason.
1b. **RESOLVED — the main carriageway is now split per direction.**
   Previously lanes 0-3 spanned y ≈ 225-1105 with a single `[1.0, 0.11]`
   direction, and 82.8 / 70.6 / 42.7 / 8.9 % of moving tracks opposed it. The
   cause was methodological as much as geometric: a *horizontal* y-band analysis
   is invalid here because the road descends to the right, so one band cuts
   across several carriageways at different x. Re-measured in 2D
   (`scripts/flow_map_2d.py`, `scripts/measure_stream_bands.py`), the two
   streams are **parallel diagonal bands** of one two-way road, and every
   horizontal band between y 300 and y 1200 is 37-62 % mixed:

   | y band | 100 | 300 | 400 | 500 | 600 | 800 | 1000 | 1200 | 1400 | 2100 |
   |--------|-----|-----|-----|-----|-----|-----|------|------|------|------|
   | eastbound % | 0.0 | 17.0 | 37.1 | 53.3 | 13.6 | 62.5 | 48.1 | 59.1 | 87.9 | 91.1 |

   The old bands fell at slope ≈ 0.15 while the real carriageway falls at
   0.31-0.53, which is *why* they mixed the streams. Lanes were rebuilt from the
   fitted band edges (see Lanes above); the old vectors were **not** flipped,
   because a flipped vector would be wrong for the opposite half of the road and
   would also invert `red_light` and turn semantics.

   Opposing share, same 10 200 frames, same window, before → after:

   | lane | before | after |
   |------|--------|-------|
   | 0 | 83.8 % (197/235) | **0.7 %** (2/270) |
   | 1 | 70.4 % (178/253) | **0.5 %** (1/202) |
   | 2 | 43.8 % (153/349) | **1.8 %** (3/165) |
   | 3 | 9.7 % (17/175) | **3.5 %** (4/113) |

   Total 545 opposing tracks → 10. `wrong_way` then produced **0 events** over
   all 10 200 frames while still firing correctly on synthetic violations
   (`scripts/validate_wrong_way_detection.py`: with-flow silent, against-flow
   fires 2.2-2.3 s in all four lanes), so the zero is a real negative and not a
   dead rule.
2. **The east leg has no readable signal — measured, not assumed.** All seven
   `stopped_vehicle` events sit at x 2476-3812, downstream of the calibrated
   stop line, in the region deliberately left without direction. Tracks #3,
   #1098 and #1238 stop at the *same* point (3665,938), (3596,936),
   (3580,939) for 28 s, 20 s + 28 s and 11 s, sequentially, each about one
   vehicle behind the last. Crops (`data/authoring/east_leg/`) show that point is
   immediately beyond a zebra crossing beside a signal pole, and at t = 290 two
   pedestrians are standing in the road there. This is queue- or
   pedestrian-yield-like behaviour, **not** an isolated broken-down vehicle.

   The governing signal **cannot be read from this camera**: the two heads on
   that pole face away, and `scripts/probe_signal_roi.py` finds 0 saturated
   pixels in the upper head and only 18 peak yellow pixels in the lower head
   across 243 sampled frames, versus 8.7 saturated red pixels *per frame* for
   the calibrated `signal_eastbound`. A whole-frame sweep
   (`scripts/probe_signal_roi.py` grid mode) found no other candidate head on
   the east leg that both saturates and *changes* colour; the only strong
   sources are static objects (a permanently red sign at `(400,480)` with
   137.8 red px/frame and zero green, and a permanently yellow object at
   `(760,440)`), which cannot be signals.

   **Consequence:** the seven events are left **reported**, not suppressed,
   because suppressing them would require inventing a signal state. They are the
   top known false-positive risk for `stopped_vehicle` on this camera. Fixing it
   properly needs a second signal + stop line near `(3760, 850-1050)`, which is
   only possible from a viewpoint where that head faces the camera.
3. **The westbound carriageway has no signal or stop line.** Its head was not
   identified, so `red_light` cannot fire for that direction.
4. **The calibrated stop line spans only 20 % of the eastbound carriageway**
   (see Stop line). Queue suppression is unaffected; `red_light` is blocked on it.
5. **Lane boundaries inside the junction box are not modelled**; the box is road
   only, so no direction is claimed there.
6. **Signal confidence is low** because the ROI spans the full 3-lamp housing.
   Per-lamp ROIs would raise it, but the colour decision is already correct.
7. **The east leg bends.** The far kerb descends to about x = 3400 and then
   rises slightly, so the road polygons use extra vertices there rather than
   assuming a straight road.
8. **`xwalk_upper` is in shadow on many frames**; it is confirmed from zoom, not
   from a bright frame.

## Verification artefacts

| File | Purpose |
|------|---------|
| `data/authoring/verify_0.jpg` … `verify_9050.jpg` | Full geometry overlay, 4 frames |
| `data/authoring/overlay_v4_view.jpg` | Final authoring overlay, frame 4618 |
| `data/authoring/signal_*_states.jpg` | Per-signal red/green state crops |
| `data/authoring/real_signal_zoom.jpg`, `left_signal_zoom.jpg` | Signals at 4× over time |
| `data/authoring/z_farbus.jpg`, `z_nearbus.jpg`, `z_blackcar.jpg`, `z_silver.jpg` | Direction evidence |
| `data/authoring/paint_thin_view.jpg` | Paint-mask view of all markings |
| `data/authoring/flow_map_2d.jpg` | 2D eastbound/westbound occupancy map that revealed the two parallel bands |
| `data/authoring/directional_lanes/lanes_*.jpg` | Recalibrated lanes + directions over 5 real frames |
| `data/authoring/east_leg/crop_*.jpg` | Where the 7 east-leg stopped events occur |
| `data/authoring/east_signal/crop_*.jpg` | The east-leg signal heads, showing they face away |
| `debug/rules/data_video1_directional_debug.mp4` | Full debug render: wrong_way now enabled |
| `debug/rules/data_video1_directional_timeline.png` | Rendered event timeline |

## Reproducing the directional calibration

```bash
# 1. where each stream physically sits (2D, not horizontal bands)
python scripts/flow_map_2d.py debug/perception/data_video1_tracks.jsonl \
    --draw data/authoring/flow_map_2d.jpg

# 2. the band edges, per x-station, with turning traffic excluded
python scripts/measure_stream_bands.py debug/perception/data_video1_tracks.jsonl \
    --stations 11 --x-start 300 --x-end 3000 --window 4.0 --min-travel 150

# 3. before/after opposing share
python scripts/audit_lane_opposing_traffic.py debug/perception/data_video1_tracks.jsonl

# 4. the remaining candidates, for manual judgement
python scripts/list_wrong_way_candidates.py debug/perception/data_video1_tracks.jsonl --min-hits 3
python scripts/inspect_opposing_window.py debug/perception/data_video1_tracks.jsonl --ids 308 18

# 5. the rule still fires on a real violation in the new geometry
python scripts/validate_wrong_way_detection.py

# 6. visual check over real frames
python scripts/preview_directional_lanes.py data/data_video1.mp4 --times 5 40 120 200 290
```
