# Traffic event detection and accident anticipation from a fixed road camera

Detects 14 classes of traffic event as time segments from a fixed CCTV view of a
Tashkent intersection, and produces a causal per-frame estimate of the risk that
a collision is about to begin.

```text
video ─► detector ─► tracker ─► trajectories ─┬─► geometry rules ─► segments   (Part A)
                                              └─► risk features ─► risk curve  (Part B)
```

Both parts read the same two files and nothing else: `solution.py` and
`configs/`. The organizers' harness is used unchanged.

## Running it

```bash
pip install -r requirements.txt
python run_submission.py --videos samples --out predictions.json --team <team>
python evaluate.py --pred predictions.json --gt my_labels.json --per-video
```

That is the whole setup. There is no manual step, no weight download, and no
network access at inference time. The detector checkpoint
(`weights/yolo11n.onnx`, 10.7 MB) is committed to the repository, and ONNX
Runtime falls back from CUDA to CPU on its own if the evaluation host cannot
load a GPU provider.

To check the install without a video:

```bash
python scripts/check_runtime.py
```

## What it actually does

| Stage | Component | Notes |
|---|---|---|
| Decode | `src/video.py` | sequential read with a configurable stride |
| Detect | `src/perception/onnx_detector.py` | YOLO11n via ONNX Runtime; COCO classes, road users and signals only |
| Track | `src/perception/tracker.py` | ByteTrack-style two-stage IoU association, no compiled dependencies |
| Track state | `src/tracking/track_manager.py` | the single source of truth: box, centre, smoothed velocity and acceleration, lane history, ground-contact history |
| Geometry | `src/scene/` | road polygons, lanes, stop and solid lines, crossings, signal boxes, optional homography |
| Rules | `src/rules/engine.py` | 14 independent rules returning `(active, confidence, evidence)` |
| Segments | `src/temporal/segmenter.py` | merges samples, applies per-class min/max duration, enforces no same-class overlap |

### Decisions worth knowing about

**Thresholds are physical, not pixel.** A fixed pixel distance means something
different at 3840×2160 than at 1920×1080, and the sample clips are 4K. A third
of the original thresholds were unreachable at 4K scale — `stopped_vehicle` at
3 px/s and `congestion` at 8 px/s could not fire, because a car at 30 km/h moves
hundreds of pixels per second. `TrackState` derives metres-per-pixel from the
apparent box height, so distances and speeds are expressed in metres and m/s and
survive a change of resolution.

**`accident` is a contact test, not a speed threshold.** Two bodies that overlap
do so at any speed. The previous form also had the wrong sign: it required
closing speed at the moment of impact, which is precisely when both vehicles are
decelerating hardest, so it tended to fire just before a collision and then stop.

**Events are capped in duration.** A rule that stays satisfied for a whole clip
is a rule that is too loose, not a real event. Reporting a 60-second "accident"
earns no temporal IoU against a 5-second ground truth, so each class has a
maximum length and the onset is preserved.

**Part B is independent of Part A.** The rules state that reusing Part A output
is a violation. `RiskEstimator` runs its own detector, tracker and rules on the
frames it is handed; it opens no file, buffers no frames, and reads nothing
Part A produced. The cost is a second detection pass, which the time budget
absorbs. Between detector samples the previous features are repeated verbatim,
which is what keeps that pass cheap.

**The camera is calibrated, and shipped as the default.** Test videos are named
`test_001.mp4` and so never match a per-video scene file; they always resolve to
`configs/scenes/tashkent_intersection.json`. An earlier version shipped an empty
default scene, which silently disabled six of the fourteen classes.

## Configuration

`configs/default.json` is the runtime configuration; environment variables
(`TRAFFIC_SCENE_CONFIG`, `TRAFFIC_ACTIVE_CLASSES`, `TRAFFIC_STRIDE`, …) override
individual keys. Per-video configs are supported via `TRAFFIC_CONFIG`.

The scene file is separate because it is the one thing that must match the
camera. To re-derive it:

```bash
python scripts/derive_scene_geometry.py data/samples_small --frames 150   # measure where vehicles are
python scripts/preview_scene_fill.py VIDEO configs/scenes/tashkent_intersection.json --frame 4500 -o out.jpg
```

The second command is the important one: an outline-only overlay hides whether a
polygon covers the pavement, so the areas are filled at partial opacity.

## Development

The organizer sample clips are 3840×2160, 10-bit 4:2:2, 140 Mbit/s — 5.4 GB
each. `scripts/shrink_samples.py` makes 1080p working copies that are ~33×
smaller with the frame count and timing preserved exactly:

```bash
python scripts/shrink_samples.py C3897.MP4 C3902.MP4 -o data/samples_small/ --height 1080 -j 2
```

It uses PyAV rather than the `ffmpeg` command line, because PyAV bundles a
complete FFmpeg including libx264 while many system builds ship without it.

```bash
python scripts/make_review_sheets.py data/samples_small/C3897_small.mp4 --step 2   # annotate
python -m unittest discover -v                                                       # 20 tests
python scripts/benchmark_part_a.py data/samples_small                                # timing
```

### The dev set

The organizer samples ship without labels, so the team annotated them by
reviewing timestamped contact sheets (`scripts/make_review_sheets.py`). Labels
live in `data/annotations/` in the ground-truth format `evaluate.py` expects,
and are what every tuning decision in this repository was checked against.

They are *our* labels and are approximate at the boundaries: a 2-second
sampling grid cannot resolve a one-second contact to better than about ±1 s.
They are also not an independent test set — they were made from the same two
clips the thresholds were tuned on, so scores against them are optimistic and
should be read as a development signal rather than an estimate of the hidden-set
score.

## Data, weights and licences

| Component | Source | Licence |
|---|---|---|
| `weights/yolo11n.onnx` | Ultralytics YOLO11n, exported to ONNX | AGPL-3.0 (Ultralytics) — see note |
| COCO class names | Microsoft COCO | CC BY 4.0 annotations |
| `third_party/byte_track/` | FoundationVision/ByteTrack | MIT |
| Sample clips `C3897.MP4`, `C3902.MP4` | provided by the organizers | used for development only, not committed |

**No training was performed.** The detector is the stock COCO checkpoint. The
team's pseudo-labelled 199-frame set under `data/traffic_coco/` is unused by the
runtime; it exists from an abandoned fine-tuning attempt and the labels are
machine-generated and unreviewed. The rules, not a learned model, carry the
event detection.

No external dataset was used for training. No hosted or paid model is called at
any stage, for inference or for writing this code.

Ultralytics ships AGPL-3.0, which is a strong copyleft. The exported ONNX graph
is a build artefact of the pretrained checkpoint; if the organizers prefer a
permissive licence, substituting an AGPL-free detector of the same interface
(`src/perception/detector.py`) is a config change, not a code change. This is
flagged rather than hidden and should be reviewed before publication.

## Known limitations

These are real and were measured, not guessed.

- **The road geometry is approximate.** The intersection is large and open, and
  the carriageway was hand-calibrated against a coordinate grid on single
  frames. Occupancies of tracked vehicles were measured to guide it, but
  occupancy alone cannot separate carriageway from surrounding paving, so the
  polygons are drawn rather than fitted. Kerbs, service roads and lane
  boundaries away from the two main corridors are not modelled.
- **The signal geometry is thin.** One signal head was positively identified and
  configured. `red_light` and `stop_line` depend on it and on stop lines that
  were not drawn, so those classes are effectively off rather than unreliable.
- **Small and occluded objects are under-detected.** Stock COCO weights at
  640 px on a 4K source miss distant pedestrians and riders; measured mean
  confidence for `traffic light` is 0.25, at the detector's floor. Counts from
  the EDA are a lower bound.
- **The tracker is the simple one.** The vendored ByteTrack needs `lap`,
  `cython_bbox` and `scipy`, none of which ship manylinux wheels, so it is not
  the default. On a busy intersection that costs identity stability.
- **`near_miss` and `accident` rest on braking and contact heuristics** that have
  not been validated against real collisions, because the sample clips contain
  no confirmed collision.
- **No claim is made about generalization beyond this camera.** The scene
  geometry is specific to one viewpoint by construction.

## Layout

```text
solution.py              the only file the organizers import
configs/                 runtime config + per-camera scene geometry
src/                     perception, tracking, scene, rules, temporal, risk
scripts/                 tooling: shrink, review sheets, calibration, benchmarks
tests/                   20 unit and pipeline tests
data/                    annotations, measurement artefacts, EDA inputs
website/                 public team site, report and live demo
third_party/             vendored ByteTrack core (MIT)
```
