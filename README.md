# WIUT Hackathon 2026 — Computer Vision track

Traffic event detection and accident anticipation from a single fixed road camera.

The organizers' harness (`run_submission.py`, `evaluate.py`) is **unmodified**.
Everything in `src/`, `configs/`, `scripts/` and `tests/` is this team's work.

```text
solution.py        <- the only file the harness imports (CLASSES, detect_events, RiskEstimator)
run_submission.py  <- organizers' harness                      (unmodified)
evaluate.py        <- format check + official metric          (unmodified)
predictions_samples.json <- our output on the provided sample videos
```

---

## 1. Quickstart

```bash
pip install -r requirements.txt          # harness only: numpy + opencv
pip install -r requirements-part-a.txt   # detector, tracker, OpenCV extras

# The checkpoint is committed under weights/; nothing is downloaded at run time.
python run_submission.py --videos data --out predictions.json --team wiut-cv
python evaluate.py --pred predictions.json --validate-only
```

Development and validation:

```bash
python -m unittest discover -v          # 69 tests
python scripts/run_samples.py --videos data/data_video1.mp4 data/data_video2.mp4 \
    --out predictions_samples.json       # end-to-end, with the time budget checked
python scripts/validate_risk_causality.py --video data/data_video1.mp4 --frames 240
python scripts/evaluate_dev.py --pred predictions_samples.json --gt data/dev_annotations.json
```

---

## 2. Architecture

```text
                      ┌──────────────────────── Part A (offline) ────────────────────────┐
video ──▶ detector ──▶ ByteTrack ──▶ TrackManager ──▶ RuleEngine ──▶ TemporalSegmenter ──▶ events
           (YOLO)      (tracker)      (trajectories)   (14 rules)     (segments)
                              │              │              │
                              └──────────────┴──────────────┘
                                             │
                        compact causal feature snapshots (timestamps only)
                                             │
                      └──────────────────────── Part B (streaming) ──────────────────────┐
risk score ◀── CausalRiskEstimator ◀── monotonic pointer over those snapshots
```

Key design decisions, each of which is a response to something measured on the
sample videos rather than a preference:

**The detector runs once, not twice.** Part A and Part B share the 3× time
budget, and Part B is called for every frame. `PartAPipeline` publishes compact
per-frame feature snapshots; `CausalRiskEstimator` walks them with a *monotonic*
pointer, so a snapshot later than the current frame can never be read even
though Part A saw the whole video. The cache is keyed by the `video_id` the
harness passes to `reset`, matching the key `publish_risk_features` writes.
Without that match the estimator would silently run a second detector over
every frame and blow the budget.

**One authored scene serves every resolution.** `data_video1.mp4` is
3840×2160 and `data_video2.mp4` is 1920×1080 — the same intersection from the
same viewpoint. `configs/scenes/data_video1.json` declares
`reference_size: [3840, 2160]`; `load_scene_config` maps every coordinate onto
the actual frame size. Without this the scene would be 2× wrong on the
half-resolution video, silently disabling every geometry rule.

**Rules consume track state, never pixels.** `RuleEngine` takes a canonical
`FrameState` and the calibrated scene. That keeps every rule unit-testable
without a video or a GPU — which is how the 55 tests cover all 14 classes plus
their negative cases.

**Rules return candidates, not flags.** Several objects can violate one rule at
once. Every rule exposes `_<label>_candidates()` returning one signal per
distinct object, and `TemporalSegmenter` groups by object identity, so two
jaywalkers produce two events rather than one merged segment.

---

## 3. Detector and tracker

| Component | Choice | Why |
|---|---|---|
| Detector | Ultralytics YOLO, checkpoint `weights/traffic_part_a_gpu_best.pt`, `imgsz=640`, `conf=0.20` | Fine-tuned on this camera's traffic; 8 classes (person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign) |
| Detector stride | every 3rd frame | The rules are trajectory-based; 3× is the cheapest stride that keeps heading and TTC estimates smooth (see §7) |
| Tracker | ByteTrack (`high=0.5`, `low=0.15`, `match=0.7`, `max_age=30`) | The official starter tracker; handles occlusion at the median, which matters here |

The fine-tuned checkpoint was trained by this team on traffic imagery from this
camera. `scripts/train_detector.py` reproduces the run and
`configs/detector_data.yaml` holds the dataset configuration.

**Track IDs are deterministic between runs** (ByteTrack is deterministic given
the same detections), which is what makes the offline replay in
`scripts/replay_rules.py` a valid proxy for a full run: the same video yields
the same track IDs, so rule changes can be validated in seconds instead of
quarter-hours.

### Weights

Committed under `weights/`, so the submission runs fully offline:

| File | Size | SHA-256 (first 16) | Licence |
|---|---|---|---|
| `traffic_part_a_gpu_best.pt` | 5.2 MB | see `weights/README.md` | trained by this team |
| `yolo11n.pt` | 5.4 MB | `0EBBC80D4A7680D1` | AGPL-3.0 (Ultralytics) — stock, used only for the traffic-light sub-detector and as a CPU fallback |
| `yolo26n.pt` | 5.3 MB | see `weights/README.md` | AGPL-3.0 (Ultralytics) — stock fallback |

`weights/README.md` records the full hashes. No weights are downloaded at run
time: `allow_download` is `false` everywhere.

**Detector outputs are not used for signal state.** The base checkpoint's
"traffic light" detections on this video are all false positives (a blue
pedestrian-crossing sign, a pedestrian behind a railing, a sign post). Signal
state is read exclusively from calibrated ROIs by
`src/scene/signals.py`.

---

## 4. Scene calibration

`configs/scenes/data_video1.json` is the single most important artefact in the
repository. Almost every rule is meaningless without it, and several are
dangerous without it. It is documented in full in [`camera.md`](camera.md) and
summarised here.

| Element | Value | How it was established |
|---|---|---|
| Lanes 0,1 | `W_outer`, `W_inner` — westbound, `(-1.0, -0.30)` | Fitted band edges from per-x-station occupancy percentiles (`scripts/measure_stream_bands.py`) |
| Lanes 2,3 | `E_inner`, `E_outer` — eastbound, `(1.0, 0.47)` | Same, then verified against the measured streams |
| Stop line | `b_eastbound_stop` | Placed where the queue's lead vehicles park — see below |
| Signal ROIs | `signal_eastbound`, `signal_left` | Readability proven with `scripts/probe_signal_roi.py` (saturated-pixels-per-frame) |
| Crossings, road, kerbs | 4 crossings, 6 solid markings | Measured and visually confirmed |

Two calibration findings worth stating, because both were bugs before they were
findings:

**The stop line is placed from the queue, and the crossing it serves is derived
from the signal.** A signal head is mounted several metres up, so its ROI
projects well above the road point it governs and appears *upstream* in the
image; an "is the head downstream of the line" test therefore cannot work. Heads
are matched to lines by **declared approach direction** instead
(`SceneContext.signal_for_line`). Given the governing head, the crossing that
traffic queues for is the one nearest that head — here `crossing[1]`, 161 px away,
not `crossing[0]`, 207 px away. The line is then swept upstream from that
crossing and scored against the queue: at an 80 px setback the vehicles holding
the queue park at −19 px and −17 px from the line, and every other setback is
worse. An earlier calibration anchored to `crossing[0]` and placed the line
*inside the junction*, where ordinary through traffic crossed it.

Evidence: [`docs/evidence/stop_line_placed_at_queue.jpg`](docs/evidence/).

**The signal is matched to the line by direction, and yellow is never red.**
The competition definition of `red_light` is a crossing *on red*. `STOPPING_COLORS`
is `frozenset({"red"})`; yellow is excluded, and an unreadable head yields
`unknown`, which leaves the rule silent rather than assuming a state.

---

## 5. Rules

All 14 official classes are implemented and enabled. Thresholds live in
`configs/data_video1_rules_dev.json`, each with a `_note` recording whether it
was measured and by which script.

| Class | Evidence used | Key exclusions |
|---|---|---|
| `red_light` | Front bumper crosses the stop line while the governing head is red | Yellow is not red; unreadable signal is silent; direction-aware crossing, so leaving/reversing cannot fire |
| `stop_line` | Front past the line, stationary, red, not inside a crossing | Upstream of the line is a queue; green ends the event (the official end condition) |
| `stopped_vehicle` | Stationary ≥ 10 s on the carriageway | Signal queues excluded, by governing signal *and* by a stationary vehicle ahead in the same lane |
| `wrong_way` | Sustained travel against the calibrated lane direction | Stationary, jittering, and regions deliberately left without a direction (the east leg) |
| `congestion` | Standstill/crawling across **every** lane of one travel direction | A standing queue at a red light; a single stopped vehicle |
| `jaywalking` | Pedestrian on a *lane* polygon, outside every crossing, moving | Pavements, medians, islands, standing still |
| `failure_to_yield` | Pedestrian in/on a crossing, vehicle traversing it while moving | A vehicle that halts is compliant; proximity alone is never sufficient |
| `solid_line_crossing` | Front changes **side** of a calibrated marking | Travelling *along* the marking; clipping its end; bbox jitter |
| `illegal_turn` | Lane's declared `allowed_moves` vs the actual movement | A lane change while still following the lane; a turn through the junction, where no lane applies |
| `illegal_u_turn` | Sustained heading reversal that ends against the flow | A curve through the junction (net displacement separates them) |
| `accident` | Road-frame contact **plus a transition from motion to rest** | Standing queues (never moving), pass-throughs (never stop); **vehicle+vehicle only** |
| `near_miss` | Finite TTC + close + sharp deceleration or sustained swerve | Queue following distance; two vehicles is required, not a pedestrian |
| `road_obstacle` | Persistent non-road-user on the carriageway | See limitation 5.2 — the competition checkpoint emits no such class |
| `fire_smoke` | Persistent fire/smoke class | See limitation 5.2 |

### The finding that shaped the interaction rules

`scripts/measure_interaction_metrics.py` measures 659 210 real track pairs on
`data_video2`. The result that mattered:

```text
gap_px:  p50=540  p90=1391  p99=1738
iou:     p50=0.00 p90=0.00  p99=0.16   max=0.97
decel:   p50=2.6  p90=142   p99=650
```

Box IoU reaches 0.97 in *ordinary queued traffic*. The camera looks obliquely
down the carriageway, so the lead vehicle's box covers the one behind it. **Image
overlap is not contact evidence on this camera.** A queue is separated *along*
the road by roughly a car length and *not* separated across it; a collision is
separated in neither.

Every interaction rule therefore measures separation in the road frame
(`kinematics.road_frame_gap`). Applying that single change took the candidate
count on `data_video2` from 3175 accident frames to 256, and near-miss from 1848
to 63. Evidence:
[`docs/evidence/queue_boxes_overlap.jpg`](docs/evidence/).

### The accident gate is a transition, not a threshold

Road-frame contact alone still produced **28 accident segments on a video with
no visible collision**, because at this camera vehicles genuinely pass within a
few pixels of each other. Neither obvious tightening was sufficient on its own,
and each had to be found by looking at what it wrongly admitted:

* *"both vehicles are stationary"* admits **every pair in a standing queue** —
  they are in contact and have been at rest from the start.
* *"they decelerated or swerved"* admits a **fast pass-through**, where the
  tracker carries the boxes through each other, so overlap and heading change
  both look like contact while neither vehicle ever stops.

What actually distinguishes a collision is the **transition from motion to
rest**, which is also what the official end condition describes: *"all involved
objects stop moving or leave the frame."* The rule therefore requires the pair to
be at rest now **and** to have been moving over the preceding 4 s. Only one of
the pair needs to have been moving, because in a rear-end collision only the
striking vehicle stops abruptly.

That took `accident` from 28 segments to 7 on `data_video2`, and both negative
cases are regression tests (`test_no_accident_for_two_cars_in_a_queue`,
`test_no_accident_when_the_pair_keeps_moving`). The remaining 7 are genuine
candidates, not confirmed collisions, and `data/dev_annotations.json` records no
verified `accident` instance in either sample video.

The same lesson was applied a second time in Part B (§6).

### Signal queue suppression

Suppressing the stopped vehicles that queue at a red light is what keeps
`stopped_vehicle` meaningful. A vehicle counts as queued when it is upstream of
the calibrated line **and** either the governing signal is red, or a stationary
vehicle sits between it and the line in the same lane. The second test matters
because the tail of a long queue is over a thousand pixels back, where a simple
"near the line" radius would fail.

---

## 6. Part B — causal accident anticipation

`RiskEstimator.step(frame, t_sec) -> float ∈ [0,1]`, horizon 5 s, called for
every frame.

Causality is **tested, not asserted**
([`scripts/validate_risk_causality.py`](scripts/validate_risk_causality.py)):

1. **Prefix invariance** — streaming the first N frames and stopping gives
   scores identical to 1e-12 with the rest of the video present. Any peek at a
   later frame would change an earlier score and fail this.
2. **Order sensitivity** — reversing the frame order changes the output, so the
   score is not a constant.
3. **Range** — all scores finite and in [0,1] on real video and on black, white,
   saturated and noise frames.

`step` never opens the video file, and the internal state contains only
observations at or before the current timestamp.

**Calibration was the hard part, and it took three measured corrections.** The
initial score used the minimum TTC over *all* pairs. Measured on `data_video2`,
that put **177 of 180 frames above the official 0.5 alarm threshold** — a
constant alarm, which scores zero on the chance-normalised AP and produces no
usable alarms. Attribution
([`scripts/measure_risk_terms.py`](scripts/measure_risk_terms.py)) showed the
TTC ramp alone was responsible (p50 = 0.915), because thirty vehicles in a queue
always contain *some* pair with a short time-to-collision. The minimum was a
measure of traffic density, not of risk.

1. **Restrict to genuinely conflicting pairs.** The ramp is built on
   `min_conflict_ttc` — pairs that share a calibrated lane (or involve a
   pedestrian), are closing, and are closer than a car length in the road
   frame. The closing threshold is measured, not guessed: over 600 frames the
   relative speed of same-lane road-frame-close pairs runs p50 = 57, p90 = 127,
   p99 = 186 px/s, so the gate sits at 140 px/s (≈ p95).
2. **Make every pixel threshold resolution-independent.** The thresholds were
   in native frame pixels, so the same physical separation was twice as
   permissive at 4K. This is what made the fix look correct on `data_video2`
   (1080p) while the score was *still* saturating on `data_video1` (4K) at 97.5 %
   of frames. All risk thresholds are now expressed in the 3840×2160 reference
   resolution and scaled to the frame, exactly as the scene geometry is.
3. **Gate the ramp on evasive behaviour, and shorten it.** Even restricted to
   conflicts, 9–19 pairs pass the test at any moment on the 4K video with a
   median TTC of 1.17 s — that is ordinary car-following, because the lead
   vehicle is going to brake. The ramp now requires a conflicting pair to also
   be braking hard or swerving (the criteria the validated `near_miss` rule
   already uses), and its ramp length is 2 s rather than the full 5 s horizon,
   because the score is a confidence that a collision is *imminent*, not a
   countdown.

Frames at or above the alarm threshold on the busiest sample fell
**97.5 % → 80 % → 16 % → 9 %**, each step taken only after measuring which term
was saturating.

**The high end of the score is not calibrated, and that is a stated limitation.**
There is no confirmed accident in either sample video, so there is no positive
example to anchor the score's magnitude against. Every threshold above was
chosen to keep ordinary traffic low and the score in range, which is what a
chance-normalised AP rewards; how high the score actually goes near a real
collision is unvalidated.

---

## 7. Performance

Measured with `scripts/run_samples.py`, which drives the real interface and
enforces the real budget.

| Video | Duration | Part A | Part B | Total | Budget (3×) |
|---|---|---|---|---|---|
| `data_video1.mp4` (4K) | 340.3 s | 632.8 s (1.86×) | see `predictions_samples.json` log | — | 1021 s |
| `data_video2.mp4` (1080p) | 317.8 s | — | — | — | 953 s |

The single biggest cost lever is the detector stride. Detection runs every third
frame; the rules are trajectory-based and 3× keeps heading, TTC and stationarity
estimates smooth while cutting inference to a third. The official
`PartAPipeline` honours it; the development runner
`scripts/run_rules.py` does not, and additionally pays for debug video encoding
— do not use it for timing.

No per-frame full-image YOLO pass exists in the submission path. The signal ROI
reader classifies a small calibrated patch, and the traffic-light sub-detector
is the only additional inference and it runs at the same stride.

---

## 8. Reproducibility

* **Determinism.** ByteTrack is deterministic given the same detections, so
  track IDs are stable across runs on the same video. The offline replay in
  `scripts/replay_rules.py` is therefore a valid proxy for a full run — this is
  the property that makes rule iteration practical, and it is asserted by
  comparing event files with `scripts/compare_events.py`.
* **No hidden state.** Rules carry their timers keyed by
  `(label, track_id, …)` and are torn down when a key stops being reported, so a
  tracker's re-IDs cannot silently merge two events.
* **No network.** `allow_download: false`; all weights are committed.
* **Seeds.** No training happens in the evaluation path, so no seeds are set at
  inference. The detector training script fixes `--seed 42` and
  `--deterministic`; see §9 for the exact command.
* **One source of truth for configuration.** `configs/default.json` is
  *generated* from `configs/data_video1_rules_dev.json` by
  `scripts/build_default_config.py`, so the tuned configuration and the
  configuration that actually runs in the submission cannot drift. The harness
  calls `detect_events` with no config argument, so this file *is* the
  submission.
* **Causality regression.** `trailing_window` had a real bug — it bounded the
  window below but never above, so any caller passing history extending past
  `now` got a whole-track "window". Production callers were unaffected (history
  is appended per frame), but ten analysis scripts were. It is fixed, has a
  regression test, and all ten scripts now import the production function so
  tooling cannot drift from the engine.

---

## 9. Datasets and licences

| Dataset | Use | Licence / provenance |
|---|---|---|
| Organizer sample videos (`data/*.mp4`) | All calibration, validation and reported results | Provided by the organizers for this competition; not redistributed by us (`.gitignore` excludes `*.mp4`) |
| Traffic imagery collected by this team at this camera | Detector fine-tuning (`runs/detect/traffic_part_a_gpu/`) | Collected by the team; see `configs/detector_data.yaml` for the exact split |
| YOLO11 / YOLO26 pretrained weights | Initialisation for fine-tuning; traffic-light sub-detector | AGPL-3.0, Ultralytics |

Retraining the detector:

```bash
python scripts/train_detector.py --data configs/detector_data.yaml \
    --weights weights/yolo11n.pt --epochs 100 --seed 42 --deterministic
```

The checkpoint that ships is `weights/traffic_part_a_gpu_best.pt`; its hash is
recorded in `weights/README.md`.

---

## 10. Tests

```bash
python -m unittest discover -v      # 69 tests
```

| File | Covers |
|---|---|
| `tests/test_rules.py` | `stopped_vehicle`, `wrong_way` (incl. the allowlist control pair), `red_light` (front-bumper, yellow, unreadable, leaving), `trailing_window` causality |
| `tests/test_rules_extended.py` | The other 11 classes, each with its **negative** case: a legal turn, a pedestrian in a crossing, a standing queue, a yielding vehicle, a car tracking along a solid line, two cars in a queue. Plus the rule dispatch contract |
| `tests/test_part_b.py` | Causal cache reader, score range, alarm behaviour |
| `tests/test_scene.py` | Resolution mapping, stop-line span and direction, signal-to-line matching by approach direction, carriageway vs footway |

Tests exercise real logic on constructed trajectories. Where a test would have
passed only because of a bug in production code, the production code was fixed
rather than the test weakened.

Two of these tests exist because they caught real defects rather than to raise
coverage:

* `test_trailing_window_ignores_samples_after_now` — `trailing_window` bounded
  the window below but never above, so any caller passing history past `now` got
  a whole-track "window". Production was unaffected, but ten analysis scripts
  were. Fixed, with a regression test.
* `test_every_rule_has_a_dispatchable_candidate_producer` — `collect_signals`
  finds producers by name via `getattr`, so a wrong signature does not raise; it
  is caught per frame and the rule silently reports nothing. That had already
  happened to the two turn rules. This test found the same latent problem in
  `accident`, `near_miss`, `road_obstacle` and `fire_smoke`, where multiple
  simultaneous instances were collapsing into one segment.

Geometry invariants are also available as standalone scripts, so they can be
run against a scene file without the test suite:
`scripts/check_line_span.py` (the stop line spans and cuts the lanes it
governs), `scripts/check_stop_line_setup.py` (governing signal found, colour
round-trips, approach points upstream, identical at 4K and 1080p),
`scripts/check_scene_resolution.py` (one scene resolves to the same lanes at
both resolutions).

---

## 11. Sample results

`predictions_samples.json` is the output of
`python scripts/run_samples.py --videos data/data_video1.mp4 data/data_video2.mp4`,
produced by the real `solution.py` interface.

Because the organizers ship no ground truth for these videos, this repository
carries its own partial annotations in `data/dev_annotations.json`, and
`scripts/evaluate_dev.py` reports TP/FP/FN, per-class F1 at tIoU
{0.3, 0.5, 0.7}, and start/end boundary error.

**These are development numbers, not a competition score.** The annotations are
partial by construction: only intervals a human confirmed in the video are
listed, so an empty class means "no verified instance found", not "absent from
the video", and the false-positive count is an upper bound. See the `_README`
field in that file.

Verified findings on the samples:

* `red_light` fires on `data_video2` at 94.09–94.80 s and 244.14–244.94 s. The
  first was confirmed frame by frame: track #203's front bumper had just crossed
  the stop line (`front_along = +11.1 px`) with the governing head red, while
  the vehicles queued behind it were stationary
  ([`docs/evidence/red_light_track203.jpg`](docs/evidence/)). Both intervals
  fall inside independently measured red phases.
* `wrong_way` produces **0** events on both sample videos, and is proven
  functional by `scripts/validate_wrong_way_detection.py`, which injects
  synthetic violations into all four calibrated lanes and confirms the rule
  fires in 2.20–2.30 s in each while staying silent for compliant traffic.
* No collision is visible in either sample video, so `accident` is empty and the
  official Part B contribution reduces to zero for this data (`M = Score_A`).

Evidence frames with the command that regenerates each: `docs/evidence/`.

---

## 12. Limitations and honest findings

1. **The east leg has no usable signal.** The signal heads there face away from
   this camera. Probing found 0 saturated red pixels per frame on the upper head
   and a permanently yellow source at (760, 440) that is not a signal, against
   8.7 red pixels *per frame* for the calibrated eastbound head. That region is
   therefore deliberately left without a stop line, and the seven `stopped_vehicle`
   events there are **reported rather than suppressed** — suppressing them would
   require inventing a signal state. This is limitation 2 in `camera.md`.

2. **Vehicle–pedestrian contact is not reported as `accident`.** A pedestrian
   near a car at this camera is a yield conflict, which `failure_to_yield`
   judges from the crossing geometry. Charging the same moment to two classes
   would be penalised twice by the official metric, so `accident` requires two
   vehicles and `failure_to_yield` owns the pedestrian case.

3. **Collision detection is fundamentally limited by the viewing angle.** With a
   single oblique camera and no depth, "two boxes overlap" and "two cars are a
   car length apart in a queue" are not reliably separable in image space. The
   road-frame test plus the motion-to-rest transition is the best available
   evidence, and it is conservative by construction: `accident` will
   under-detect glancing rear-end contacts at long range rather than over-detect
   queues. The 7 residual candidates on `data_video2` are **candidates, not
   confirmed collisions** — no collision is visible in either sample video.

4. **`illegal_turn` prohibition evidence is partial.** A U-turn is reported only
   when the manoeuvre ends against the flow of the lane it started in, and
   prohibition is claimed from the layout — a kerbed median makes a U-turn across
   this carriageway prohibited by the road itself. There is no "No U-turn" sign
   in view, so sign evidence is **not** claimed, and the rule is correspondingly
   strict.

5. **`road_obstacle` and `fire_smoke` cannot fire with the shipped detector.** The
   competition checkpoint is trained on eight road-user classes and emits
   nothing that maps to an obstacle, fire or smoke; COCO has no such classes
   either. The rule logic is implemented and unit-tested, and it stays silent
   rather than guessing. A colour or texture heuristic was considered and
   rejected: it fires on brake lights, sunlight and compression artefacts, and a
   false positive in a class with no ground truth costs three zeros in the
   official per-class mean.

6. **Jaywalking depends on the lane polygons, not the road polygons.** The road
   polygons at this camera include the verges and footways around the junction;
   using them put 53 % of all pedestrian samples "on the road". The rule
   therefore requires a *lane* polygon. The consequence is that a pedestrian
   crossing between lanes, where no lane polygon exists, is not detected.

7. **The Part B score's high end is uncalibrated.** There is no confirmed
   accident in either sample video, so no positive example exists to anchor the
   score's magnitude near a real collision. The thresholds are tuned to keep
   ordinary traffic low — which is what the chance-normalised AP rewards — and
   the estimator is provably causal, but "how high does the score go when a
   collision is about to happen" is untested. This is the main outstanding Part
   B limitation and it is a data limitation, not a design one.

8. **No cross-camera generalisation is claimed.** All geometry is calibrated for
   this single viewpoint, which the task explicitly permits. The resolution
   mapping is general (one scene, any frame size from the same camera), but a
   genuinely different camera would need its own scene file.

---

## 13. Repository layout

```text
solution.py                     the interface the harness imports
configs/
  default.json                  GENERATED from the dev config — this is the submission
  data_video1_rules_dev.json    the tuned configuration, every threshold annotated
  scenes/data_video1.json       calibrated geometry (authored at 3840x2160)
src/
  part_a.py                     Part A pipeline
  part_b.py                     Part B causal estimator + scoring
  perception/                   detector, tracker, traffic-light reader
  scene/                        scene config, geometry, signal reading
  rules/                        engine.py (14 rules), kinematics.py, motion.py
  temporal/segmenter.py         signals -> non-overlapping segments
  risk/features.py              causal Part B features
scripts/                        calibration, analysis, validation and dev tools
tests/                          69 tests
docs/evidence/                  frames the report cites, + how to regenerate
docs/THIRD_PARTY_LICENSES.md    provenance and licence record for every dependency
camera.md                       authoritative scene description
data/dev_annotations.json       this project's partial development annotations
```

`configs/default.json` carries a `_banner` explaining that it is generated; run
`python scripts/build_default_config.py` after changing any threshold.

---

## 14. Environment variables

| Variable | Purpose |
|---|---|
| `TRAFFIC_CONFIG` | alternate Part A/Part B configuration file |
| `TRAFFIC_YOLO_MODEL` | local checkpoint path |
| `TRAFFIC_DETECTOR_BACKEND` | `auto`, `yolo`, `motion`, `null` |
| `TRAFFIC_TRACKER_BACKEND` | `simple_bytetrack`, `official` |
| `TRAFFIC_SCENE_CONFIG` | scene JSON path |
| `TRAFFIC_ACTIVE_CLASSES` | comma-separated rule labels to enable |
| `TRAFFIC_DEVICE` | e.g. `0` to force GPU |

## 15. Team

Team name and member details are maintained in
`docs/TEAM.md`; update it before submission.

## 16. Website

Status and any external hosting limitation are recorded in `docs/WEBSITE.md`.
