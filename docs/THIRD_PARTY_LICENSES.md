# Third-party provenance and licence record

Final state of every external component, for the record. A repository licence
does not automatically grant dataset or model-weight rights, so those are listed
separately.

## Runtime dependencies

| Component | Version | Purpose | Licence |
|---|---|---|---|
| numpy | >=1.24 | arrays throughout | BSD-3-Clause |
| opencv-python-headless | >=4.8 | frame decode, ROI colour classification, geometry helpers | Apache-2.0 |
| ultralytics | >=8.4,<9 | detector inference and the ByteTrack tracker | **AGPL-3.0** — see the note below |
| scipy, lap, cython_bbox | see `requirements-part-a.txt` | ByteTrack's assignment solver | BSD-3-Clause / MIT |
| onnx, onnxruntime-gpu | see `requirements-part-a.txt` | optional lighter inference path | Apache-2.0 / MIT |

### The AGPL note

`ultralytics` is AGPL-3.0. It is included because the competition runs offline
and the detector has to work without network access. If redistribution outside
the competition context is a concern, note that:

* the **fine-tuned checkpoint alone** (`weights/traffic_part_a_gpu_best.pt`) is
  sufficient for Part A;
* the traffic-light sub-detector, the only other use, can be disabled by
  setting `detector.traffic_light` to `null` in the configuration, at the cost of
  the dynamic-light fallback (calibrated ROIs are the primary signal source
  regardless).

## Model weights

| Checkpoint | Origin | Licence |
|---|---|---|
| `weights/traffic_part_a_gpu_best.pt` | Fine-tuned by this team on imagery this team collected at the competition camera | Trained by us; no third-party weights embedded beyond the initialisation below |
| `weights/yolo11n.pt` | Ultralytics stock release | AGPL-3.0 |
| `weights/yolo26n.pt` | Ultralytics stock release | AGPL-3.0 |
| `weights/yolo11n.onnx` | ONNX export of `yolo11n.pt` by this team | inherits AGPL-3.0 |

## Datasets

| Dataset | Provenance | Status |
|---|---|---|
| `data/*.mp4` | Provided by the organizers for this competition | **Not redistributed** — `.gitignore` excludes `*.mp4`. Used for all calibration, validation and reported results. |
| Detector training set | Frames extracted from the organizer sample video by this team and labelled by the team | **Not redistributed** (≈400 MB of duplicated frames; the videos are not in the repo either). Regenerable via `scripts/train_detector.py`; the split is defined in `configs/detector_data.yaml`. |

## Ideas consulted, no code copied

These were reviewed while designing the rule set. No runtime code was taken from
any of them, and none is vendored.

| Project | Used for | Action taken |
|---|---|---|
| Official ByteTrack | Tracker design | The official implementation is used as a **package dependency** (`ultralytics`), not vendored, so its licence travels with it |
| Smart-Traffic-Analysis-With-Yolo | Lane and trajectory concepts | Concepts reimplemented from the geometry; no code copied |
| SignalWatch | Calibration and rule taxonomy | Concepts only |
| TrafficVision | Integration reference | No code copied |
| traffic-violation-detection | Red-light and wrong-way test ideas | No code copied |
| Vision Patrol | Rule taxonomy reference | No code copied (GPL-3.0, deliberately not vendored) |
| DoTA / DSTA | Accident anticipation research | Not part of the runtime; see `README.md` §6 for the causal scoring actually used |

## Development annotations

`data/dev_annotations.json` was produced by this team by inspecting the sample
videos. It is **partial by construction** and is not ground truth; see the
`_README` field in that file and `README.md` §11.

## Event evidence

`docs/evidence/*.jpg` are frames from the organizer sample video, included to
support specific claims in the report. Each is paired with the command in
`docs/evidence/README.md` that regenerates it.
