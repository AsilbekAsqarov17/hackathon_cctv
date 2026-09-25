# Third-party provenance and license record

This file must be updated before distributing the final submission.

| Component | Intended use | Current action | License status |
|---|---|---|---|
| Official ByteTrack | Canonical tracker | Optional adapter; dependency-free fallback included | Verify upstream version/license before vendoring |
| Smart-Traffic-Analysis-With-Yolo | Lane/trajectory ideas | Reimplement or adapt only after source review | User supplied MIT status; preserve notice if copied |
| SignalWatch | Calibration/rule design | Reimplement concepts; no code copied | License clarification required |
| TrafficVision | Integration reference | No runtime code copied | No license confirmed |
| traffic-violation-detection | Red-light/wrong-way test ideas | No runtime code copied | No license confirmed |
| Vision Patrol | Rule taxonomy reference | No runtime code copied | GPL-3.0; do not include without deliberate compliance |
| Ultralytics/YOLO | Optional detector | `ultralytics 8.4.163`; `yolo11n.pt` local checkpoint used during smoke testing | Review AGPL/enterprise terms separately |
| DoTA/DSTA | Later research/training | Not part of Part A runtime | Code and dataset rights are separate |

Record the exact commit and model checkpoint in the final README. A repository
license does not automatically grant dataset or model-weight rights.
