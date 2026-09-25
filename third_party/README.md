# Third-party snapshots

This directory contains selected upstream source references used by Part A.

- `byte_track/`: official ByteTrack tracker core, MIT license.
- `smart_traffic/`: Smart-Traffic-Analysis-With-Yolo reference snapshot, MIT license.

The competition runtime does not import the reference application directly. It
uses adapters and its own canonical `TrackManager`; this keeps the submission
compatible with the fixed harness and makes inherited code boundaries explicit.
