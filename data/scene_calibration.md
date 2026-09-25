# data_video1 scene calibration

Source video: `data/data_video1.mp4`

- Resolution: 3840×2160
- FPS: 29.97002997
- Duration: 340.34 s
- Frames: 10,200
- Camera: fixed elevated CCTV

## Current calibration files

- `configs/scenes/data_video1_base.json` — conservative road polygons and
  crosswalk regions used for the first real-data run.
- `configs/scenes/data_video1.json` — draft lane/stop-line/signal geometry;
  review it against tracked trajectories before enabling those rules.
- `scripts/preview_scene.py` — renders the geometry over a selected frame.

## Calibration decisions

The road is represented as two diagonal corridors crossing at the center:

- northwest–southeast corridor;
- southwest–northeast corridor.

The first run intentionally uses no stop lines or traffic-light ROIs because
small signal heads and lane boundaries require trajectory review. This avoids
generating red-light/solid-line false positives from an unverified draft.

## Next calibration pass

1. Run the detector/tracker on the full video and save track JSONL.
2. Overlay track centers and trajectories on representative frames.
3. Adjust lane polygons to contain stable track centers.
4. Add stop lines only after confirming the approach direction.
5. Add traffic-light ROIs from zoomed signal crops and verify HSV states over
   time.
6. Re-run preview and compare against several frames, not only frame 5000.
