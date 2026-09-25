# Part B labeling and calibration

Part B uses the official accident/near-miss event intervals. Keep the labels in
the same video-level JSON format used by the evaluator:

```json
{
  "video_001.mp4": {
    "duration": 60.0,
    "fps": 25.0,
    "events": [[20.0, 26.0, "accident"]]
  }
}
```

For tuning, create a video-level validation split and measure:

- chance-normalized AP;
- alarm precision/recall/F1 at score 0.5;
- mean time-to-alarm.

Tune the causal scorer only on validation data. Do not train or calibrate on
future accident intervals or use Part A's final event list as a Part B input.
