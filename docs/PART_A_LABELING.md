# Part A labeling workflow

Label development videos before tuning thresholds. Keep the split at video
level.

1. Watch each full video and mark the exact start/end of each official event.
2. Record affected track IDs when available.
3. Include negative clips and empty-event videos.
4. For detector fine-tuning, create YOLO-format object boxes separately.
5. For scene configuration, copy the per-video template and draw lane,
   stop-line, crossing, road, and traffic-light regions on a representative
   frame.
6. Run `scripts/validate_labels.py` and then `evaluate.py --per-video`.

Do not use Part B accident anticipation labels to create Part A event
segments; keep the two annotation tasks separate.
