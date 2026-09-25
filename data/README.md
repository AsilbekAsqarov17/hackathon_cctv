# Data layout

Create a video-level split before fine-tuning:

```text
data/
├── traffic_coco/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── annotations/
│   └── video_events.json
└── scenes/
```

Object detector labels are separate from temporal event annotations. Keep
negative videos and empty-event videos in the validation split so false
positives can be measured.

For the supplied development video:

```bash
python scripts/prepare_detection_dataset.py data/data_video1.mp4 --num-frames 200
python scripts/pseudo_label_frames.py --device 0
python scripts/split_detection_dataset.py
```

The generated labels are pseudo-labels and must be reviewed before being used
for a reported result. A temporal split from one video is for development only;
use multiple videos for a meaningful final validation split.

After review, start a small development fine-tune with:

```bash
python scripts/train_detector.py --model weights/yolo11n.pt --data data/traffic_coco/dataset.yaml --epochs 20 --imgsz 640 --batch 4 --device 0 --workers 0
```
