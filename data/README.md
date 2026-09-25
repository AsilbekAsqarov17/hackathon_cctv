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
