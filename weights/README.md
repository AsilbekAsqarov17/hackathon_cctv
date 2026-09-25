# Weights

Place the selected Part A detector checkpoint here before an offline run.
For example, with the default configuration:

```text
weights/yolo11n.pt
```

The current smoke-test checkpoint was obtained from the Ultralytics assets
release for `yolo11n.pt` and is not automatically downloaded by the runtime.
SHA-256: `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`.
Set `TRAFFIC_YOLO_MODEL` or edit `configs/default.json` to use another local
checkpoint. Review the model license before distributing it.
