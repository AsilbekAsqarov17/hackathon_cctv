# Weights

All checkpoints are committed so the submission runs with **no network access**.
Nothing is downloaded at run time: `allow_download` is `false` in every config,
and `src/perception/detector.py` never fetches a model implicitly.

The competition limit is 5 GB of weights; the total here is under 25 MB.

## Files

| File | Size | Used for |
|---|---|---|
| `traffic_part_a_gpu_best.pt` | 5.2 MB | **The Part A detector.** Fine-tuned by this team on traffic imagery from the competition camera. 8 classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign. Referenced by `configs/default.json`. |
| `yolo11n.pt` | 5.4 MB | Stock Ultralytics YOLO11n. Used only for the traffic-light sub-detector in the hybrid backend, and as the CPU fallback. |
| `yolo26n.pt` | 5.3 MB | Stock Ultralytics YOLO26n. Alternative fallback. |
| `yolo11n.onnx` | 10.2 MB | ONNX export of `yolo11n.pt` for the lighter ONNX Runtime path. |

## Integrity

Regenerate the table below with:

```bash
Get-ChildItem weights -Filter *.pt | ForEach-Object {
  "{0,-30} {1,8:N1} MB  {2}" -f $_.Name, ($_.Length/1MB), (Get-FileHash $_.FullName -Algorithm SHA256).Hash
}
```

| File | SHA-256 |
|---|---|
| `traffic_part_a_gpu_best.pt` | `D76725D6EB2055618683F71E6B26098600456B0E0FC789817FBE6E28E345D710` |
| `yolo11n.pt` | `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1` |
| `yolo11n.onnx` | see `git log` for the hash at the release commit |

## Selecting a different checkpoint

Without editing code:

```powershell
$env:TRAFFIC_YOLO_MODEL = "weights/your_checkpoint.pt"
```

Or set `detector.primary.model` in the configuration. Note that
`configs/default.json` is **generated** from
`configs/data_video1_rules_dev.json` by `scripts/build_default_config.py`; edit
the development config, then regenerate.

## Licensing

* `traffic_part_a_gpu_best.pt` — trained by this team on imagery this team
  collected. The competition camera is provided by the organizers for this task.
* `yolo11n.pt`, `yolo26n.pt`, `yolo11n.onnx` — Ultralytics, **AGPL-3.0**. Review
  the AGPL implications before redistributing outside the competition context.
  They are included here because the runtime must work offline; if that is a
  problem, the fine-tuned checkpoint alone is sufficient for Part A and the
  traffic-light sub-detector can be disabled by setting
  `detector.traffic_light` to `null`.

## Retraining

```bash
python scripts/train_detector.py --data configs/detector_data.yaml \
    --weights weights/yolo11n.pt --epochs 100 --seed 42 --deterministic
```

The run writes to `runs/detect/` (gitignored). Copy the resulting `best.pt` to
`weights/` and update the hash table above.
