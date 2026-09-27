# GPU detector training run — results

Environment: RTX 3050 6GB Laptop GPU (sm_86, 6144 MiB), driver 610.62.
Windows, Python 3.13.12, Ultralytics 8.4.163.

## What changed in the environment

The machine only had `torch 2.14.0+cpu` / `torchvision 0.29.0+cpu`, so training
ran on CPU. Both were replaced with the CUDA builds of the *same* versions:

    torch 2.14.0+cu126
    torchvision 0.29.0+cu126

`torch.cuda.get_arch_list()` includes `sm_86`, so the GPU is used natively rather
than through PTX JIT. `scripts/setup_cuda.ps1` now pins these versions and
verifies CUDA NMS.

The Windows torch wheel is a single 2.6 GB file. Repeated `pip install` and plain
`curl` attempts were both killed part-way on this machine (low free RAM), so the
wheel was fetched with a resume loop (`D:/Hackathon/dl_wheel.ps1`) and then
installed from disk. The wheel was verified as a readable zip (12245 entries, no
bad members) and byte-count matched the server's `content-length`.

## Dataset

`data/traffic_coco`, 199 frames sampled from `data_video1.mp4` (4K, 29.97 fps),
5189 boxes. `development` == `train` (159) + `val` (40), disjoint — a temporal
split, not an independent test set.

`configs/detector_data.yaml` was added because the example YAML pointed at
`D:/Hackathon/data/traffic_coco` (missing the `wiut_cv_scripts` segment) and
listed a `test` split that does not exist.

## Important caveat: the labels are self-generated

`scripts/pseudo_label_frames.py` produced these labels with `weights/yolo11n.pt`
— the same checkpoint that was fine-tuned. The reported mAP therefore measures
**agreement with the base model's own pseudo-labels**, not accuracy against real
ground truth. Expect the numbers to be optimistic and the val set to be far
easier than the competition test set.

## Training

    python scripts/train_detector.py --model weights/yolo11n.pt ^
      --data configs/detector_data.yaml --epochs 50 --imgsz 640 ^
      --batch 8 --device 0 --workers 4 --project runs/detect --name traffic_part_a_gpu

Completed in 0.086 h on GPU (`CUDA:0 (NVIDIA GeForce RTX 3050 6GB Laptop GPU)`),
peak 1.43 GB VRAM. Weights: `runs/detect/runs/detect/traffic_part_a_gpu/weights/best.pt`
(note the doubled path: Ultralytics nests `runs/detect` again, and the script's
final print does not reflect the real location).

### Validation on the 40 held-out frames (930 instances)

| Class     | Images | Inst | P     | R     | mAP50 | mAP50-95 |
|-----------|--------|------|-------|-------|-------|----------|
| all       | 40     | 930  | 0.811 | 0.585 | 0.764 | 0.626    |
| person    | 38     | 71   | 0.856 | 0.817 | 0.906 | 0.703    |
| car       | 40     | 812  | 0.877 | 0.947 | 0.976 | 0.855    |
| bus       | 22     | 25   | 0.788 | 0.744 | 0.888 | 0.708    |
| truck     | 10     | 12   | 0.536 | 0.417 | 0.423 | 0.348    |
| traffic light | 10 | 10   | 1.000 | 0.000 | 0.624 | 0.518    |

## GPU vs CPU (40 val frames, imgsz 640, `scripts/bench_detector_device.py`)

| Model             | CPU ms/img | GPU ms/img | Speedup |
|-------------------|-----------|-----------|---------|
| base yolo11n.pt   | 49.7      | 14.7      | 3.4x    |
| fine-tuned best.pt| 47.1      | 13.2      | 3.6x    |

## Detection counts on the 40 val frames (conf 0.20)

| Class         | base | fine-tuned | mean conf base -> fine-tuned |
|---------------|------|------------|------------------------------|
| car           | 839  | 957        | 0.585 -> 0.806               |
| person        | 78   | 115        | 0.490 -> 0.515               |
| bus           | 25   | 31         | 0.487 -> 0.426               |
| truck         | 12   | 12         | 0.359 -> 0.636               |
| traffic light | 10   | **0**      | 0.228 -> none                |

## Regression to fix before submitting

The fine-tuned model **lost traffic-light detection entirely**. It predicts none
at any threshold, while the base model still emits 10 at conf 0.20, 28 at 0.10,
42 at 0.05 and 207 at 0.01. Val shows P=1.0 with R=0.0, so the boxes rank first
but never clear threshold.

Cause: only 55 traffic-light pseudo-label boxes exist across all 199 frames, and
at `imgsz 640` a 4K traffic light is only a few pixels. The class collapsed to
"never predict".

This matters because `red_light` is one of the 14 scored event classes and the
scene configs use traffic-light ROIs. Before submitting, either add real
traffic-light labels, or keep the base checkpoint for that class.

## Runtime budget check

`run_submission.py` with `configs/data_video1_dev.json` (stride 3), GPU:

| Clip | Duration | Budget (3x) | Used   | Ratio |
|------|----------|-------------|--------|-------|
| 6 s  | 6.0 s    | 18 s        | 17.5 s | 2.92x |
| 30 s | 30.0 s   | 90 s        | 43.5 s | 1.45x |

Fitted cost is roughly 11 s fixed startup + 0.108 s per sampled frame, so the
6 s clip is dominated by startup. Extrapolating to the full 340.3 s
`data_video1.mp4` (10200 frames, ~3400 samples) gives ~379 s against a 1021 s
budget (~0.37x). The budget is not the limiting factor.
