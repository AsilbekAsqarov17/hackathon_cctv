# Part A operations

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-part-a.txt  # optional detector
```

## Check runtime

```bash
python scripts/check_runtime.py
```

## Smoke test

```bash
python scripts/make_synthetic_video.py debug/synthetic.mp4
python -m unittest discover -v
python scripts/benchmark_part_a.py debug/synthetic.mp4
```

## Use a scene

Copy `configs/scenes/example_fixed_camera.json` to
`configs/scenes/<video-stem>.json`, adjust the polygons/ROIs, and run:

```bash
python scripts/benchmark_part_a.py path/to/video.mp4
```

If world coordinates are available, add four or more image/world point pairs
and compute the homography:

```bash
python scripts/compute_homography.py configs/scenes/<video-stem>.json \
  --image-points '[[0,0],[100,0],[100,100],[0,100]]' \
  --world-points '[[0,0],[1,0],[1,1],[0,1]]'
```

## Official-style Part A run

```bash
python run_submission.py --videos path/to/videos --out predictions.json --team team --no-risk
python evaluate.py --pred predictions.json --validate-only
```

Once development labels exist, run:

```bash
python evaluate.py --pred predictions.json --gt labels.json --per-video
```
