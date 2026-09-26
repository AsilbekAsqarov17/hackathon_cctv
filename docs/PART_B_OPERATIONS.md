# Part B operations

## Run through the official harness

```bash
pip install -r requirements.txt
python run_submission.py --videos path/to/videos --out predictions.json --team team
```

`RiskEstimator.step` is called for every frame. Part B is independent of Part A:
it runs its own detector, tracker and rules on the frames it receives, opens no
file, and reads nothing Part A produced. Between detector samples it repeats the
previous features verbatim, which the task permits.

## Timing

The budget is 3x the video duration for Part A and Part B together. Measure
before trusting that:

```bash
python scripts/benchmark_part_b.py data/samples_small
```

If a run approaches the budget, raise `risk.detector_stride` before touching
anything else — it is the cheapest lever and Part B's features are smooth in
time.

## Inspect a run

```bash
python evaluate.py --pred predictions.json --gt labels.json --per-video
```

The report includes:

- chance-normalized AP;
- alarm precision/recall/F1;
- mean time-to-alarm;
- Part B score.

## Tune

Start with `configs/risk.example.json` and copy the `risk` section into the
active Part A configuration. Tune in this order:

1. TTC floor/progress and near-distance threshold;
2. closing-speed/deceleration scales;
3. accident/near-miss boosts;
4. score decay and alarm threshold.

Do not tune on the official test videos. Keep a video-level validation split.
See `docs/PART_B_LABELING.md` for the annotation format.

## Causality test

```bash
python -m unittest tests.test_part_b -v
```

The cache test changes future samples and verifies that the score at an earlier
timestamp does not change.
