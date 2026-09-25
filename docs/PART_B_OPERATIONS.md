# Part B operations

## Run through the official harness

```bash
pip install -r requirements-part-a.txt
python run_submission.py --videos path/to/videos --out predictions.json --team team
```

`RiskEstimator.step` is called for every frame. The Part A pass publishes only
compact causal features; the Part B reader never uses a feature from a later
timestamp.

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

1. TTC scale and near-distance threshold;
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
