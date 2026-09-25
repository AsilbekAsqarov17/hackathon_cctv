# Part B architecture

## Goal

Estimate, causally:

```text
P(accident starts within the next 5 seconds)
```

for every frame, without reading future frames or final Part A intervals.

## Data flow

```text
current/past tracks
        ↓
RiskFeatures
        ↓
causal risk score
        ↓
[time_sec, score]
```

The normal harness path is:

```text
Part A pass
  → compact causal RiskFeatures cache
  → RiskEstimator reads only samples with timestamp <= current t
```

This avoids running the detector twice while preserving causality. If the
cache is unavailable, `CausalRiskEstimator` creates a fallback detector/tracker
runtime that consumes only the frame passed to `step`.

## Features

- minimum pairwise TTC;
- minimum track distance;
- closing speed;
- deceleration and acceleration;
- pedestrian/vehicle conflict;
- current accident/near-miss/wrong-way rule candidates;
- stopped and closing-pair counts.

The initial scorer is a heuristic, not a future-label classifier. Its
weights and thresholds must be calibrated on a labeled validation split. A
learned temporal model can later replace `_score_features` while keeping the
same feature contract.

## Files

- `src/part_b.py` — cache, causal runtime, scorer, harness estimator.
- `src/risk/features.py` — feature extraction contract.
- `configs/risk.example.json` — risk thresholds.
- `tests/test_part_b.py` — causality and feature tests.
