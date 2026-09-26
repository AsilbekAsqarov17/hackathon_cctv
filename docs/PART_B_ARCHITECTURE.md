# Part B architecture

## Goal

At every frame, estimate the probability that an `accident` begins within the
next 5 s (H = 5 s in the metric), using only frames already received.

## Causality

The estimator is self-contained. It runs its own detector, tracker and rule
engine over the frames `step()` receives, opens no file, and buffers no frames.

An earlier version read a compact feature cache that Part A published while
processing the whole clip. Each cached sample was individually causal — it was
computed from state at that timestamp and no other — but the task rules state
that reusing Part A output is a violation, and a reviewer reading `solution.py`
would have had to take our word for the distinction. The ambiguity is not worth
the saved pass. `publish_risk_features` / `get_risk_features` remain as no-ops so
that an older code path cannot quietly reintroduce the dependency, and
`tests/test_part_b.py` asserts both that a published cache is ignored and that
replaying with a different tail leaves earlier scores bit-identical.

## Features

`src/risk/features.py` extracts, per frame, only information available at that
timestamp: vehicle and person counts, close-pair count, minimum TTC, minimum
distance, maximum closing speed, maximum deceleration and acceleration,
pedestrian conflict, and whether the rule engine currently sees an accident,
near-miss or wrong-way candidate.

## Scoring

`src/part_b.py::_score_features` maps features to `[0, 1]`. The dominant term
is a smooth TTC ramp over the horizon: a pair with a finite TTC receives
`ttc_floor + (1 - ttc_floor) * (1 - ttc/horizon)`, floored below the alarm
threshold so that proximity alone does not raise an alarm. Proximity, closing
speed and braking contribute smaller terms, and an active accident candidate
overrides. The result is smoothed against the previous frame to avoid a
one-frame spike becoming an alarm.

This is a **heuristic scorer, not a learned classifier.** Its weights have not
been fitted against labelled accidents, because the sample clips contain no
confirmed collision to fit them on. The alarm threshold is fixed at 0.5 by the
metric, so the only free parameter is how eagerly the ramp reaches it.

## Cost

The rule engine is quadratic in the number of tracks, and it was being
evaluated on every frame while detection ran every fifth — so most of that work
recomputed an unchanged answer. Rules and pairwise features now run on detector
samples only, and in between the previous features are repeated verbatim. The
task permits skipping frames internally and returning the last score; repeating
the exact previous value is the strongest form of that. This took Part B from
126 s to 41 s on a 60-second clip, and the end-to-end run from 94% of the time
budget to roughly half of it, on a machine with no GPU.

`risk.detector_stride` in `configs/default.json` controls the Part B sampling
rate independently of Part A's `detector.stride`.
