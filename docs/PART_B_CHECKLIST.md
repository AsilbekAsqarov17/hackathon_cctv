# Part B completion checklist

- [x] Keep the required `RiskEstimator.reset(meta)` and `.step(frame, t)` API.
- [x] Add a canonical causal track-feature contract.
- [x] Add TTC, distance, closing-speed, braking, and pedestrian features.
- [x] Add an initial five-second causal risk scorer.
- [x] Prevent future cached features from being read by earlier timestamps.
- [x] Add a fallback causal detector/tracker runtime when no cache exists.
- [x] Share compact features with the Part A pass to avoid duplicate inference.
- [x] Add Part B unit tests and a harness smoke test.
- [x] Add a risk-curve benchmark script.
- [ ] Build a labeled accident/near-miss development set.
- [ ] Tune score ranking and the 0.5 alarm threshold.
- [ ] Compare heuristic risk with DSTA/DoTA-derived temporal features.
- [ ] Verify AP, alarm F1, and mTTA on held-out videos.
- [ ] Run the final offline Part A + Part B timing test.
