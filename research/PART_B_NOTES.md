# Part B research notes

The initial runtime scorer is intentionally lightweight and causal. DSTA and
DoTA remain research candidates for a later learned temporal head:

- DSTA: accident anticipation model/reference; old environment and dashcam
  domain, so it is not the runtime foundation.
- DoTA: temporal anomaly annotations, object tracks, and optical-flow features;
  useful for training/evaluation, but fixed-CCTV domain adaptation is required.

The next learned experiment should consume the existing `RiskFeatures`
contract, not replace the causal feature boundary. Compare it against the
heuristic scorer on a video-level validation split using AP, alarm F1, and
mTTA.
