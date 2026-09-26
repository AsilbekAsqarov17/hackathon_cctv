# Dev-set labels: what is in them, and what is deliberately not

`my_labels.json` holds three labelled events across 10.6 minutes of the two
organizer sample clips. This note records how they were produced, what was
excluded and why, because a dev set is only useful if you know what it does not
contain.

## How they were made

Two independent reviews of each clip, from timestamped contact sheets at 1–1.5 s
cadence (`scripts/make_review_sheets.py`), with 0.25 s confirmation of
boundaries. The full reasoning of each reviewer, including the candidates they
rejected, is in:

- `annot_C3897.json` — review 1 of C3897
- `annot_C3897_review2.json` — review 2 of C3897, independent
- `annot_C3902.json` — review of C3902

Only events that survived both the evidence and a tie-break are labelled.

## What is labelled

| Clip | Event | Span | Confidence |
|---|---|---|---|
| C3897 | `stopped_vehicle` | 0.0 – 25.5 s | medium-high |
| C3897 | *(everything else)* | — | certified absent |
| C3902 | *(everything)* | — | certified absent |

The `stopped_vehicle` is a dark SUV standing alone in the open junction area.
Both reviewers verified it independently at pixel level — one by frame-differencing
its bounding box (under 1.0 mean absolute difference at every sample from t=0 to
24.8, then 16.7 at 25.2), the other by template-matching its roof panel (score
0.99, same pixel throughout). It has a driver, no damage, no debris, and no
vehicle queued in front of or behind it.

## What was excluded, and why

**`jaywalking` C3897 0.0–8.4 s** — disputed. Review 1 called it high confidence:
a man in a dark maroon T-shirt crossing 0.5–1.5 m below the zebra, reaching the
pink island at about 7 s. Review 2 built a pedestrian-density map over 18,511
detections and ran eight targeted high-zoom checks, finding all pedestrian flow
on the two marked crossings or on pavements. Re-checked directly at 0.25 s
cadence: the figure is real and moving, but a pedestrian is roughly 20 px tall in
this view, and "0.5 m off the crossing" is finer than the image can settle.
Excluded rather than guessed — a false label is worse than a missing one.

**`congestion` C3897 0.0–21.0 s** — the judgement call most likely to be wrong in
the other direction. It matches the wording literally (10+ vehicles across all
lanes, at a standstill), but review 2 measured the signal going red at t=0 and the
queue forming on red and clearing on green, five times per clip. The
`stopped_vehicle` definition explicitly carves out signal queues, and a queue
that forms on red and clears on green is normal operation. A labeller reading the
wording literally would add roughly five segments.

**`stopped_vehicle` C3902 96.0–118.5 s** — an orange delivery scooter, stationary
in a marked lane for 22.5 s, more than double the threshold. Rejected because it
departed within a second of the adjacent queue discharging and the signal turned
green at t=117, i.e. it was waiting at a stop line. It was, however, entirely
alone in its lane with nothing ahead of it — so if the intended reading is that
"in a queue" excludes only *multi-vehicle* queues, this is a valid event.

**`failure_to_yield` C3897 67.5–69.3 s and C3902 173.5–186 s** — a car does
traverse a zebra while pedestrians are on it in both cases. In C3897 the car is
following *behind* the pedestrians and the overlap falls inside the timing error.
In C3902 no pedestrian could be shown to be standing on the carriageway portion of
the crossing at the moment the car entered.

## Adjudicating two reviews that disagreed

The two reviews of C3902 reached opposite conclusions: one certified **no
events at all**, the other reported four. Both used pixel-level checks, so this
was settled by measurement rather than by argument. The lime-green bus was
tracked by colour at 1 s steps (`annot_C3902_review1.json` said it was creeping
throughout; `annot_C3902.json` called it `stopped_vehicle` at 66.5–80 s with
high confidence):

| t (s) | centroid x | Δx per second |
|---|---|---|
| 54–58 | 872 → 1633 | +198, +209, +207, +145 — driving across the frame |
| 59–64 | 1633 → 1746 | +50, +24, +6, +1, +16, +17 — decelerating to a stop |
| **65–68** | 1746 → 1749 | **+1.8, +0.2, −0.6, +2.0 — stationary** |
| 69–72 | 1762 → 1813 | +13, +19, +18, +13 — pulling away |

So the bus genuinely did stop, but for about **four seconds (t ≈ 64.5–68.5)**,
not the 13.5 s the second review reported. That is well under the 10 s
`stopped_vehicle` threshold, so the label does not hold. The first review's
*conclusion* was right and its *reasoning* ("creeping throughout") was wrong.

The colour tracker switches to a second green bus after t = 73, so positions
beyond that point are not the same object and are not used here.

**Still unresolved:** the second review's `jaywalking` at 82.3–95 s (three people
crossing open carriageway, template-tracked) was not re-checked, and the first
review explicitly looked for and denied jaywalking in this clip. Rather than
label an event I could not adjudicate, C3902 is kept as a negative sample. If a
human confirms those three people were on the carriageway and off the markings,
C3902 is no longer a clean negative and the false-positive count measured
against it needs revisiting.

## Limitations you should hold me to
- **Three events is far too few to tune thresholds on.** Fitting against this
  would be overfitting to two clips. Thresholds are therefore derived from the
  task definitions and physical reasoning, and this set is used for the one thing
  it can do: show that the false-positive rate has collapsed.
- **These are not an independent test set.** They are the clips the thresholds
  were checked against, so scores against them are optimistic.
- **C3902 is a negative sample and is the most valuable clip here.** It is
  certified to contain no event of any class — four repetitions of one ordinary
  signal cycle — so every event predicted on it is a false positive. Under
  macro-F1 that is exactly the quantity worth measuring.
- **The `near_miss` absence is weak evidence.** At this camera height, braking and
  swerving genuinely cannot be resolved, so a near miss could have occurred
  unobserved by both reviews.
- **`red_light` and `stop_line` are unresolved, not clean negatives.** The
  transverse stop lines were never located, and which of the two visible signal
  heads governs which approach could not be settled, so a red-runner on other
  movements would have been missed entirely.
- **`illegal_turn` and `solid_line_crossing` are unresolved.** The lane markings
  are too worn to judge and the camera angle does not resolve a turn.
