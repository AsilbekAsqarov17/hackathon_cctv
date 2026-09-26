# `static/data/` — generated, do not hand-edit

Both JSON files in this directory are **build output**. They are produced by the
scripts in `website/tools/`, and the page renders them verbatim. Editing them by
hand is pointless: the next regeneration overwrites the change, and the git
commit stamp in each file exists precisely so a hand edit is detectable.

| file | written by | what it holds |
|---|---|---|
| `eda.json` | `website/tools/analyze_samples.py` | container facts, per-frame object counts, the 1-second density series, two spatial heatmaps, per-class statistics, cross-clip comparison |
| `results.json` | `website/tools/run_samples.py` | predicted event segments, the per-frame risk curve, alarms extracted with the metric's own function, runtimes, format validation, annotated-frame manifests |

Regenerate from the **repository root**:

```bash
python website/tools/analyze_samples.py     # ~9 min for both clips on CPU
python website/tools/run_samples.py         # ~15 min for both clips on CPU
```

`run_samples.py` flushes `results.json` after every clip and accepts `--resume`
to skip clips already present, so an interrupted run never loses finished work.

## Provenance fields

Each file carries the git commit, whether the working tree was dirty, and a UTC
timestamp. `results.json` additionally carries, **per clip**:

* `code_revision` -- the commit that clip was analysed at;
* `code_dirty` -- whether the tree was dirty at that moment;
* `code_fingerprint` -- a digest of the *actual content* of `src/`, `configs/`,
  `solution.py` and `evaluate.py`, so two runs at the same commit with different
  uncommitted work are still told apart;
* `config_sha` -- a short SHA-256 of `configs/default.json`.

If two clips were analysed from different code states, the file says so in a
top-level `warning` and sets `single_fingerprint` / `single_revision` /
`single_config` to `false`. The page surfaces that as an amber banner.

`--metadata-only` re-derives the scene block, the detector status and the metric
score without re-running inference. It never rewrites the four fields above,
because they describe the events, not the metadata. If the tree has moved on it
records the present state in `tree_at_refresh` and sets `metadata_stale: true`.

## `scored`

When `data/annotations/my_labels.json` exists, `run_samples.py` runs
`evaluate.evaluate()` from the repository root -- unmodified -- and embeds the
result. The labels are the team's own, and their own caveats are carried through
verbatim.

One of the two clips is annotated as containing **no reportable event of any
class**, so it is a negative sample. `Score_A` on it is therefore a
false-positive measurement, not a detection score, and a low value is the
informative outcome rather than a bug. Part B is `null` for the same reason: the
official metric returns `M = Score_A` when a test set contains no accidents.

## Reading them without the page

```bash
python3 -c "import json; d=json.load(open('eda.json')); \
  print([(v['id'], v['container']['frames_decoded'], \
          v['detection']['objects_per_sampled_frame']['mean']) for v in d['videos']])"

python3 -c "import json; d=json.load(open('results.json')); \
  [print(v['file'], v['code_fingerprint'], len(v['events']), \
        v['risk']['mean']) for v in d['videos']]; \
  print('Score_A:', (d.get('scored') or {}).get('score_a'))"
```
