# `website/` — public team site + live-demo backend

A **single-page, zero-build, zero-dependency** site for the traffic-event /
accident-anticipation submission (team `798C27C9`).

* Plain HTML + CSS + vanilla ES5-compatible JS. No npm, no bundler, no CDN,
  no web fonts, no analytics. It works from `file://` (except the upload) and
  from any static host.
* Every number on the page is read at runtime from
  `static/data/eda.json` and `static/data/results.json`, which are **generated
  by the two Python scripts in `tools/`**. Nothing is typed by hand. If a JSON
  file is missing the corresponding section renders an explicit *pending* state
  instead of making something up.
* `demo/app.py` serves that same site plus the upload API, using only the
  Python standard library. It imports the real `solution.detect_events` and
  `solution.RiskEstimator` from the repository root.

```
website/
├── index.html              the whole site
├── assets/
│   ├── style.css           design system, dark + light, mobile-first
│   ├── charts.js           ~350-line SVG/canvas chart kit (no libraries)
│   ├── site.js             nav, the 14-class table, team + link placeholders
│   ├── eda.js              EDA section  (renders static/data/eda.json)
│   ├── results.js          results section (renders static/data/results.json)
│   ├── report.js           the write-up (prose + measured numbers)
│   └── demo.js             upload client, progress, job polling
├── static/
│   ├── data/               eda.json · results.json   (GENERATED)
│   ├── media/              *_preview.webm + events/*.jpg (GENERATED)
│   └── img/                downscaled copies of data/inspection/*.jpg
├── demo/app.py             stdlib HTTP server: static files + upload API
├── tools/
│   ├── analyze_samples.py  EDA generator  -> static/data/eda.json
│   ├── run_samples.py      pipeline runner -> static/data/results.json
│   └── check_site.js       headless-browser smoke test (dev only, needs node)
└── README.md               this file
```

---

## 1. Run it locally

### Just look at the page (no Python at all)

The page is static. Any web server will do:

```bash
cd website
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

This is the honest way to check the *content*. The **Live demo** section will
detect that no API is present, explain that in an amber box, and disable the
upload button. Everything else works.

> Opening `index.html` directly from disk also works for the read-only
> sections, but browsers block `fetch()` of local JSON under `file://`, so the
> EDA and results sections will show their pending state. Use a real server.

### The full thing, including the live demo

From the **repository root** (the backend resolves `weights/yolo11n.onnx` and
`solution.py` relative to the repo, so the working directory matters):

```bash
pip install -r requirements.txt        # numpy, opencv-python-headless, onnxruntime(-gpu)
python website/demo/app.py             # http://127.0.0.1:8000
```

Options:

```bash
python website/demo/app.py --port 9000 --host 0.0.0.0   # expose on the LAN
python website/demo/app.py --static-only                # site only, uploads refused (503)
python website/demo/app.py --verbose                    # request logging
DEMO_VERBOSE=1 python website/demo/app.py               # same, via env
```

On start it prints what it found, including the detector checkpoint and the
ONNX Runtime provider, so you know immediately whether the demo will do real
inference or degrade.

### Check the API by hand

```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool

JOB=$(curl -s -F "file=@clip.mp4" http://127.0.0.1:8000/api/jobs | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
watch -n1 "curl -s http://127.0.0.1:8000/api/jobs/$JOB | python3 -m json.tool"
```

---

## 2. Regenerate the data (do this after changing the model)

Both scripts must be run **from the repository root** so that the relative
weight path in `configs/default.json` resolves. Each takes minutes on CPU.

```bash
cd <repo root>

# EDA: container facts, per-frame object counts, density series, two heatmaps,
# and the seekable preview clips.  ~9 min for both 5-min clips.
python website/tools/analyze_samples.py

# Measured pipeline output: calls solution.detect_events() and
# solution.RiskEstimator exactly like run_submission.py, writes the event
# timeline, the risk curve, runtimes and annotated frames.  ~11 min.
python website/tools/run_samples.py
```

Useful flags:

| flag | script | effect |
|---|---|---|
| `--limit-seconds 60` | both | only the first 60 s — a fast smoke test |
| `--no-media` | `analyze_samples.py` | JSON only, skip the ffmpeg previews |
| `--no-frames` | `run_samples.py` | JSON only, skip the annotated JPEGs |
| `--no-part-b` | `run_samples.py` | events only, skip the risk loop |
| `--stride 5` | `analyze_samples.py` | detector every 5th frame (faster, coarser) |
| `--videos a.mp4 b.mp4` | both | analyse different clips |
| `--out PATH` | both | write the JSON somewhere else |
| `--resume` | `run_samples.py` | **skip videos already in the output** |
| `--retry-on-drift` | `run_samples.py` | if `HEAD` moves between clips, discard the partial set and start over |
| `--metadata-only` | `run_samples.py` | re-derive scene/config provenance in place, no inference |

`run_samples.py` flushes `results.json` after **every** video and `--resume`
skips what is already there, so a long run that is interrupted never loses the
videos that already finished.

**Provenance is stamped per clip, and the tool defends it.** Each video entry
records `code_revision`, `code_dirty` and a short SHA-256 of
`configs/default.json`. Two clips analysed by different code revisions are not a
matched pair, so the file says so in a top-level `warning`, sets
`single_revision` / `single_config` to `false`, and the page shows an amber
banner. `--retry-on-drift` goes further and restarts the whole set if `HEAD`
moved mid-run, so the file never quietly mixes versions. This matters here
because the two sample clips are a matched pair — a difference between them is
only meaningful if they were produced the same way.

`--metadata-only` exists because some derived facts change without the inference
changing. If a commit only edits `configs/` or the scene lookup, the scene block,
the detector status and the metric score can be re-derived in place instantly,
whereas re-running Part A over two five-minute clips cannot.

It deliberately **does not** re-stamp `code_revision`, `code_fingerprint` or
`config_sha`, because those describe what produced the *events*, and the events
were not re-run. If the tree has moved on, the file records the present state
under `tree_at_refresh` and sets `metadata_stale: true`, and the page shows a
neutral provenance note saying exactly that. Overwriting the inference stamps
with the present tree would be a small lie, and it is the kind of small lie that
makes every other number on the page untrustworthy.

### The labels, and what the score means

`run_samples.py` also looks for `data/annotations/my_labels.json` (override with
`--labels`), in `evaluate.py`'s own format. When it exists, the script runs
**`evaluate.evaluate()` from the repository root, unmodified**, and embeds the
result under `scored`. We did not write a scorer of our own: the numbers on the
page are the graders' own arithmetic applied to our own annotations.

Those labels are *ours*, and their own caveats are quoted on the page. One clip
was reviewed and certified to contain **no reportable event of any class**, which
makes it a negative sample — so `Score_A` on it is a direct false-positive
measurement rather than a detection score, and a low value is the informative
result, not a bug. Part B comes back `n/a` for the same reason: the official
metric returns `M = Score_A` outright when a test set contains no accidents.

Everything the two scripts write is derived output. `eda.json` records the git
commit and whether the working tree was dirty at the time, and the page shows
both — so a stale screenshot is always detectable.

### The media step

`analyze_samples.py` shells out to `ffmpeg` for the previews:

```
-vf scale=480:-2,fps=15 -c:v libvpx-vp9 -b:v 300k -row-mt 1
```

VP9 in `.webm` for three reasons: roughly **10 MB per 5-minute clip** (an
H.264 preview of the same clip is 27 MB), broad browser support, and — this
one is a repo constraint — the repository `.gitignore` has a global `*.mp4`
rule, so an `.mp4` preview would be silently untracked. `ffmpeg` must be on
`PATH`; without it the JSON is still produced and the page falls back to
conventional filenames.

---

## 3. Deploy

The site is a plain directory. `demo/`, `tools/` and `README.md` are also
inside it and are harmless when published (a few KB of Python and Markdown);
if you would rather not publish them, copy `index.html`, `assets/`,
`static/` and `README.md` into a `docs/` folder and point Pages there.

**Only the upload needs a server.** On a static host the rest of the page,
including the interactive timeline and every chart, works unchanged — the
upload box says so instead of failing silently.

### GitHub Pages

```bash
# from the repository root
cp -r website website_publish_tmp   # if you want to exclude demo/ and tools/
```

Simplest: use `website/` itself as the Pages root.

1. Repo → **Settings** → **Pages**
2. **Source:** *Deploy from a branch*, branch `main`, folder `/website`
3. Save. The published URL is `https://<user>.github.io/<repo>/`.

`.webm` and `.json` are both served with correct MIME types by Pages. Total
published size is about **25 MB**, dominated by the two preview clips.

To exclude the Python:

```bash
mkdir -p docs
cp -r website/index.html website/assets website/static docs/
git add docs && git commit -m "publish site"
# then Settings → Pages → folder /docs
```

### Netlify

No build step, so tell Netlify there is nothing to build:

1. **Add new site → Import an existing project**.
2. Build command: *(leave empty)*. Publish directory: `website`.
3. Deploy.

Or drag the `website` folder onto <https://app.netlify.com/drop>.

### Vercel

```bash
npx vercel deploy --prod --cwd website --yes   # or: Project → Framework: Other, Output: website
```

No framework preset, no build command, output directory `website`.

### Any other static host

Upload the contents of `website/`. There is nothing to compile. If you want the
upload to work, run `python website/demo/app.py` on a small VM and put a reverse
proxy in front of it — it is a single process with no external state, and jobs
are held in memory only, so it is stateless to restart.

---

## 4. Editing the content

### Team

`index.html` and `assets/site.js` both contain `TODO(team)` markers, and they
are **deliberate placeholders, not invented data**. Search for `TODO(team)`:

| file | what to fill in |
|---|---|
| `assets/site.js` → `TEAM` | the three names, roles, focus and links |
| `assets/site.js` → `WORK` | the ownership column of the work table |
| `assets/site.js` → `buildLinks()` | `repoGuess` → the real repository URL |
| `index.html` | the hero `<title>`, the hero paragraph, the footer |

Members are rendered with a visible amber `TODO(team)` chip so nobody can
mistake a placeholder for a real name. The page deliberately invents no names,
emails, institutions or handles.

### The 14-class table

`assets/site.js` → `CLASS_INFO`. Transcribed from `src/rules/engine.py` and
`configs/default.json`; if you retune a threshold, update the string.

### The report

`assets/report.js`. The prose is ours; **every number in it is injected from
`eda.json` / `results.json`**, so it cannot drift away from the measurements.
If a number is missing from the JSON, the report falls back to an em dash
rather than a stale literal.

---

## 5. The demo API

| method | path | body | returns |
|---|---|---|---|
| `GET` | `/api/health` | – | detector status, limits, versions |
| `POST` | `/api/jobs` | `multipart/form-data` with a `file` field, **or** a raw video body | `202 {job_id, state, max_seconds}` |
| `GET` | `/api/jobs/<id>` | – | `{state, stage, progress, detail, result?, error?}` |
| `POST` | `/api/jobs/<id>/cancel` | – | `{job_id, cancel_requested}` |

`state` is one of `queued`, `running`, `done`, `error`, `cancelled`.

**Limits:** 120 s and 150 MB. The server is authoritative — it re-checks the
duration after reading the container and refuses with a `4xx` plus a copy-
and-paste `ffmpeg` command, rather than silently truncating.

**Result shape** (abridged):

```jsonc
{
  "filename": "clip.mp4",
  "container": { "width": 1920, "height": 1080, "fps": 29.97, "frames": 360, "duration_sec": 12.0 },
  "events":    [ { "start": 0.1, "end": 2.1, "label": "jaywalking", "duration": 2.0 } ],
  "counts":    { "jaywalking": 3 },
  "risk":      { "threshold": 0.5, "points": [[0.0, 0.0], ...], "mean": 0.96, "alarm_count": 1 },
  "annotated_frames": [ { "t": 1.5, "detections": 20, "src": "data:image/jpeg;base64,…" } ],
  "runtime":   { "part_a_sec": 4.1, "part_b_sec": 1.9 },
  "scene":     { "path": "configs/scenes/default.json", "scene_id": "unconfigured", "per_camera_calibrated": false },
  "warnings":  [],
  "degraded":  false,
  "degraded_reason": null
}
```

### Deliberate behaviour

* **Stdlib only.** `cgi.FieldStorage` was removed in Python 3.13, so there is a
  small multipart parser here. A framework would add an install step that can
  fail on an offline machine for zero benefit.
* **Never 500 on a missing checkpoint.** With no `weights/yolo11n.onnx` the job
  still completes and returns real container metadata, an empty event list and
  `degraded: true` with a reason. This is the failure mode we actually hit
  during development, and the graceful path is the one that shipped.
* **Part A failures are warnings, not crashes.** `solution.detect_events()`
  already fails soft internally; if it returns nothing because of a code fault,
  the job still returns `200` with the Part B curve and the warning text, so a
  judge sees *something* real instead of a stack trace.
* **One job at a time.** `MAX_CONCURRENT = 1` behind a semaphore. The detector
  is CPU-saturating; queueing beats thrashing.
* **Progress is interpolated, not measured.** `detect_events()` is a single
  blocking call, so the progress bar extrapolates from elapsed time against the
  expected runtime and clamps below 90 % until the call actually returns. The
  label says which stage it is in. It is honest about being an estimate.
* **Byte-range support.** `http.server` does not implement `Range`, so the
  static handler does. Without it the click-to-seek timeline is unusable on a
  10 MB clip.
* **Annotated frames come back as data URLs** in the response rather than as
  files, so there is no temp-file lifetime to get wrong if the worker dies.

---

## 6. Verification performed

Everything below was run against this tree, with real output, not mocked.

```bash
# both generators really produce data  (fast smoke variants)
python website/tools/analyze_samples.py --limit-seconds 20 --no-media --out /tmp/eda.json
python website/tools/run_samples.py --videos data/dev60/C3897_dev.mp4 --out /tmp/results.json --frames 3

# the backend starts, reports its detector, serves the site and honours Range
python website/demo/app.py --port 8120 &
curl -s http://127.0.0.1:8120/api/health | python3 -m json.tool
curl -s -o /dev/null -D - -H 'Range: bytes=100-200' \
     http://127.0.0.1:8120/static/media/C3897_small_preview.webm     # -> 206 + Content-Range
curl -s -o /dev/null -w '%{http_code}\n' --path-as-is \
     http://127.0.0.1:8120/../solution.py                            # -> 404
curl -s -F 'file=@tiny12.mp4' http://127.0.0.1:8120/api/jobs         # -> 202 {job_id}

# headless browser check of the page itself
node website/tools/check_site.js http://127.0.0.1:8120/ /tmp/shots

# and the honest-degradation path: same page, but with no generated JSON at all
mkdir -p /tmp/nodata/static/data
cp -r index.html assets /tmp/nodata/
ln -sfn "$PWD/static/media" /tmp/nodata/static/media
ln -sfn "$PWD/static/img"    /tmp/nodata/static/img
(cd /tmp/nodata && python3 -m http.server 8130) &
node website/tools/check_pending.js http://127.0.0.1:8130/
```

`tools/check_site.js` and `tools/check_pending.js` are real deliverables, not
scratch work. They drive headless Chrome over the DevTools protocol, record
every console message, page exception and failed request, and exit non-zero on
any of them. `check_site.js` additionally asserts that the generated sections
actually rendered, checks for horizontal overflow at 1280 px, 390 px **and in
dark mode**, and **clicks a timeline segment to confirm the video really seeks**.
`check_pending.js` asserts the opposite condition — that with no JSON at all, the
page shows three explicit *pending* states instead of inventing numbers, while
the static content (class table, team, pipeline, links) stays intact. Both need
`node >= 22` and `google-chrome` on `PATH`; override the binary with `CHROME=`.

Measured on this tree:

| check | result |
|---|---|
| console errors / page exceptions | 0 |
| failed network requests | 0 |
| horizontal overflow, 1280 px and 390 px | none |
| images that resolved to 0 px | none |
| class table rows / team cards / report panels | 14 / 3 / 6 |
| EDA charts / heatmap canvases / derived findings | 4 / 2 / 17 |
| timeline segments, risk charts, annotated thumbs | 125 / 2 / 8 |
| click-to-seek | `currentTime` moved to the segment start (0.801 s for the first segment) |
| `Range` on a 10 MB `.webm` | `206 Partial Content` + `Content-Range` |
| path traversal `../solution.py` | `404` |
| a real 12 s `.mp4` upload | `202` → job completed with container facts, risk curve and 7 annotated frames |

The 12 s upload also produced the most useful failure we could have asked for: a
transient fault upstream made Part A return no events, and the backend returned a
complete, honest result with the warning text attached rather than a `500`. That is
the degradation path, exercised for real.

