# Website status

**Status: not built.** This file records what remains, so the gap is explicit
rather than implied by an empty directory.

## Why

The task requires a project website covering team, problem, approach, pipeline
diagram, EDA, sample-video visualizations, event timelines, risk curves,
example event evidence, failure cases, a live upload demo, the report, and
repository/weights links.

## What blocks it

1. **No hosting or domain.** No account, credentials or DNS access for this
   project is available in the working environment. Publishing requires either
   GitHub Pages (needs the repository to be public, or Pages enabled) or an
   external host.
2. **The site must use the final results.** The event timelines, risk curves
   and failure cases it would show come from `predictions_samples.json` and the
   evidence frames in `docs/evidence/`. Building the site before those are final
   would mean publishing numbers that then change.

## What is ready to publish

Everything a site would display already exists as files in this repository:

| Site section | Source |
|---|---|
| Approach, architecture, pipeline diagram | `README.md` §2–§4 |
| EDA (lane fits, occupancy percentiles) | `camera.md` |
| Sample-video visualizations | `docs/evidence/*.jpg` |
| Event timelines | `predictions_samples.json` |
| Risk curves | the `risk` array in `predictions_samples.json` (`--save-risk`) |
| Example event evidence | `docs/evidence/README.md` (each frame has its regeneration command) |
| Failure cases | `README.md` §12 and `camera.md` "Known ambiguities" |
| Report, repository, weights | `README.md`, this repository, `weights/README.md` |

## To finish it

Once hosting is available, the site is a static build over those files. Nothing
in the detection pipeline needs to change, and no metric needs to be invented:
`README.md` §11 and §12 already separate verified results from development
numbers, and every published number should come from the two commands in §1.

**No metric on the site may be fabricated.** `data/dev_annotations.json` is
partial by construction, so figures derived from it are development numbers and
must be labelled as such.
