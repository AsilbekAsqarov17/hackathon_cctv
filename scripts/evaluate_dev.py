"""Evaluate a predictions file against the project's development annotations.

The hackathon ships no ground truth for the sample videos, so this repository
carries its own annotations in ``data/dev_annotations.json``. They were produced
by inspecting the sample videos frame by frame with the tooling in ``scripts/``
(``crop_region.py``, ``overlay_geometry.py``, ``inspect_rule_frame.py``) and
recording the interval in which each event is actually visible.

What is reported mirrors the official metric so the numbers are comparable:

* per class, greedy one-to-one matching by temporal IoU at tau in
  {0.3, 0.5, 0.7}, giving TP/FP/FN;
* ``Score_A = mean over classes present in GT or predictions of mean over tau
  of F1``;
* boundary quality, because the official metric is purely temporal and a rule
  that fires at roughly the right moment scores the same as one that pins the
  instant exactly. Both the start and end errors are reported.

Example::

    python scripts/evaluate_dev.py --pred predictions_samples.json ^
        --gt data/dev_annotations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import OFFICIAL_CLASSES, TIOU_THRESHOLDS, tiou  # noqa: E402


def greedy_match(gt: list[tuple[float, float]], pred: list[tuple[float, float]],
                 threshold: float) -> tuple[int, int, int]:
    """One-to-one matching by descending temporal IoU, as the official metric does."""
    pairs: list[tuple[float, int, int]] = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            value = tiou(g, p)
            if value >= threshold:
                pairs.append((value, i, j))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    tp = 0
    for _value, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"p": round(precision, 4), "r": round(recall, 4), "f1": round(f1, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="predictions_samples.json")
    parser.add_argument("--gt", default="data/dev_annotations.json")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    pred_videos = pred.get("videos", pred)
    gt_videos = gt.get("videos", gt)

    per_class: dict[str, dict[str, dict[str, float]]] = {}
    boundary: list[tuple[str, str, float, float, float, float]] = []
    unmatched: list[tuple[str, str, str, float, float]] = []
    extra: list[tuple[str, str, float, float]] = []

    names = sorted(set(pred_videos) | set(gt_videos))
    for name in names:
        p_events = pred_videos.get(name, {}).get("events", [])
        g_events = gt_videos.get(name, {}).get("events", [])
        for label in OFFICIAL_CLASSES:
            p_segs = [(float(e[0]), float(e[1])) for e in p_events if e[2] == label]
            g_segs = [(float(e[0]), float(e[1])) for e in g_events if e[2] == label]
            if not p_segs and not g_segs:
                continue
            bucket = per_class.setdefault(
                label, {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in TIOU_THRESHOLDS}
            )
            # Boundary error is reported at the loosest threshold so a slightly
            # mistimed but correctly detected event is not hidden.
            tp, fp, fn = greedy_match(g_segs, p_segs, 0.3)
            bucket["0.3"]["tp"] += tp
            bucket["0.3"]["fp"] += fp
            bucket["0.3"]["fn"] += fn
            for tau in TIOU_THRESHOLDS:
                if tau == 0.3:
                    continue
                t, p, f = greedy_match(g_segs, p_segs, tau)
                bucket[str(tau)]["tp"] += t
                bucket[str(tau)]["fp"] += p
                bucket[str(tau)]["fn"] += f
            matched_p: set[int] = set()
            pairs = sorted(
                ((tiou(g, p), gi, pi) for gi, g in enumerate(g_segs)
                 for pi, p in enumerate(p_segs) if tiou(g, p) >= 0.3),
                reverse=True,
            )
            used_g: set[int] = set()
            for value, gi, pi in pairs:
                if gi in used_g or pi in matched_p:
                    continue
                used_g.add(gi)
                matched_p.add(pi)
                g, p = g_segs[gi], p_segs[pi]
                boundary.append((name, label, g[0] - p[0], g[1] - p[1], value, g[1] - g[0]))
            for gi, g in enumerate(g_segs):
                if gi not in used_g:
                    unmatched.append(("FN", name, label, g[0], g[1]))
            for pi, p in enumerate(p_segs):
                if pi not in matched_p:
                    extra.append((name, label, p[0], p[1]))

    if not args.quiet:
        print(f"predictions: {args.pred}")
        print(f"annotations: {args.gt}")
        print(f"videos compared: {len(names)}\n")
        header = f"{'class':<20}" + "".join(
            f"  tau={t}: {'TP':>3} {'FP':>3} {'FN':>3} {'P':>6} {'R':>6} {'F1':>6}" for t in TIOU_THRESHOLDS
        )
        print(header)
        print("-" * len(header))
        score_a_parts: list[float] = []
        for label in sorted(per_class):
            row = f"{label:<20}"
            f1s = []
            for tau in TIOU_THRESHOLDS:
                counts = per_class[label][str(tau)]
                metrics = prf(counts["tp"], counts["fp"], counts["fn"])
                f1s.append(metrics["f1"])
                row += (f"  tau={tau}: {counts['tp']:>3} {counts['fp']:>3} "
                        f"{counts['fn']:>3} {metrics['p']:>6.3f} {metrics['r']:>6.3f} "
                        f"{metrics['f1']:>6.3f}")
            score_a_parts.append(sum(f1s) / len(f1s))
            print(row)
        if score_a_parts:
            print("-" * len(header))
            print(f"Score_A (mean of per-class mean-of-tau F1) = "
                  f"{sum(score_a_parts) / len(score_a_parts):.4f}  "
                  f"over {len(score_a_parts)} classes")

        if boundary:
            print(f"\nboundary error on matched events (tIoU >= 0.3), "
                  f"{len(boundary)} matches:")
            starts = [abs(b[2]) for b in boundary]
            ends = [abs(b[3]) for b in boundary]
            print(f"  start: median {sorted(starts)[len(starts) // 2]:.2f}s  "
                  f"max {max(starts):.2f}s")
            print(f"  end:   median {sorted(ends)[len(ends) // 2]:.2f}s  "
                  f"max {max(ends):.2f}s")
        if unmatched:
            print(f"\nfalse negatives ({len(unmatched)}):")
            for _, name, label, start, end in unmatched:
                print(f"  {name}  {label:<20} {start:7.2f} -> {end:7.2f}")
        if extra:
            print(f"\nfalse positives ({len(extra)}):")
            for name, label, start, end in extra:
                print(f"  {name}  {label:<20} {start:7.2f} -> {end:7.2f}")

    summary = {
        "score_a": round(sum(
            sum(prf(c["tp"], c["fp"], c["fn"])["f1"] for c in
                (per_class[label][str(t)] for t in TIOU_THRESHOLDS)) / len(TIOU_THRESHOLDS)
            for label in per_class) / max(1, len(per_class)), 4),
        "per_class": {
            label: {str(tau): prf(per_class[label][str(tau)]["tp"],
                                 per_class[label][str(tau)]["fp"],
                                 per_class[label][str(tau)]["fn"])
                    for tau in TIOU_THRESHOLDS}
            for label in sorted(per_class)
        },
        "matched": len(boundary),
        "false_negatives": len(unmatched),
        "false_positives": len(extra),
        "start_error_median": round(sorted(abs(b[2]) for b in boundary)[len(boundary) // 2], 3) if boundary else None,
        "end_error_median": round(sorted(abs(b[3]) for b in boundary)[len(boundary) // 2], 3) if boundary else None,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
