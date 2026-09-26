#!/usr/bin/env python
"""
run_samples.py — pre-generate the *measured* results shown on the website.

Runs the exact competition entry points on the sample clips:

    solution.detect_events(path)                 # Part A
    solution.RiskEstimator().reset/step(...)     # Part B, one call per frame

with the same ordering the organizers' ``run_submission.py`` uses, then writes
``website/static/data/results.json`` plus a handful of annotated JPEGs the page
shows. Alarm extraction reuses ``evaluate.alarm_starts`` so what the page calls
an alarm is exactly what the official metric calls an alarm.

Read-only with respect to the rest of the repository.

Usage
-----
    cd <repo root>
    python website/tools/run_samples.py                 # both sample clips
    python website/tools/run_samples.py --videos data/dev60/C3897_dev.mp4
    python website/tools/run_samples.py --no-frames     # JSON only, faster
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import solution  # noqa: E402
from evaluate import MERGE_GAP, THETA, alarm_starts  # noqa: E402

WEBSITE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = WEBSITE / "static" / "data" / "results.json"
FRAME_DIR = WEBSITE / "static" / "media" / "events"

CLASS_COLORS = {
    "accident": (60, 60, 240),
    "near_miss": (0, 165, 255),
    "red_light": (220, 80, 220),
    "wrong_way": (255, 120, 0),
    "illegal_u_turn": (0, 200, 200),
    "stopped_vehicle": (200, 200, 60),
    "jaywalking": (0, 210, 120),
    "failure_to_yield": (120, 60, 220),
    "illegal_turn": (255, 170, 0),
    "solid_line_crossing": (170, 170, 255),
    "stop_line": (255, 60, 140),
    "congestion": (90, 90, 250),
    "road_obstacle": (40, 190, 190),
    "fire_smoke": (30, 60, 255),
}
PERSON_COLOR = (60, 220, 60)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True
    ).stdout.strip()


def _r(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _sha256(path: Path) -> str | None:
    """Fingerprint of the threshold config, so a stale JSON is detectable."""
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def code_fingerprint() -> str:
    """A short digest of *everything* that decides the output.

    ``git rev-parse HEAD`` is not enough: a teammate may be mid-edit, and two
    runs at the same commit with different uncommitted work are just as
    incomparable as two different commits. This hashes the commit together with
    the actual tracked-file diff, so any change to the code or the config moves
    the fingerprint.
    """
    import hashlib

    h = hashlib.sha256()
    h.update((git("rev-parse", "HEAD") or "no-git").encode())
    for path in ("src", "configs", "solution.py", "evaluate.py"):
        h.update(_tree_digest(REPO_ROOT / path).encode())
    return h.hexdigest()[:12]


def _tree_digest(path: Path) -> str:
    """Deterministic digest of a directory's file contents (name + bytes)."""
    import hashlib

    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
        return h.hexdigest()[:12]
    if not path.is_dir():
        return "missing"
    for item in sorted(path.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            h.update(str(item.relative_to(path)).encode())
            try:
                h.update(item.read_bytes())
            except OSError:
                pass
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
def detector_status() -> dict[str, Any]:
    """Which detector will actually run, and on which execution provider."""
    from src.config import load_config

    config = load_config(REPO_ROOT / "configs" / "default.json")
    det = config.get("detector", {})
    model = str(det.get("model", ""))
    path = REPO_ROOT / model
    info: dict[str, Any] = {
        "backend": det.get("backend"),
        "model": model,
        "model_present": path.exists(),
        "model_bytes": path.stat().st_size if path.exists() else None,
        "imgsz": det.get("imgsz"),
        "confidence": det.get("confidence"),
        "iou": det.get("iou"),
        "stride": det.get("stride"),
    }
    if path.exists():
        try:
            from src.perception.onnx_detector import OnnxDetector

            detector = OnnxDetector(det)
            info["providers"] = list(detector.active_providers or [])
            info["provider"] = info["providers"][0] if info["providers"] else None
        except Exception as exc:  # pragma: no cover - environment dependent
            info["load_error"] = str(exc)
    return info


def scene_status(video_path: Path) -> dict[str, Any]:
    """Which scene JSON the pipeline will actually select for this clip.

    This mirrors ``PartAPipeline._scene_for_video`` exactly. The order matters:
    an explicit ``scene.path`` in the active config wins over the per-video
    filename, which wins over the generic default. Getting this wrong makes the
    website claim a calibration was "not selected" when in fact it is — so the
    two candidate lists below are both reported.

    Also lists scene files that exist but are reachable by no route, because a
    calibration nobody selects is a failure mode worth surfacing.
    """
    candidates: list[Path] = []
    try:
        from src.config import load_config

        configured = str((load_config(REPO_ROOT / "configs" / "default.json") or {})
                         .get("scene", {}).get("path", "") or "")
    except Exception:
        configured = ""
    if configured:
        p = Path(configured)
        candidates.append(p if p.is_absolute() else REPO_ROOT / p)
    env_path = os.getenv("TRAFFIC_SCENE_CONFIG")
    if env_path:
        p = Path(env_path)
        candidates.append(p if p.is_absolute() else REPO_ROOT / p)
    candidates.extend([
        REPO_ROOT / "configs" / "scenes" / f"{video_path.stem}.json",
        REPO_ROOT / "configs" / "scenes" / "default.json",
    ])

    selected: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            selected = candidate
            break

    def describe(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"path": _rel(path), "error": str(exc)}
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        out = {
            "path": rel,
            "scene_id": data.get("scene_id"),
            "lanes": len(data.get("lanes", [])),
            "stop_lines": len(data.get("stop_lines", [])),
            "solid_lines": len(data.get("solid_lines", [])),
            "crosswalks": len(data.get("crosswalks", [])),
            "traffic_lights": len(data.get("traffic_lights", [])),
            "road_polygons": len(data.get("road_polygons", [])) or (1 if data.get("road_polygon") else 0),
            "has_homography": bool(data.get("homography")),
            "auto_road_fallback": data.get("auto_road_fallback"),
        }
        # Signal-head ROIs in pixels. The report quotes these when it explains
        # how hard the red-light rules are, so they are measured from the scene
        # file rather than asserted in prose.
        rois = []
        for tl in data.get("traffic_lights", []) or []:
            r = tl.get("roi")
            if not (isinstance(r, (list, tuple)) and len(r) == 4):
                continue
            w = (float(r[2]) - float(r[0])) * 1920
            h = (float(r[3]) - float(r[1])) * 1080
            rois.append({
                "id": tl.get("id"),
                "w_px_1080p": round(w, 1),
                "h_px_1080p": round(h, 1),
                "w_px_at_640": round(w * 640 / 1920, 1),
                "h_px_at_640": round(h * 640 / 1920, 1),
            })
        if rois:
            out["traffic_light_rois"] = rois
            out["smallest_roi_px_at_640"] = round(
                min(min(x["w_px_at_640"], x["h_px_at_640"]) for x in rois), 1)
        return out

    info = describe(selected) if selected else {"path": None}
    info["selected_via"] = (
        "config scene.path" if (selected and configured and selected == candidates[0])
        else ("TRAFFIC_SCENE_CONFIG" if (selected and env_path and selected == candidates[1])
              else ("per-video filename" if (selected and selected.stem == video_path.stem)
                    else ("configs/scenes/default.json" if selected else None)))
    )
    # per-camera calibration = a file named after this clip
    info["per_camera_calibrated"] = bool(selected and selected.stem == video_path.stem)
    info["lookup_order"] = [_rel(p) for p in candidates]

    others = []
    scene_dir = REPO_ROOT / "configs" / "scenes"
    if scene_dir.is_dir():
        for candidate in sorted(scene_dir.glob("*.json")):
            if selected and candidate.resolve() == selected.resolve():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            entries = {k: len(data.get(k, []))
                       for k in ("lanes", "stop_lines", "solid_lines", "crosswalks", "traffic_lights")}
            if sum(entries.values()) == 0:
                continue
            others.append({"path": f"configs/scenes/{candidate.name}", **entries})
    info["other_scene_files_with_geometry"] = others
    return info


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
def draw_annotated(frame: np.ndarray, detections: list[Any], header: str) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        name = det.class_name.lower()
        color = PERSON_COLOR if name in {"person", "pedestrian"} else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{det.class_name} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(14, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (20, 20, 20), -1)
        cv2.putText(out, tag, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # banner
    bar = out[0:34, :]
    bar[:] = (18, 18, 18)
    cv2.putText(
        out, header, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA
    )
    return out


def pick_frame_times(events: list[list], duration: float, count: int) -> list[tuple[float, list[str]]]:
    """Choose the timestamps worth showing: mid-event first, then a spread."""
    picks: list[tuple[float, list[str]]] = []
    seen_labels: set[str] = set()
    # longest segment per label first
    by_label: dict[str, list[list]] = defaultdict(list)
    for ev in events:
        by_label[ev[2]].append(ev)
    ranked = sorted(
        ((label, max(segs, key=lambda e: e[1] - e[0])) for label, segs in by_label.items()),
        key=lambda item: item[1][1] - item[1][0],
        reverse=True,
    )
    for label, seg in ranked:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        picks.append(((float(seg[0]) + float(seg[1])) / 2.0, [label]))
        if len(picks) >= count:
            return picks
    # fill with an even spread
    step = max(1e-6, duration / max(1, count))
    i = len(picks)
    while len(picks) < count and i * step < duration:
        t = round(i * step, 2)
        picks.append((t, []))
        i += 1
    return picks


def run_part_b(video_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Replay the harness' Part B loop: one step() per frame, in order."""
    estimator = solution.RiskEstimator()
    estimator.reset(meta)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or meta.get("fps") or 25.0)
    curve: list[list[float]] = []
    index = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        score = estimator.step(frame, t)
        curve.append([round(t, 4), round(float(score), 5)])
        index += 1
    cap.release()
    elapsed = time.perf_counter() - started
    return {"curve": curve, "elapsed": elapsed, "frames": index, "fps": fps}


def summarise_risk(curve: list[list[float]]) -> dict[str, Any]:
    scores = [s for _, s in curve]
    if not scores:
        return {}
    starts = alarm_starts(curve, theta=THETA, merge_gap=MERGE_GAP)
    runs: list[dict[str, Any]] = []
    for s in starts:
        window = [sc for t, sc in curve if s <= t <= s + MERGE_GAP + 1.0]
        runs.append(
            {
                "start": _r(s, 2),
                "peak": _r(max(window) if window else 0.0, 3),
            }
        )
    above = sum(1 for s in scores if s >= THETA)
    # longest contiguous stretch above threshold
    longest = current = 0.0
    run_start = None
    longest_start = None
    for t, s in curve:
        if s >= THETA:
            if run_start is None:
                run_start = t
            current = t - run_start
            if current > longest:
                longest = current
                longest_start = run_start
        else:
            run_start = None
            current = 0.0
    return {
        "threshold": THETA,
        "merge_gap_sec": MERGE_GAP,
        "alarm_count": len(starts),
        "alarm_starts": runs,
        "frames_scored": len(curve),
        "frames_at_or_above_threshold": above,
        "share_at_or_above_threshold": _r(above / len(curve), 4),
        "max": _r(max(scores), 4),
        "mean": _r(float(np.mean(scores)), 4),
        "median": _r(float(np.median(scores)), 4),
        "min": _r(min(scores), 4),
        "longest_run_above_threshold_sec": _r(longest, 2),
        "longest_run_start_sec": _r(longest_start, 2) if longest_start is not None else None,
    }


def write_annotated_frames(
    video_path: Path,
    events: list[list],
    curve: list[list[float]],
    duration: float,
    count: int,
) -> list[dict[str, Any]]:
    try:
        from src.perception.detector import build_detector
        from src.config import load_config
    except Exception as exc:  # pragma: no cover
        print(f"[frames] detector import failed: {exc}", file=sys.stderr)
        return []
    config = load_config(REPO_ROOT / "configs" / "default.json")
    try:
        detector = build_detector(config.get("detector", {}))
    except Exception as exc:
        print(f"[frames] detector unavailable: {exc}", file=sys.stderr)
        return []

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    picks = pick_frame_times(events, duration, count)
    output: list[dict[str, Any]] = []
    for t, forced_labels in picks:
        target = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            continue
        detections = detector.predict(frame)
        risk = 0.0
        for ct, cs in curve:
            if ct <= t + 1e-6:
                risk = cs
            else:
                break
        active = sorted(
            {
                ev[2]
                for ev in events
                if float(ev[0]) - 1e-6 <= t <= float(ev[1]) + 1e-6
            }
            | set(forced_labels)
        )
        header = f"t={t:7.2f}s   risk={risk:.3f}   detections={len(detections)}"
        annotated = draw_annotated(frame, detections, header)
        name = f"{video_path.stem}_t{t:07.2f}.jpg".replace(".", "_", 1)
        out_path = FRAME_DIR / name
        cv2.imwrite(str(out_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        output.append(
            {
                "t": _r(t, 2),
                "frame_index": target,
                "src": f"static/media/events/{name}",
                "detections": len(detections),
                "risk": _r(risk, 4),
                "active_labels": active,
            }
        )
    cap.release()
    return output


# --------------------------------------------------------------------------- #
def score_against_labels(entries: list[dict[str, Any]], labels_path: Path) -> dict[str, Any] | None:
    """Run the OFFICIAL metric against the team's own labels.

    Uses ``evaluate.evaluate`` unmodified, so the numbers here are the graders'
    numbers, not a reimplementation. A labels file that does not exist simply
    yields ``None`` — these are our own annotations, so they are optional and
    the site must work without them.
    """
    if not labels_path.exists():
        return None
    try:
        import evaluate as official
    except Exception as exc:  # pragma: no cover
        return {"error": f"could not import evaluate.py: {exc}"}

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    caveats = {k: v for k, v in raw.items() if k.startswith("_")}
    gt = {k: v for k, v in raw.items() if not k.startswith("_")}

    pred = {
        "team": "website",
        "videos": {
            e["file"]: {"events": [[ev["start"], ev["end"], ev["label"]] for ev in e["events"]],
                        "risk": e.get("risk", {}).get("curve", [])}
            for e in entries
        },
    }
    report = official.evaluate(gt, pred, per_video=True)
    part_a = report.get("part_a") or {}
    per_class = []
    for label, pc in (part_a.get("per_class") or {}).items():
        per_class.append({
            "label": label,
            **{f"f1@{t}": round(pc[str(t)]["f1"], 4) for t in official.TIOU_THRESHOLDS},
            "f1_mean": round(pc["f1_mean"], 4),
            **{
                f"tp_fp_fn@{t}": [pc[str(t)]["tp"], pc[str(t)]["fp"], pc[str(t)]["fn"]]
                for t in official.TIOU_THRESHOLDS
            },
        })
    per_class.sort(key=lambda r: -r["f1_mean"])
    out: dict[str, Any] = {
        "labels_file": str(labels_path.relative_to(REPO_ROOT))
        if labels_path.is_relative_to(REPO_ROOT) else str(labels_path),
        "labels_are": "the team's own annotations, not the organizers'",
        "labelled_videos": sorted(gt),
        "unlabelled_videos": sorted(set(pred["videos"]) - set(gt)),
        "score_a": round(float(part_a.get("score_a", 0.0)), 4),
        "micro": {
            f"f1@{t}": round((part_a.get("micro") or {}).get(str(t), {}).get("f1", 0.0), 4)
            for t in official.TIOU_THRESHOLDS
        },
        "class_agnostic": {
            f"f1@{t}": round((part_a.get("class_agnostic") or {}).get(str(t), {}).get("f1", 0.0), 4)
            for t in official.TIOU_THRESHOLDS
        },
        "per_class": per_class,
        "per_video": part_a.get("per_video"),
        "part_b": None,
        "thresholds": list(official.TIOU_THRESHOLDS),
        "computed_by": "evaluate.evaluate() from the repository root, unmodified",
    }
    part_b = report.get("part_b")
    if part_b:
        out["part_b"] = {
            "score_b": round(float(part_b["score_b"]), 4),
            "ap": round(float(part_b["ap"]), 4),
            "f1_alarm": round(float(part_b["f1_alarm"]), 4),
            "mtta_sec": round(float(part_b["mtta_sec"]), 4),
            "n_accidents": part_b["n_accidents"],
        }
    out["model_score"] = round(float(report.get("model_score", 0.0)), 4)
    if caveats.get("_caveats"):
        out["caveats"] = caveats["_caveats"]
    if caveats.get("_comment"):
        out["comment"] = caveats["_comment"]
    return out


def build_payload(entries: list[dict[str, Any]], args, out: Path) -> dict[str, Any]:
    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "website/tools/run_samples.py",
        "repo_commit": git("rev-parse", "--short", "HEAD") or None,
        "repo_dirty": bool(git("status", "--porcelain")),
        "entry_points": ["solution.detect_events", "solution.RiskEstimator.step"],
        "config": "configs/default.json (unmodified)",
        "detector": detector_status(),
        "class_list": list(solution.CLASSES),
        "risk_horizon_sec": solution.RISK_HORIZON_SEC,
        "alarm_rule": {
            "threshold": THETA,
            "merge_gap_sec": MERGE_GAP,
            "source": "evaluate.alarm_starts (the official metric's own function)",
        },
        "notes": [
            "This file is regenerated output, not a claim about the hidden test set.",
            "Numbers are the current pipeline on these two dev clips with the current "
            "configs; the code fingerprint above pins exactly what produced them, so "
            "re-running the script after a change may legitimately give different numbers.",
        ],
        "videos": entries,
        "expected_videos": len(args.videos),
        "partial": len(entries) < len(args.videos),
    }
    scored = score_against_labels(entries, REPO_ROOT / args.labels)
    if scored is not None:
        payload["scored"] = scored
        payload["notes"] = [
            n for n in payload["notes"]
            if "No human labels exist" not in n
        ] + [
            "A team-authored labels file was found, so the event counts above ARE "
            "scored against it. See the 'scored' block and the scoring section of the page."
        ]
    return payload


def process_video(path: Path, args) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = cap_frames / fps if fps else 0.0

    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        events = solution.detect_events(str(path))
        part_a_sec = time.perf_counter() - t0
        for item in caught:
            text = str(item.message)
            if text not in captured:
                captured.append(text)

    # validate the way the official harness would
    by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ev in events:
        by_class[ev[2]].append((float(ev[0]), float(ev[1])))
    overlaps = 0
    for _label, segs in by_class.items():
        segs.sort()
        for (s1, e1), (s2, _e2) in zip(segs, segs[1:]):
            if s2 < e1:
                overlaps += 1
    bad_labels = sorted({ev[2] for ev in events if ev[2] not in solution.CLASSES})

    print(
        f"           part A {part_a_sec:.1f}s -> {len(events)} events "
        f"({len(by_class)} classes, {overlaps} same-class overlaps)",
        flush=True,
    )

    risk_block: dict[str, Any] = {}
    curve: list[list[float]] = []
    part_b_sec = None
    if not args.no_part_b:
        meta = {
            "video_id": path.name,
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames": cap_frames,
        }
        part_b = run_part_b(path, meta)
        part_b_sec = part_b["elapsed"]
        curve = part_b["curve"]
        summary = summarise_risk(curve)
        risk_block = {
            "every_n_frames": args.risk_stride,
            "points_total": len(curve),
            **summary,
            "curve": curve[:: args.risk_stride],
        }
        print(
            f"           part B {part_b_sec:.1f}s -> {len(curve)} points, "
            f"max {summary.get('max')}, {summary.get('alarm_count')} alarms",
            flush=True,
        )

    frames: list[dict[str, Any]] = []
    if not args.no_frames:
        frames = write_annotated_frames(path, events, curve, duration, args.frames)
        print(f"           wrote {len(frames)} annotated frames", flush=True)

    event_records = [
        {
            "start": _r(float(e[0]), 3),
            "end": _r(float(e[1]), 3),
            "label": e[2],
            "duration": _r(float(e[1]) - float(e[0]), 3),
        }
        for e in events
    ]
    event_records.sort(key=lambda e: (e["start"], e["label"]))

    return {
        "id": path.stem,
        "file": path.name,
        "source": "measured",
        # Stamped per video, not once per file: a long run can straddle someone
        # else's commit, and two clips analysed by different revisions of the
        # pipeline are not comparable. The page shows this.
        "code_revision": git("rev-parse", "--short", "HEAD") or None,
        "code_dirty": bool(git("status", "--porcelain")),
        "code_fingerprint": code_fingerprint(),
        "config_sha": _sha256(REPO_ROOT / "configs" / "default.json"),
        "container": {
            "width": width,
            "height": height,
            "fps": _r(fps, 5),
            "frames": cap_frames,
            "duration_sec": _r(duration, 3),
        },
        "scene": scene_status(path),
        "events": event_records,
        "event_counts_by_label": {k: len(v) for k, v in sorted(by_class.items())},
        "event_seconds_by_label": {
            k: _r(sum(e - s for s, e in v), 2) for k, v in sorted(by_class.items())
        },
        "validation": {
            "same_class_overlaps": overlaps,
            "labels_outside_official_list": bad_labels,
            "all_intervals_valid": all(
                0.0 <= e["start"] < e["end"] <= duration + 0.5 for e in event_records
            ),
        },
        "runtime": {
            "part_a_sec": _r(part_a_sec, 1),
            "part_b_sec": _r(part_b_sec, 1) if part_b_sec is not None else None,
            "video_duration_sec": _r(duration, 2),
            "part_a_realtime_factor": _r(part_a_sec / duration, 3) if duration else None,
            "part_b_realtime_factor": _r(part_b_sec / duration, 3) if (part_b_sec and duration) else None,
        },
        "risk": risk_block,
        "annotated_frames": frames,
        "warnings": captured,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[
            str(REPO_ROOT / "data" / "samples_small" / "C3897_small.mp4"),
            str(REPO_ROOT / "data" / "samples_small" / "C3902_small.mp4"),
        ],
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--risk-stride", type=int, default=3, help="keep every Nth risk point in the JSON")
    parser.add_argument("--frames", type=int, default=8, help="annotated frames per video")
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--no-part-b", action="store_true", help="events only, skip the risk loop")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip videos already present in --out (each video is flushed as it finishes)",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="re-derive the scene/config provenance already in --out, without "
             "re-running inference. Use after a commit changes only configs/ or "
             "the scene resolution order.",
    )
    parser.add_argument(
        "--retry-on-drift", action="store_true",
        help="if the code changes between clips, discard the partial set and start "
             "over, so the file never mixes code states (default: warn only)",
    )
    parser.add_argument(
        "--max-restarts", type=int, default=2,
        help="how many times --retry-on-drift may restart before giving up (default 2)",
    )
    parser.add_argument(
        "--labels", default="data/annotations/my_labels.json",
        help="optional team-authored ground truth in evaluate.py's format. When "
             "present, the official metric is run against it and the result is "
             "embedded in the JSON. Missing file is not an error.",
    )
    args = parser.parse_args()

    out = Path(args.out)

    if args.metadata_only:
        if not out.exists():
            print(f"[results] {out} does not exist; nothing to refresh", file=sys.stderr)
            return 2
        payload = json.loads(out.read_text(encoding="utf-8"))
        by_name = {Path(v).name: Path(v) for v in args.videos}
        changed = 0
        for entry in payload.get("videos", []):
            path = by_name.get(entry.get("file"))
            if path is None or not path.exists():
                continue
            fresh = scene_status(path)
            if fresh != entry.get("scene"):
                print(f"[results] scene for {entry['file']}: "
                      f"{(entry.get('scene') or {}).get('path')} -> {fresh.get('path')}", flush=True)
                entry["scene"] = fresh
                changed += 1
            # NOTE: code_revision / code_fingerprint / config_sha describe what
            # produced the EVENTS. Re-stamping them here would be a lie: the
            # inference did not run again. They are deliberately left alone, and
            # any divergence from the current tree is reported instead.
        cfgs = {e.get("config_sha") for e in payload.get("videos", [])}
        fps = {e.get("code_fingerprint") for e in payload.get("videos", []) if e.get("code_fingerprint")}
        payload["detector"] = detector_status()
        payload["metadata_refreshed_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["tree_at_refresh"] = {
            "code_fingerprint": code_fingerprint(),
            "repo_commit": git("rev-parse", "--short", "HEAD") or None,
            "repo_dirty": bool(git("status", "--porcelain")),
            "config_sha": _sha256(REPO_ROOT / "configs" / "default.json"),
        }
        now_fp = payload["tree_at_refresh"]["code_fingerprint"]
        if fps and now_fp not in fps:
            payload["warning"] = (
                "The metadata (scene selection, detector availability, metric score) in this file "
                "was refreshed from a DIFFERENT code state (" + now_fp + ") than the events "
                "(" + ", ".join(sorted(fps)) + "). The event and risk numbers are unchanged; re-run "
                "website/tools/run_samples.py to regenerate them against the current tree."
            )
            payload["metadata_stale"] = True
        else:
            payload["metadata_stale"] = False
        payload.pop("single_revision", None)
        payload["single_fingerprint"] = len(fps) <= 1
        # The metric score is a pure function of the events already in this file
        # plus the labels, so refreshing it costs nothing and must not require
        # re-running inference. This is also how a newly written labels file
        # gets picked up.
        scored = score_against_labels(payload.get("videos", []), REPO_ROOT / args.labels)
        if scored is not None:
            payload["scored"] = scored
            payload["notes"] = [n for n in payload.get("notes", [])
                                if "No human labels exist" not in n] + [
                "A team-authored labels file was found, so the event counts ARE scored "
                "against it. See the 'scored' block."
            ]
            print(f"[results] scored against {scored.get('labels_file')}: "
                  f"Score_A = {scored.get('score_a')}", flush=True)
        else:
            payload.pop("scored", None)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"[results] refreshed metadata in {out} ({changed} scene block(s) changed); "
              f"events and risk untouched", flush=True)
        if payload.get("metadata_stale"):
            print("[results] WARNING  : " + payload["warning"], flush=True)
        return 0

    videos = [Path(v) for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        print("missing videos: " + ", ".join(str(m) for m in missing), file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    if args.resume and out.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8"))
            entries = list(previous.get("videos") or [])
            if entries:
                print(f"[results] resuming, already have: " + ", ".join(e["file"] for e in entries), flush=True)
        except Exception as exc:
            print(f"[results] could not resume from {out}: {exc}", file=sys.stderr)
            entries = []

    done = {e["id"] for e in entries}
    restarts = 0
    for path in videos:
        if path.stem in done:
            print(f"[results] {path.name} already present, skipping", flush=True)
            continue
        print(f"[results] {path.name}", flush=True)
        entry = process_video(path, args)
        entries = [e for e in entries if e["id"] != entry["id"]] + [entry]
        entries.sort(key=lambda e: e["id"])
        # flush after EVERY video: a long run that gets interrupted still leaves
        # a loadable, complete-for-what-it-has JSON behind
        out.write_text(json.dumps(build_payload(entries, args, out), indent=1), encoding="utf-8")
        print(f"[results] flushed {out} ({len(entries)}/{len(videos)} videos)", flush=True)
        # Drift guard. Two clips analysed by different revisions of the pipeline
        # are not a matched pair, and showing them side by side is the whole
        # point. If someone committed while we were running, throw the work away
        # and start over rather than shipping a file that quietly mixes versions.
        if args.retry_on_drift and entries and len(entries) < len(videos):
            head_now = git("rev-parse", "--short", "HEAD") or None
            fp_now = code_fingerprint()
            revs = {e.get("code_revision") for e in entries}
            fps = {e.get("code_fingerprint") for e in entries}
            drifted = (len(fps) > 1 or len(revs) > 1
                       or entries[-1].get("code_fingerprint") != fp_now)
            if drifted and restarts < args.max_restarts:
                restarts += 1
                print(f"[results] the code changed under us during the run (revisions "
                      + ", ".join(sorted(str(r) for r in revs)) + " -> " + str(head_now)
                      + "; fingerprint " + ", ".join(sorted(str(f) for f in fps))
                      + f" -> {fp_now}); discarding and restarting "
                      f"({restarts}/{args.max_restarts})", flush=True)
                entries = []
                out.unlink(missing_ok=True)
                done = set()
            elif drifted:
                print("[results] WARNING  : the code kept changing and the restart budget is "
                      f"spent; finishing with a mixed set and flagging it in the output", flush=True)
                out.write_text(json.dumps(build_payload(entries, args, out), indent=1), encoding="utf-8")

    payload = build_payload(entries, args, out)
    revs = {e.get("code_revision") for e in entries}
    cfgs = {e.get("config_sha") for e in entries}
    fps = {e.get("code_fingerprint") for e in entries}
    payload["code_fingerprint"] = code_fingerprint()
    payload["single_revision"] = len(revs) == 1
    payload["single_config"] = len(cfgs) == 1
    payload["single_fingerprint"] = len(fps) <= 1
    if not payload["single_fingerprint"]:
        payload["warning"] = (
            "The clips in this file were analysed from DIFFERENT code states ("
            + ", ".join(sorted(str(f) for f in fps)) + "; revisions "
            + ", ".join(sorted(str(r) for r in revs))
            + "). Re-run website/tools/run_samples.py --retry-on-drift to make them comparable."
        )
    elif not payload["single_revision"]:
        payload["warning"] = (
            "The clips in this file were analysed by DIFFERENT code revisions "
            + ", ".join(sorted(str(r) for r in revs))
            + ". Re-run website/tools/run_samples.py without --resume to make them comparable."
        )
    elif not payload["single_config"]:
        payload["warning"] = "The clips in this file used different configs/default.json contents."
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[results] wrote {out} ({out.stat().st_size / 1024:.0f} KB)", flush=True)
    if payload.get("warning"):
        print("[results] WARNING  : " + payload["warning"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
