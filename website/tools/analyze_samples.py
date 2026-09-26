#!/usr/bin/env python
"""
analyze_samples.py — EDA pre-generator for the public team website.

Runs the repository's own YOLO11-ONNX detector plus a frame-difference motion
estimate over the two dev copies of the sample clips and writes ONE JSON file,
``website/static/data/eda.json``, that the static page renders. Every number on
the EDA section of the website comes from this file — nothing is hand written.

It also writes small preview media used by the page (see ``--media``).

Read-only with respect to the rest of the repository: it imports ``src`` but
never writes outside ``website/``.

Usage
-----
    cd <repo root>                       # so weights/yolo11n.onnx resolves
    python website/tools/analyze_samples.py

    # fast smoke run (only the first 60 s of each clip)
    python website/tools/analyze_samples.py --limit-seconds 60 --out /tmp/eda_smoke.json

    # skip the media step (JSON only)
    python website/tools/analyze_samples.py --no-media
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.detector import RELEVANT_NAMES  # noqa: E402
from src.perception.onnx_detector import OnnxDetector  # noqa: E402

WEBSITE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = WEBSITE / "static" / "data" / "eda.json"
DEFAULT_MEDIA = WEBSITE / "static" / "media"
DEFAULT_IMG = WEBSITE / "static" / "img"

# Classes we treat as "vehicle" for the density chart (mirrors TrackManager).
VEHICLE_NAMES = {"car", "bus", "truck", "motorcycle", "bicycle", "vehicle"}
PERSON_NAMES = {"person", "pedestrian"}

# Motion grid resolution for the spatial activity heatmap.
MOTION_W, MOTION_H = 64, 36


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ffprobe(path: Path) -> dict[str, Any]:
    """Container-level facts straight from ffprobe (independent of OpenCV)."""
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate,format_name",
        "-show_entries",
        "stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt",
        "-of", "json", str(path),
    ]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout)
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": str(exc)}
    out: dict[str, Any] = {}
    fmt = data.get("format", {})
    for key in ("duration", "size", "bit_rate", "format_name"):
        if key in fmt:
            out[key] = float(fmt[key]) if key != "format_name" else fmt[key]
    streams = [s for s in data.get("streams", []) if s.get("width")]
    if streams:
        out["stream"] = streams[0]
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _r(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


# --------------------------------------------------------------------------- #
# main per-video analysis
# --------------------------------------------------------------------------- #
def analyse_video(
    path: Path,
    detector: OnnxDetector,
    stride: int,
    limit_seconds: float | None,
    motion_stride: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    meta_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    # Re-open for the actual pass: the metadata read above consumed nothing but
    # leaving it closed would silently decode zero frames.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot re-open {path}")

    probe = ffprobe(path)

    # ---- accumulators ----------------------------------------------------- #
    class_totals: dict[str, int] = defaultdict(int)
    class_frames_present: dict[str, int] = defaultdict(int)
    class_max: dict[str, int] = defaultdict(int)
    sampled_frames = 0
    objects_per_frame: list[int] = []
    vehicles_per_frame: list[int] = []
    persons_per_frame: list[int] = []
    bus_counts: list[tuple[int, int]] = []
    truck_counts: list[tuple[int, int]] = []

    # per-second buckets (index = floor(t))
    bucket_objects: dict[int, list[int]] = defaultdict(list)
    bucket_vehicles: dict[int, list[int]] = defaultdict(list)
    bucket_persons: dict[int, list[int]] = defaultdict(list)
    bucket_buses: dict[int, list[int]] = defaultdict(list)
    bucket_trucks: dict[int, list[int]] = defaultdict(list)
    bucket_brightness: dict[int, list[float]] = defaultdict(list)
    bucket_motion: dict[int, list[float]] = defaultdict(list)
    bucket_matched: dict[int, int] = defaultdict(int)

    motion_sum = np.zeros((MOTION_H, MOTION_W), dtype=np.float64)
    motion_hits = np.zeros((MOTION_H, MOTION_W), dtype=np.float64)
    occupancy = np.zeros((MOTION_H, MOTION_W), dtype=np.float64)
    occupancy_hits = np.zeros((MOTION_H, MOTION_W), dtype=np.float64)
    brightness_all: list[float] = []

    previous_small: np.ndarray | None = None
    decoded = 0
    last_t = 0.0
    t0_wall = _dt.datetime.now()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = decoded / meta_fps if meta_fps > 0 else 0.0
        if limit_seconds is not None and t >= limit_seconds:
            break
        decoded += 1
        last_t = t
        second = int(t)

        # ---- motion / brightness (every motion_stride-th frame) ---------- #
        if decoded % motion_stride == 0 or previous_small is None:
            small = cv2.resize(frame, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
            bright = float(gray.mean())
            brightness_all.append(bright)
            bucket_brightness[second].append(bright)
            if previous_small is not None:
                diff = cv2.absdiff(gray, previous_small).astype(np.float64)
                motion_sum += diff
                motion_hits += 1.0
                bucket_motion[second].append(float(diff.mean()))
            previous_small = gray

        # ---- detector (every stride-th frame) --------------------------- #
        if (decoded - 1) % stride == 0:
            detections = detector.predict(frame)
            sampled_frames += 1
            per_class: dict[str, int] = defaultdict(int)
            vehicles = persons = buses = trucks = 0
            for det in detections:
                name = det.class_name.lower().replace(" ", "_")
                per_class[name] += 1
                if name in VEHICLE_NAMES:
                    vehicles += 1
                if name in PERSON_NAMES:
                    persons += 1
                if name == "bus":
                    buses += 1
                if name == "truck":
                    trucks += 1
                # occupancy: bottom-centre of the box = ground contact point
                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2.0
                cy = y2
                gx = min(MOTION_W - 1, max(0, int(cx / max(1, meta_width) * MOTION_W)))
                gy = min(MOTION_H - 1, max(0, int(cy / max(1, meta_height) * MOTION_H)))
                occupancy[gy, gx] += 1.0
                occupancy_hits[gy, gx] += 1.0
            total = len(detections)
            for name, count in per_class.items():
                class_totals[name] += count
                class_frames_present[name] += 1
                class_max[name] = max(class_max[name], count)
            objects_per_frame.append(total)
            vehicles_per_frame.append(vehicles)
            persons_per_frame.append(persons)
            bus_counts.append((decoded, buses))
            truck_counts.append((decoded, trucks))
            bucket_objects[second].append(total)
            bucket_vehicles[second].append(vehicles)
            bucket_persons[second].append(persons)
            bucket_buses[second].append(buses)
            bucket_trucks[second].append(trucks)
            bucket_matched[second] += 1
        del frame

    cap.release()
    wall = (_dt.datetime.now() - t0_wall).total_seconds()

    fps = meta_fps
    duration = decoded / fps if fps > 0 else 0.0

    # ---- per-second timeline --------------------------------------------- #
    timeline: list[dict[str, Any]] = []
    for second in sorted(bucket_objects):
        timeline.append(
            {
                "t": second,
                "objects": _r(statistics.fmean(bucket_objects[second]), 2),
                "vehicles": _r(statistics.fmean(bucket_vehicles[second]), 2),
                "persons": _r(statistics.fmean(bucket_persons[second]), 2),
                "buses": _r(statistics.fmean(bucket_buses[second]), 2),
                "trucks": _r(statistics.fmean(bucket_trucks[second]), 2),
                "brightness": _r(statistics.fmean(bucket_brightness[second]), 1) if bucket_brightness[second] else None,
                "motion": _r(statistics.fmean(bucket_motion[second]), 3) if bucket_motion[second] else None,
                "sampled_frames": bucket_matched[second],
            }
        )

    # ---- windowed statistics (10 s sliding windows) ---------------------- #
    def window_stats(series: list[tuple[int, int]], window: int = 10) -> dict[str, Any]:
        if not series:
            return {}
        by_second: dict[int, list[int]] = defaultdict(list)
        for sec, value in series:
            by_second[sec].append(value)
        per_sec = {s: statistics.fmean(v) for s, v in by_second.items()}
        seconds = sorted(per_sec)
        best = worst = None
        if len(seconds) >= window:
            for i in range(len(seconds) - window + 1):
                chunk = [per_sec[seconds[i + k]] for k in range(window)]
                mean = statistics.fmean(chunk)
                if best is None or mean > best[0]:
                    best = (mean, seconds[i], seconds[i + window - 1] + 1)
                if worst is None or mean < worst[0]:
                    worst = (mean, seconds[i], seconds[i + window - 1] + 1)
        out: dict[str, Any] = {"window_sec": window}
        if best:
            out["busiest"] = {
                "start_sec": best[1], "end_sec": best[2],
                "mean_objects": _r(best[0], 2),
            }
        if worst:
            out["quietest"] = {
                "start_sec": worst[1], "end_sec": worst[2],
                "mean_objects": _r(worst[0], 2),
            }
        if best and worst and worst[0] > 0:
            out["busiest_over_quietest"] = _r(best[0] / worst[0], 2)
        return out

    obj_series = [
        (_r(i * stride / fps, 2) if fps else i * stride, v)
        for i, v in enumerate(objects_per_frame)
    ]

    # ---- heatmaps --------------------------------------------------------- #
    with np.errstate(invalid="ignore", divide="ignore"):
        motion_mean = np.where(motion_hits > 0, motion_sum / np.maximum(motion_hits, 1e-9), 0.0)
    occ_mean = np.where(occupancy_hits > 0, occupancy / np.maximum(occupancy_hits, 1e-9), 0.0)

    def pack(grid: np.ndarray, note: str, totals: np.ndarray | None = None) -> dict[str, Any]:
        payload = {
            "grid_w": MOTION_W,
            "grid_h": MOTION_H,
            "min": _r(float(grid.min()), 4),
            "max": _r(float(grid.max()), 4),
            "mean": _r(float(grid.mean()), 4),
            "note": note,
            "values": [round(float(v), 4) for v in grid.reshape(-1)],
        }
        if totals is not None:
            payload["totals"] = [int(v) for v in totals.reshape(-1)]
            payload["total_max"] = int(totals.max())
        return payload

    # ---- assemble --------------------------------------------------------- #
    per_class: dict[str, Any] = {}
    for name in sorted(class_totals):
        per_class[name] = {
            "detections_total": int(class_totals[name]),
            "sampled_frames_with_at_least_one": int(class_frames_present[name]),
            "share_of_sampled_frames": _r(
                class_frames_present[name] / max(1, sampled_frames), 4
            ),
            "max_in_one_frame": int(class_max[name]),
            "mean_per_sampled_frame": _r(class_totals[name] / max(1, sampled_frames), 3),
        }

    ped_frames = sum(1 for p in persons_per_frame if p > 0)
    bus_series = [(int(i * stride / fps) if fps else 0, c) for i, (_, c) in enumerate(bus_counts)]
    truck_series = [(int(i * stride / fps) if fps else 0, c) for i, (_, c) in enumerate(truck_counts)]

    result: dict[str, Any] = {
        "id": path.stem,
        "file": path.name,
        "container": {
            "width": meta_width,
            "height": meta_height,
            "megapixels": _r(meta_width * meta_height / 1e6, 2),
            "fps": _r(fps, 5),
            "frames_decoded": decoded,
            "frames_reported_by_container": meta_frames,
            "frame_count_agrees": bool(meta_frames == decoded),
            "duration_sec": _r(duration, 3),
            "size_bytes": int(probe.get("size", path.stat().st_size)),
            "bitrate_bps": int(probe.get("bit_rate", 0)) or None,
            "codec": probe.get("stream", {}).get("codec_name"),
            "profile": probe.get("stream", {}).get("profile"),
            "pix_fmt": probe.get("stream", {}).get("pix_fmt"),
            "fps_rational": probe.get("stream", {}).get("r_frame_rate"),
            "audio_streams": 0,
        },
        "detection": {
            "stride": stride,
            "motion_stride": motion_stride,
            "sampled_frames": sampled_frames,
            "detections_total": int(sum(class_totals.values())),
            "objects_per_sampled_frame": {
                "mean": _r(statistics.fmean(objects_per_frame) if objects_per_frame else 0.0, 3),
                "median": _r(statistics.median(objects_per_frame) if objects_per_frame else 0.0, 3),
                "min": int(min(objects_per_frame)) if objects_per_frame else 0,
                "max": int(max(objects_per_frame)) if objects_per_frame else 0,
                "p05": _r(percentile([float(v) for v in objects_per_frame], 5), 2),
                "p95": _r(percentile([float(v) for v in objects_per_frame], 95), 2),
                "stdev": _r(statistics.pstdev(objects_per_frame) if objects_per_frame else 0.0, 3),
            },
            "vehicles_per_sampled_frame_mean": _r(
                statistics.fmean(vehicles_per_frame) if vehicles_per_frame else 0.0, 3
            ),
            "pedestrians": {
                "sampled_frames_with_person": ped_frames,
                "share_of_sampled_frames": _r(ped_frames / max(1, sampled_frames), 4),
                "mean_per_sampled_frame": _r(
                    statistics.fmean(persons_per_frame) if persons_per_frame else 0.0, 3
                ),
                "max_in_one_frame": int(max(persons_per_frame)) if persons_per_frame else 0,
                "peak_time_sec": _r(
                    (persons_per_frame.index(max(persons_per_frame)) * stride / fps)
                    if persons_per_frame and fps else 0.0, 2
                ),
            },
            "per_class": per_class,
            "windows": window_stats(obj_series),
            "bus_peak": {
                "max_concurrent": int(max((c for _, c in bus_counts), default=0)),
                "at_sec": _r(
                    max(bus_counts, key=lambda item: item[1])[0] / fps if fps and bus_counts else 0.0, 2
                ) if bus_counts else 0.0,
            },
            "truck_peak": {
                "max_concurrent": int(max((c for _, c in truck_counts), default=0)),
                "at_sec": _r(
                    max(truck_counts, key=lambda item: item[1])[0] / fps if fps and truck_counts else 0.0, 2
                ) if truck_counts else 0.0,
            },
            "timeline_1s": timeline,
        },
        "heatmaps": {
            "motion": pack(
                motion_mean,
                "Mean absolute frame difference per cell, grey frames resized to "
                f"{MOTION_W}x{MOTION_H}; 0 = nothing moved, max = the busiest cell.",
            ),
            "occupancy": pack(
                occ_mean,
                "Mean number of detected road users per cell, using the bottom-centre "
                "(ground-contact) point of each box. This is where traffic actually is.",
                totals=occupancy,
            ),
        },
        "brightness": {
            "mean": _r(statistics.fmean(brightness_all) if brightness_all else 0.0, 2),
            "min": _r(min(brightness_all) if brightness_all else 0.0, 2),
            "max": _r(max(brightness_all) if brightness_all else 0.0, 2),
            "samples": len(brightness_all),
        },
        "runtime_sec": _r(wall, 1),
        "last_timestamp_sec": _r(last_t, 3),
    }
    return result


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #
def build_media(
    paths: list[Path],
    out_dir: Path,
    img_dir: Path,
    width: int,
    preview_fps: int,
    video_bitrate: str,
) -> list[dict[str, Any]]:
    """Encode seekable low-bitrate previews + a still from each clip."""
    if shutil.which("ffmpeg") is None:
        print("[media] ffmpeg not found - skipping previews", file=sys.stderr)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, Any]] = []
    for path in paths:
        stem = path.stem
        webm = out_dir / f"{stem}_preview.webm"
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(path), "-an",
            "-vf", f"scale={width}:-2,fps={preview_fps}",
            "-c:v", "libvpx-vp9", "-b:v", video_bitrate, "-deadline", "good",
            "-cpu-used", "4", "-row-mt", "1", "-pix_fmt", "yuv420p", str(webm),
        ]
        subprocess.run(cmd, check=True)
        # still frame from the middle of the clip
        dur = None
        probe = ffprobe(path)
        if probe.get("duration"):
            dur = float(probe["duration"])
        still = img_dir / f"{stem}_still.jpg"
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{dur * 0.5 if dur else 5:.2f}",
            "-i", str(path), "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "4", str(still),
        ]
        subprocess.run(cmd, check=True)
        made.append(
            {
                "id": stem,
                "video": f"static/media/{webm.name}",
                "still": f"static/img/{still.name}",
                "bytes": webm.stat().st_size,
                "width": width,
                "height": int(round(width * 1080 / 1920)),
                "fps": preview_fps,
            }
        )
        print(f"[media] {webm.name}  {webm.stat().st_size / 1e6:.1f} MB", flush=True)
    return made


# --------------------------------------------------------------------------- #
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
    parser.add_argument("--stride", type=int, default=3, help="detector frame stride")
    parser.add_argument("--motion-stride", type=int, default=1, help="frame-difference stride")
    parser.add_argument("--limit-seconds", type=float, default=None)
    parser.add_argument("--model", default="weights/yolo11n.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--media-width", type=int, default=480)
    parser.add_argument("--media-fps", type=int, default=15)
    parser.add_argument("--media-bitrate", default="300k")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        print("missing sample videos: " + ", ".join(str(m) for m in missing), file=sys.stderr)
        return 2

    print(f"[eda] loading detector {args.model}", flush=True)
    detector = OnnxDetector(
        {
            "model": args.model,
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "iou": args.iou,
            "device": "cpu",
        }
    )
    model_file = REPO_ROOT / args.model
    detector_info = {
        "model": args.model,
        "model_bytes": model_file.stat().st_size if model_file.exists() else None,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "iou": args.iou,
        "provider": (detector.active_providers or ["unknown"])[0],
        "providers": list(detector.active_providers or []),
        "stride": args.stride,
        "class_names_filter": sorted(RELEVANT_NAMES),
        "vehicle_classes": sorted(VEHICLE_NAMES),
    }

    results: list[dict[str, Any]] = []
    for path in videos:
        print(f"[eda] {path.name} ...", flush=True)
        results.append(
            analyse_video(path, detector, args.stride, args.limit_seconds, args.motion_stride)
        )
        print(
            f"      {results[-1]['container']['frames_decoded']} frames, "
            f"{results[-1]['detection']['detections_total']} detections, "
            f"{results[-1]['runtime_sec']}s",
            flush=True,
        )

    media: list[dict[str, Any]] = []
    if not args.no_media:
        print("[eda] encoding previews ...", flush=True)
        media = build_media(
            videos, DEFAULT_MEDIA, DEFAULT_IMG, args.media_width, args.media_fps, args.media_bitrate
        )

    # ---- cross-video comparison ------------------------------------------ #
    comparison: dict[str, Any] = {}
    if len(results) == 2:
        a, b = results
        comparison = {
            "same_resolution": a["container"]["width"] == b["container"]["width"]
            and a["container"]["height"] == b["container"]["height"],
            "same_fps": a["container"]["fps"] == b["container"]["fps"],
            "same_frame_count": a["container"]["frames_decoded"] == b["container"]["frames_decoded"],
            "mean_objects_a": a["detection"]["objects_per_sampled_frame"]["mean"],
            "mean_objects_b": b["detection"]["objects_per_sampled_frame"]["mean"],
            "pedestrian_share_a": a["detection"]["pedestrians"]["share_of_sampled_frames"],
            "pedestrian_share_b": b["detection"]["pedestrians"]["share_of_sampled_frames"],
            "busiest_start_a": a["detection"]["windows"].get("busiest", {}).get("start_sec"),
            "busiest_start_b": b["detection"]["windows"].get("busiest", {}).get("start_sec"),
        }

    notes = [
        "Both clips are 1920x1080 downscaled dev copies of the 4K originals; the originals "
        "are 5.4 GB each and are not committed to git.",
        "The two clips are only ~5.3 minutes long, so NOTHING here supports a claim about "
        "time-of-day traffic profiles. All 'busiest' statements are within-clip.",
        "Object counts come from the bundled COCO-pretrained yolo11n.onnx, not from a model "
        "fine-tuned for this camera. Small/occluded objects are systematically under-counted, "
        "so treat absolute counts as a lower bound.",
        "Traffic light heads and distant riders are near the detector's confidence floor, so "
        "'traffic light' appears in very few frames.",
    ]

    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "website/tools/analyze_samples.py",
        "repo_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or None,
        "repo_dirty": bool(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                capture_output=True, text=True,
            ).stdout.strip()
        ),
        "limit_seconds": args.limit_seconds,
        "detector": detector_info,
        "videos": results,
        "media": media,
        "comparison": comparison,
        "notes": notes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[eda] wrote {out} ({out.stat().st_size / 1024:.0f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
