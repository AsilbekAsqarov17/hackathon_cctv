#!/usr/bin/env python3
"""
app.py — the live-demo backend for the team website.

Design constraints, in priority order:

1. **Use only the standard library.** The repository already needs numpy,
   OpenCV and ONNX Runtime for inference; adding FastAPI/uvicorn on top would
   add an install step that can fail on an offline judging machine, for no
   benefit. ``http.server`` is enough for one file, one job, one result.

2. **Never 500 on a missing checkpoint.** If ``weights/yolo11n.onnx`` is not
   present the job still completes: it reports the real container metadata,
   returns an empty event list, a flat risk curve and ``degraded: true`` with a
   reason a human can act on.

3. **Call the real entry points.** ``solution.detect_events`` and
   ``solution.RiskEstimator`` are imported from the repository root, so the demo
   cannot silently diverge from what the graders run.

4. **Report progress.** A 2-minute clip takes roughly 2 minutes on CPU. The
   client polls ``GET /api/jobs/<id>``; this process publishes a stage and a
   percentage from a worker thread while Part A runs.

5. **Support HTTP Range.** The preview clips must seek instantly; the stdlib
   handler does not do ranges, so they are implemented here.

Run
---
    python website/demo/app.py                 # 127.0.0.1:8000
    python website/demo/app.py --port 9000 --host 0.0.0.0
    python website/demo/app.py --static-only   # serve the site, disable upload

Endpoints
---------
    GET  /                      the site
    GET  /api/health            detector status, limits, versions
    POST /api/jobs              multipart or raw body -> {"job_id": "..."}
    GET  /api/jobs/<id>         {"state","stage","progress","result"?,"error"?}
    POST /api/jobs/<id>/cancel  best-effort cancellation
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
import warnings
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

WEBSITE = Path(__file__).resolve().parent.parent
REPO_ROOT = WEBSITE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------ limits
MAX_SECONDS = 120.0
MAX_BYTES = 150 * 1024 * 1024
MAX_CONCURRENT = 1
JOB_TTL_SEC = 3600
# frames rendered back as annotated JPEG
ANNOTATED_FRAMES = 8
RISK_STRIDE = 3

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}
_ALLOWED = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime",
            "m4v": "video/x-m4v", "avi": "video/x-msvideo", "mkv": "video/x-matroska"}


# ------------------------------------------------------------------ repo
class Pipeline:
    """Lazily imports the competition code and caches it across jobs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._solution = None
        self.import_error: str | None = None
        self.detector: dict[str, Any] = {}

    def probe_detector(self) -> dict[str, Any]:
        """Which detector will actually run, without importing the whole app."""
        from src.config import load_config

        try:
            config = load_config(REPO_ROOT / "configs" / "default.json")
        except Exception as exc:  # pragma: no cover
            return {"model": None, "present": False, "error": f"config unreadable: {exc}"}
        det = config.get("detector", {}) or {}
        model = str(det.get("model", ""))
        path = (REPO_ROOT / model) if model and not Path(model).is_absolute() else Path(model)
        info: dict[str, Any] = {
            "backend": det.get("backend"),
            "model": model,
            "present": bool(model) and path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "imgsz": det.get("imgsz"),
            "confidence": det.get("confidence"),
            "stride": det.get("stride"),
            "provider": None,
        }
        if not info["present"]:
            info["error"] = (
                f"checkpoint not found at {path}. Place weights/yolo11n.onnx in the repository, "
                "or set TRAFFIC_YOLO_MODEL. Uploads will run degraded (no detections)."
            )
        elif info["bytes"] is None:
            info["error"] = "checkpoint could not be read"
        else:
            try:
                from src.perception.onnx_detector import OnnxDetector

                d = OnnxDetector(det)
                info["provider"] = (list(d.active_providers) or ["?"])[0]
                info["providers"] = list(d.active_providers or [])
            except Exception as exc:
                info["provider"] = None
                info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    def load(self):
        with self.lock:
            if self._solution is not None or self.import_error:
                return self._solution
            try:
                import solution  # noqa: WPS433 - deliberate late import

                solution.detect_events  # noqa: B018 - fail fast if the API changed
                self._solution = solution
            except Exception as exc:
                self.import_error = f"{type(exc).__name__}: {exc}"
            return self._solution


PIPELINE = Pipeline()


# ------------------------------------------------------------------ jobs
class Job:
    def __init__(self, job_id: str, filename: str, data: bytes) -> None:
        self.id = job_id
        self.filename = filename
        self.size = len(data)
        self.state = "queued"          # queued | running | done | error | cancelled
        self.stage = "queued"
        self.progress = 0.0
        self.detail = ""
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.degraded_reason: str | None = None
        self.cancel = threading.Event()
        self.created = time.time()
        self.path: Path | None = None
        self.lock = threading.Lock()

    def update(self, stage: str, progress: float, detail: str = "") -> None:
        with self.lock:
            self.stage = stage
            self.progress = max(0.0, min(1.0, float(progress)))
            if detail:
                self.detail = detail

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            out = {
                "job_id": self.id,
                "state": self.state,
                "stage": self.stage,
                "progress": round(self.progress, 3),
                "detail": self.detail,
                "filename": self.filename,
                "bytes": self.size,
            }
            if self.state == "done" and self.result is not None:
                out["result"] = self.result
            if self.error:
                out["error"] = self.error
            if self.degraded_reason:
                out["degraded"] = True
                out["degraded_reason"] = self.degraded_reason
            return out


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
RUN_SLOTS = threading.Semaphore(MAX_CONCURRENT)


# ------------------------------------------------------------------ upload
_BOUNDARY = re.compile(rb'boundary="?([^";]+)"?')


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Minimal multipart/form-data reader.

    ``cgi.FieldStorage`` was removed in Python 3.13, and pulling in a web
    framework just to read one file field is not worth the dependency. This
    handles the shape a browser actually sends for
    ``<input type=file>`` + ``FormData``.
    """
    m = _BOUNDARY.search(content_type.encode() if isinstance(content_type, str) else content_type)
    if not m:
        raise ValueError("multipart request without a boundary")
    boundary = b"--" + m.group(1)
    parts = body.split(boundary)
    for part in parts:
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        part = part.lstrip(b"\r\n")
        head, _, payload = part.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = head.decode("latin-1", "replace")
        disp = re.search(r'name="([^"]*)"', headers)
        fname = re.search(r'filename="([^"]*)"', headers)
        if not disp or disp.group(1) != "file":
            continue
        payload = payload.rstrip(b"\r\n")
        if payload.endswith(b"--"):
            payload = payload[:-2].rstrip(b"\r\n")
        return (fname.group(1) if fname else "upload.mp4"), payload
    raise ValueError("no 'file' field in the multipart body")


# ------------------------------------------------------------------ worker
def _open_cv():
    import cv2  # noqa: WPS433

    return cv2


def _container_info(cv2, path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("the uploaded file could not be opened as a video")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 25.0), 5),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    if not info["frames"]:
        # some containers lie; count by decoding (cheap enough at 1080p)
        cap = cv2.VideoCapture(str(path))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        info["frames"] = n
    info["duration_sec"] = round(info["frames"] / info["fps"], 3) if info["fps"] else 0.0
    return info


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _scene_info(video_name: str) -> dict[str, Any]:
    """Mirror ``PartAPipeline._scene_for_video`` exactly.

    The order matters and is easy to get wrong: an explicit ``scene.path`` in
    the active config wins over the per-video filename, which wins over the
    generic default. Reporting the wrong file here would tell a judge the demo
    runs without geometry when it does not.
    """
    config_error: str | None = None
    configured = ""
    try:
        from src.config import load_config

        configured = str((load_config(REPO_ROOT / "configs" / "default.json") or {})
                         .get("scene", {}).get("path", "") or "")
    except Exception as exc:
        config_error = str(exc)

    candidates: list[Path] = []
    if configured:
        p = Path(configured)
        candidates.append(p if p.is_absolute() else REPO_ROOT / p)
    env_path = os.getenv("TRAFFIC_SCENE_CONFIG")
    if env_path:
        p = Path(env_path)
        candidates.append(p if p.is_absolute() else REPO_ROOT / p)
    candidates.extend([
        REPO_ROOT / "configs" / "scenes" / f"{Path(video_name).stem}.json",
        REPO_ROOT / "configs" / "scenes" / "default.json",
    ])

    selected: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            selected = candidate
            break
    lookup = [_rel(p) for p in candidates]
    if selected is None:
        return {"path": None, "per_camera_calibrated": False,
                "lookup_order": lookup, "error": config_error}

    try:
        data = json.loads(selected.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": _rel(selected), "error": str(exc), "lookup_order": lookup}

    if configured and selected == candidates[0]:
        via = "config scene.path"
    elif env_path and selected == candidates[1]:
        via = "TRAFFIC_SCENE_CONFIG"
    elif selected.stem == Path(video_name).stem:
        via = "per-video filename"
    else:
        via = "configs/scenes/default.json"
    return {
        "path": _rel(selected),
        "scene_id": data.get("scene_id"),
        "via": via,
        "lanes": len(data.get("lanes", [])),
        "stop_lines": len(data.get("stop_lines", [])),
        "solid_lines": len(data.get("solid_lines", [])),
        "crosswalks": len(data.get("crosswalks", [])),
        "traffic_lights": len(data.get("traffic_lights", [])),
        "road_polygons": len(data.get("road_polygons", [])) or (1 if data.get("road_polygon") else 0),
        "has_homography": bool(data.get("homography")),
        "per_camera_calibrated": selected.stem == Path(video_name).stem,
        "lookup_order": lookup,
    }




def _annotated_frames(cv2, path: Path, events: list[list], duration: float,
                      curve: list[list[float]], count: int) -> list[dict[str, Any]]:
    """Render a handful of frames with the detector's boxes drawn on them."""
    try:
        from src.config import load_config
        from src.perception.detector import build_detector
    except Exception:
        return []
    try:
        detector = build_detector((load_config(REPO_ROOT / "configs" / "default.json") or {}).get("detector", {}))
    except Exception:
        return []
    if type(detector).__name__ == "NullDetector":
        return []

    # prefer the middle of the longest segment per class, then spread out
    by_label: dict[str, list[list]] = {}
    for ev in events:
        by_label.setdefault(ev[2], []).append(ev)
    picks: list[tuple[float, list[str]]] = []
    for label, segs in sorted(by_label.items(), key=lambda kv: -max(e[1] - e[0] for e in kv[1])):
        best = max(segs, key=lambda e: e[1] - e[0])
        picks.append(((best[0] + best[1]) / 2.0, [label]))
        if len(picks) >= count:
            break
    step = max(1e-6, duration / max(1, count))
    i = 1
    while len(picks) < count and i * step < duration:
        picks.append((round(i * step, 2), []))
        i += 1

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    out: list[dict[str, Any]] = []
    for t, forced in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
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
        active = sorted({ev[2] for ev in events if float(ev[0]) - 1e-6 <= t <= float(ev[1]) + 1e-6} | set(forced))
        img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            name = det.class_name.lower()
            color = (60, 220, 60) if name in {"person", "pedestrian"} else (0, 200, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            tag = f"{det.class_name} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(14, y1 - 4)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (20, 20, 20), -1)
            cv2.putText(img, tag, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        img[0:34, :] = (18, 18, 18)
        cv2.putText(img, f"t={t:7.2f}s  risk={risk:.3f}  detections={len(detections)}",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
        if not ok:
            continue
        b64 = _b64(buf.tobytes())
        out.append({
            "t": round(float(t), 2),
            "detections": len(detections),
            "risk": round(float(risk), 4),
            "active_labels": active,
            # data URL keeps the response self-contained: no temp-file lifetime
            # problems, and nothing to clean up if the worker dies
            "src": f"data:image/jpeg;base64,{b64}",
            "bytes": len(buf),
        })
    cap.release()
    return out


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


def run_job(job: Job) -> None:
    cv2 = None
    tmpdir = None
    with RUN_SLOTS:
        if job.cancel.is_set():
            job.state = "cancelled"
            return
        try:
            job.state = "running"
            job.update("importing solution.py", 0.02)
            solution = PIPELINE.load()
            if solution is None:
                job.update("importing solution.py", 0.05)
                raise RuntimeError(
                    "could not import solution.py from the repository root: "
                    + (PIPELINE.import_error or "unknown error")
                )
            cv2 = _open_cv()

            tmpdir = Path(tempfile.mkdtemp(prefix="demo_"))
            job.path = tmpdir / (Path(job.filename).name or "upload.mp4")
            job.path.write_bytes(job._payload)  # type: ignore[attr-defined]
            job.update("reading container", 0.08)
            container = _container_info(cv2, job.path)
            duration = container["duration_sec"]
            if duration > MAX_SECONDS + 0.5:
                raise ValueError(
                    f"clip is {duration:.1f} s; this demo accepts at most {MAX_SECONDS:.0f} s. "
                    f"Trim it: ffmpeg -i in.mp4 -t {int(MAX_SECONDS)} -c copy out.mp4"
                )
            job.update("analysing", 0.12, f"{container['frames']} frames @ {container['fps']} fps")

            # ---- Part A, with a progress thread ------------------------- #
            holder: dict[str, Any] = {}

            def part_a() -> None:
                t0 = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    holder["events"] = solution.detect_events(str(job.path))
                holder["seconds"] = time.perf_counter() - t0
                holder["warnings"] = list({str(w.message) for w in caught})

            worker = threading.Thread(target=part_a, daemon=True)
            worker.start()
            start = time.perf_counter()
            # A real progress bar is impossible: detect_events is a single
            # blocking call. We interpolate against elapsed/expected so the
            # user sees movement, and clamp below 82% until it really returns.
            while worker.is_alive():
                if job.cancel.is_set():
                    break
                elapsed = time.perf_counter() - start
                expected = max(8.0, duration * 1.1)
                frac = 0.12 + 0.70 * min(1.0, elapsed / expected)
                job.update("part A — detector + tracker + rules", frac,
                           f"{elapsed:0.0f}s elapsed" +
                           (f", {len(holder.get('events', []))} events" if holder.get("events") else ""))
                worker.join(1.0)
            worker.join()
            if job.cancel.is_set():
                job.state = "cancelled"
                return
            if "events" not in holder:
                raise RuntimeError("Part A did not return")

            events = holder["events"] or []
            part_a_sec = holder["seconds"]
            job.update("part A done", 0.82, f"{len(events)} segments in {part_a_sec:.0f}s")

            # ---- Part B, replayed one frame at a time ------------------- #
            curve: list[list[float]] = []
            part_b_sec = None
            estimator = None
            try:
                from evaluate import MERGE_GAP, THETA  # the metric's own constants

                estimator = solution.RiskEstimator()
            except Exception as exc:
                holder.setdefault("warnings", []).append(f"Part B unavailable: {exc}")

            if estimator is not None:
                estimator.reset({
                    "video_id": job.path.name,
                    "fps": container["fps"],
                    "width": container["width"],
                    "height": container["height"],
                    "n_frames": container["frames"],
                })
                cap = cv2.VideoCapture(str(job.path))
                fps = container["fps"] or 25.0
                total_frames = max(1, container["frames"])
                t0 = time.perf_counter()
                index = 0
                while True:
                    if job.cancel.is_set():
                        cap.release()
                        job.state = "cancelled"
                        return
                    ok, frame = cap.read()
                    if not ok:
                        break
                    curve.append([round(index / fps, 4), round(float(estimator.step(frame, index / fps)), 5)])
                    index += 1
                    if index % 30 == 0:
                        # Part B builds its own detector/tracker stack, so it is
                        # a real second pass, not a cache read. Budget it
                        # separately or the bar sits at 95% for minutes.
                        frac = 0.82 + 0.12 * min(1.0, index / total_frames)
                        job.update("part B — second causal pass", frac,
                                   f"{index}/{container['frames']} frames "
                                   f"({time.perf_counter() - t0:.0f}s elapsed)")
                cap.release()
                part_b_sec = time.perf_counter() - t0

            job.update("drawing frames", 0.96)
            frames = _annotated_frames(cv2, job.path, events, duration, curve, ANNOTATED_FRAMES)

            # ---- summarise ------------------------------------------------ #
            counts = Counter(ev[2] for ev in events)
            risk_block: dict[str, Any] = {"threshold": 0.5, "merge_gap_sec": 2.0}
            if curve:
                scores = [s for _, s in curve]
                above = sum(1 for s in scores if s >= 0.5)
                risk_block.update({
                    "points_total": len(curve),
                    "mean": round(sum(scores) / len(scores), 4),
                    "max": round(max(scores), 4),
                    "min": round(min(scores), 4),
                    "share_at_or_above_threshold": round(above / len(curve), 4),
                    "alarm_count": sum(
                        1 for i, s in enumerate(scores)
                        if s >= 0.5 and (i == 0 or scores[i - 1] < 0.5)
                    ),
                    "curve": curve[::RISK_STRIDE],
                })

            degraded = PIPELINE.detector.get("present") is False
            job.result = {
                "filename": job.filename,
                "container": container,
                "events": [
                    {"start": round(float(e[0]), 3), "end": round(float(e[1]), 3),
                     "label": e[2], "duration": round(float(e[1]) - float(e[0]), 3)}
                    for e in events
                ],
                "counts": dict(counts),
                "risk": risk_block,
                "annotated_frames": frames,
                "runtime": {"part_a_sec": round(part_a_sec, 1),
                            "part_b_sec": round(part_b_sec, 1) if part_b_sec else None},
                "scene": _scene_info(job.filename),
                "warnings": sorted(holder.get("warnings", [])),
                "degraded": degraded,
                "degraded_reason": (
                    "the detector checkpoint is missing, so there are no tracks and therefore no events"
                    if degraded else None
                ),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            job.state = "done"
            job.update("done", 1.0, f"{len(events)} segments")
        except Exception as exc:
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.update("failed", job.progress, job.error)
            traceback.print_exc()
        finally:
            if job.path and job.path.exists():
                try:
                    job.path.unlink()
                except OSError:
                    pass
            if tmpdir and tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)


def _spawn(job: Job) -> None:
    # The job stays in JOBS until JOB_TTL_SEC so the client can still poll it
    # after completion; reap_jobs() prunes it later.
    def target() -> None:
        try:
            run_job(job)
        except Exception:  # pragma: no cover - run_job already guards
            traceback.print_exc()
            job.state = "error"
            job.error = "unhandled worker error"

    threading.Thread(target=target, daemon=True).start()


def reap_jobs() -> None:
    now = time.time()
    with JOBS_LOCK:
        for jid in [k for k, v in JOBS.items() if now - v.created > JOB_TTL_SEC]:
            JOBS.pop(jid, None)


# ------------------------------------------------------------------ static
class Handler(BaseHTTPRequestHandler):
    server_version = "TrafficDemo/1.0"
    protocol_version = "HTTP/1.1"
    root = WEBSITE
    static_only = False

    # -------- helpers ---------------------------------------------------- #
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
        if os.environ.get("DEMO_VERBOSE"):
            sys.stderr.write("[demo] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message, "status": code})

    def _resolve(self, path: str) -> Path | None:
        rel = path.split("?", 1)[0].split("#", 1)[0]
        rel = rel.lstrip("/")
        if not rel:
            rel = "index.html"
        target = (self.root / rel).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError:
            return None
        if target.is_dir():
            target = target / "index.html"
        return target if target.is_file() else None

    def _serve_file(self, target: Path) -> None:
        ext = target.suffix.lower()
        ctype = _ALLOWED.get(ext.lstrip(".")) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        size = target.stat().st_size
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                if self.command != "HEAD":
                    with target.open("rb") as fh:
                        fh.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = fh.read(min(262144, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control",
                         "public, max-age=3600" if ext in {".webm", ".mp4", ".jpg", ".png"} else "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            with target.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile, 262144)

    # -------- verbs ------------------------------------------------------ #
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self._health()
        if path.startswith("/api/jobs/"):
            jid = path[len("/api/jobs/"):].strip("/")
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._error(404, "unknown or expired job id")
            return self._json(200, job.snapshot())
        target = self._resolve(path)
        if target is None:
            body = b"404 - not found. The site root is " + str(self.root).encode() + b"\n"
            return self._send(404, body, "text/plain; charset=utf-8")
        try:
            self._serve_file(target)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/jobs":
            return self._create_job()
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            jid = path[len("/api/jobs/"):-len("/cancel")].strip("/")
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._error(404, "unknown job id")
            job.cancel.set()
            return self._json(200, {"job_id": jid, "cancel_requested": True})
        return self._error(404, "no such endpoint")

    # -------- api -------------------------------------------------------- #
    def _health(self) -> None:
        reap_jobs()
        det = PIPELINE.probe_detector()
        PIPELINE.detector = det
        self._json(200, {
            "ok": True,
            "static_only": self.static_only,
            "python": sys.version.split()[0],
            "repo_root": str(REPO_ROOT),
            "website_root": str(WEBSITE),
            "detector": {
                "model": det.get("model"),
                "present": det.get("present"),
                "bytes": det.get("bytes"),
                "provider": det.get("provider"),
                "providers": det.get("providers"),
                "backend": det.get("backend"),
                "stride": det.get("stride"),
                # surfaced rather than swallowed: a null provider with no
                # reason is the least useful thing a health check can return
                "error": det.get("error"),
                "error_or_none": det.get("error") or det.get("load_error"),
            },
            "limits": {"max_seconds": MAX_SECONDS, "max_bytes": MAX_BYTES,
                       "accepted": sorted(VIDEO_EXTS)},
            "concurrent_jobs": MAX_CONCURRENT,
            "annotated_frames": ANNOTATED_FRAMES,
            "recent_part_a_realtime_factor": self._realtime_factor(),
        })

    def _realtime_factor(self) -> float | None:
        """Measured Part A seconds-per-video-second, from completed jobs.

        The page shows this instead of a guess, because how long a judge will
        wait is the one thing they will notice first.
        """
        with JOBS_LOCK:
            factors = [
                (j.result or {}).get("runtime", {}).get("part_a_sec") /
                max(1e-6, (j.result or {}).get("container", {}).get("duration_sec") or 1.0)
                for j in JOBS.values()
                if j.state == "done" and (j.result or {}).get("runtime", {}).get("part_a_sec")
            ]
        if not factors:
            return None
        return round(sum(factors) / len(factors), 3)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_BYTES + 4096:
            raise ValueError(
                f"upload is {length / 1048576:.1f} MB; the limit is {MAX_BYTES / 1048576:.0f} MB"
            )
        return self.rfile.read(length)

    def _create_job(self) -> None:
        if self.static_only:
            return self._error(503, "this instance was started with --static-only; upload is disabled")
        try:
            raw = self._read_body()
        except ValueError as exc:
            return self._error(413, str(exc))
        ctype = self.headers.get("Content-Type", "")
        filename = self.headers.get("X-Filename", "upload.mp4")
        try:
            if ctype.startswith("multipart/form-data"):
                filename, payload = parse_multipart(raw, ctype)
            else:
                payload = raw
        except ValueError as exc:
            return self._error(400, f"could not read the upload: {exc}")
        if not payload:
            return self._error(400, "empty upload")
        if len(payload) > MAX_BYTES:
            return self._error(413, f"{len(payload) / 1048576:.1f} MB exceeds the "
                                    f"{MAX_BYTES / 1048576:.0f} MB limit")
        ext = Path(filename).suffix.lower()
        if ext and ext not in VIDEO_EXTS:
            return self._error(415, f"{ext} is not accepted; use one of "
                                    + ", ".join(sorted(VIDEO_EXTS)))
        if not ext:
            filename += ".mp4"
        job = Job(uuid.uuid4().hex[:16], Path(filename).name, payload)
        job._payload = payload  # type: ignore[attr-defined]
        with JOBS_LOCK:
            JOBS[job.id] = job
        _spawn(job)
        self._json(202, {"job_id": job.id, "state": job.state, "max_seconds": MAX_SECONDS})


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--static-only", action="store_true",
                    help="serve the site but refuse uploads (for a quick preview)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        os.environ["DEMO_VERBOSE"] = "1"

    Handler.static_only = args.static_only

    det = PIPELINE.probe_detector()
    PIPELINE.detector = det

    def say(msg: str) -> None:
        print(msg, flush=True)

    say(f"[demo] website root : {WEBSITE}")
    say(f"[demo] repository   : {REPO_ROOT}")
    say(f"[demo] detector     : {det.get('model')} present={det.get('present')} "
        f"provider={det.get('provider')}")
    if det.get("error"):
        say(f"[demo] WARNING  : {det['error']}")
    elif not det.get("present"):
        say("[demo] WARNING  : no checkpoint. Uploads will still run and will report "
            "degraded=true with real metadata and no events.")
    try:
        import numpy, cv2  # noqa: F401
        say(f"[demo] deps        : numpy {numpy.__version__}, opencv {cv2.__version__}")
    except Exception as exc:
        say(f"[demo] WARNING  : {exc} — the site will serve but uploads cannot run.")
    say(f"[demo] limits      : {MAX_SECONDS:.0f}s / {MAX_BYTES / 1048576:.0f} MB, "
        f"{MAX_CONCURRENT} job at a time")
    say(f"[demo] listening on http://{args.host}:{args.port}/")

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        address_family = socket.AF_INET

    try:
        Server((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
