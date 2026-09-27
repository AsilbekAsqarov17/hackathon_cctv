"""Provision a working CUDA 12 ONNX Runtime GPU stack for local development.

Why this script exists
----------------------
`onnxruntime-gpu` >= 1.27 is built against CUDA 13, but NVIDIA does not publish
`nvidia-cublas-cu13` / `nvidia-cuda-runtime-cu13` wheels for Windows. ONNX
Runtime 1.21.x-1.26.x are the CUDA 12.8 + cuDNN 9 builds, so we pin one of those
and pair it with the `nvidia-*-cu12` runtime packages.

The default PyPI download path is extremely slow from this machine, so each
wheel is fetched with parallel HTTP range requests. Every part and wheel is
verified by exact byte count, and the whole run is resumable.

This is a development helper. It is not imported by the competition runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

WHEELS = [
    (
        "onnxruntime_gpu-1.26.0-cp313-cp313-win_amd64.whl",
        "https://files.pythonhosted.org/packages/67/3f/59f1777a394625ecc9a85636de57dc47c25dbb5f888da050f1463955a0ce/onnxruntime_gpu-1.26.0-cp313-cp313-win_amd64.whl",
        226_548_083,
        "f370ba72cea116a19497961332daa1fb",
    ),
    (
        "nvidia_cudnn_cu12-9.26.0.51-py3-none-win_amd64.whl",
        "https://files.pythonhosted.org/packages/c5/ee/baebebf270df5a57830e40879b4016de47ca43961095cf18b7749452150f/nvidia_cudnn_cu12-9.26.0.51-py3-none-win_amd64.whl",
        746_465_599,
        "3121745e82d2ad244e71f395bb502a0d",
    ),
    (
        "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        "https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
        553_162_896,
        "9f5b07778be475374448f5479337e998",
    ),
    (
        "nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
        "https://files.pythonhosted.org/packages/52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
        76_408_187,
        "cb2dbc1b9896b04faf6f9ddd81ef7d71",
    ),
    (
        "nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        3_591_604,
        "77cf381ad55288e5790c91a27d45cfb8",
    ),
]

TOTAL_BYTES = sum(size for _, _, size, _ in WHEELS)


class Progress:
    """Best-effort total throughput reporter across all in-flight parts."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last = 0
        self._last_time = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._last = self._downloaded()
        self._last_time = time.time()
        self._thread.start()

    def _downloaded(self) -> int:
        total = 0
        for path in self.cache.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def _run(self) -> None:
        while not self._stop.wait(20):
            with self._lock:
                current = self._downloaded()
                now = time.time()
                elapsed = max(now - self._last_time, 1e-6)
                rate = (current - self._last) / elapsed
                self._last = current
                self._last_time = now
                pct = 100.0 * current / TOTAL_BYTES
                remaining = TOTAL_BYTES - current
                eta = remaining / rate / 60 if rate > 1024 else float("inf")
                eta_text = f"{eta:.0f} min" if eta != float("inf") else "unknown"
                print(
                    f"  [progress] {current/1e6:7.1f}/{TOTAL_BYTES/1e6:.0f} MB "
                    f"({pct:5.1f}%)  {rate/1024:6.0f} KiB/s  eta {eta_text}",
                    flush=True,
                )

    def stop(self) -> None:
        self._stop.set()


def download_part(url: str, start: int, end: int, part: Path, retries: int = 40) -> int:
    """Fetch one byte range, resuming a partial part across retries.

    Connections on this machine drop near the end of large ranges, so restarting
    a part from zero almost never converges. We instead track how much of the
    part is already in the temp file and request only the missing tail.
    """
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return expected
    tmp = part.with_suffix(part.suffix + ".part")
    if tmp.exists() and tmp.stat().st_size > expected:
        tmp.unlink()

    last: Exception | None = None
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have == expected:
            tmp.replace(part)
            return expected
        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                if have and response.status_code != 206:
                    raise RuntimeError(f"resume needs HTTP 206, got {response.status_code}")
                if not have and response.status_code not in (200, 206):
                    raise RuntimeError(f"expected HTTP 200/206, got {response.status_code}")
                with tmp.open("ab" if have else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size != expected:
                raise RuntimeError(f"short part {tmp.stat().st_size}/{expected}")
            tmp.replace(part)
            return expected
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last = exc
            time.sleep(min(15, 1 + attempt // 2))
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts: {last}")


def fetch_wheel(name: str, url: str, size: int, md5: str, cache: Path, parts: int) -> Path:
    target = cache / name
    if target.exists() and target.stat().st_size == size:
        if md5_file(target) == md5:
            print(f"  cached  {name}", flush=True)
            return target
        print(f"  STALE   {name} (md5 mismatch, re-fetching)", flush=True)
        target.unlink()
    elif target.exists():
        print(f"  STALE   {name} (wrong size, re-fetching)", flush=True)
        target.unlink()
    part_dir = cache / (name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = (size + parts - 1) // parts
    jobs = []
    for index in range(parts):
        start = index * chunk
        end = min(size - 1, start + chunk - 1)
        if start <= end:
            jobs.append((start, end, part_dir / f"{index:04d}.part"))
    print(f"  fetch   {name} ({size/1e6:.1f} MB in {len(jobs)} parts)", flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(download_part, url, s, e, p) for s, e, p in jobs]
        for future in futures:
            future.result()
    tmp = target.with_suffix(target.suffix + ".assembling")
    with tmp.open("wb") as out:
        for _, _, part in jobs:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out, 1024 * 1024)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{name}: assembled {tmp.stat().st_size} bytes, expected {size}")
    actual = md5_file(tmp)
    if actual != md5:
        tmp.unlink()
        shutil.rmtree(part_dir, ignore_errors=True)
        raise RuntimeError(f"{name}: md5 {actual} != expected {md5}")
    tmp.replace(target)
    shutil.rmtree(part_dir, ignore_errors=True)
    return target


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=r"C:\Users\Oliya\AppData\Local\Temp\opencode\ort_wheels",
    )
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"cache: {cache}")
    print(f"total: {TOTAL_BYTES/1e6:.0f} MB across {len(WHEELS)} wheels\n", flush=True)

    progress = Progress(cache)
    progress.start()
    started = time.time()
    try:
        for name, url, size, md5 in WHEELS:
            fetch_wheel(name, url, size, md5, cache, args.parts)
    finally:
        progress.stop()
    print(f"\ndownload phase finished in {(time.time()-started)/60:.1f} min", flush=True)

    if args.download_only:
        return 0

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(cache),
            "--force-reinstall",
            "--no-deps",
            "onnxruntime-gpu==1.26.0",
            "nvidia-cudnn-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cuda-nvrtc-cu12",
        ]
    )
    run([sys.executable, str(Path(__file__).with_name("onnx_gpu_smoke.py"))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
