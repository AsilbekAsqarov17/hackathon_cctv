"""Download a large wheel with parallel HTTP range requests.

Used only for local development dependency setup when a single PyPI connection
stalls. It is not part of the competition runtime.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download_part(url: str, start: int, end: int, part: Path, retries: int = 8) -> int:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return expected
    for attempt in range(retries):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned {response.status_code}")
                tmp = part.with_suffix(part.suffix + ".part")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                if tmp.stat().st_size != expected:
                    raise RuntimeError(f"short part {tmp.stat().st_size}/{expected}")
                tmp.replace(part)
                return expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"failed range {start}-{end}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("size", type=int)
    parser.add_argument("--parts", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    part_dir = output.with_suffix(output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = (args.size + args.parts - 1) // args.parts
    ranges = []
    for index in range(args.parts):
        start = index * chunk
        end = min(args.size - 1, start + chunk - 1)
        if start <= end:
            ranges.append((index, start, end, part_dir / f"{index:04d}.part"))
    with ThreadPoolExecutor(max_workers=args.parts) as pool:
        futures = [pool.submit(download_part, args.url, start, end, path) for _, start, end, path in ranges]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"part {index}/{len(futures)} complete", flush=True)
    with output.open("wb") as target:
        for _, start, end, part in ranges:
            with part.open("rb") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
    if output.stat().st_size != args.size:
        raise RuntimeError(f"size mismatch {output.stat().st_size}/{args.size}")
    for _, _, _, part in ranges:
        part.unlink(missing_ok=True)
    part_dir.rmdir()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
