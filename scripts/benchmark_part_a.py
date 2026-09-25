"""Run Part A on one or more videos and print simple timing diagnostics."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.part_a import run_part_a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    for value in args.videos:
        path = Path(value)
        videos = sorted(path.glob("*.mp4")) if path.is_dir() else [path]
        for video in videos:
            start = time.perf_counter()
            events = run_part_a(str(video), args.config)
            print(f"{video}: {len(events)} events in {time.perf_counter() - start:.2f}s")
            for event in events:
                print(" ", event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
