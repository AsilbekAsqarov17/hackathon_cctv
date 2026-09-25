"""Extract a development frame set for detector fine-tuning.

With one source video, train/validation frames are a temporal development split,
not an independent test set. Replace them with multiple videos before drawing
final conclusions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", default="data/traffic_coco")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=0, help="optional max width; 0 keeps source size")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps else 0.0
    start = max(0.0, args.start)
    end = duration if args.end is None else min(duration, args.end)
    if end <= start:
        raise SystemExit("empty time range")
    count = max(1, args.num_frames)
    times = [start + (end - start) * i / max(1, count - 1) for i in range(count)]
    root = Path(args.output)
    image_dir = root / "images" / "development"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_indices = [int(round(timestamp * fps)) for timestamp in times]
    targets = {frame_index: (sample_index, target_indices[sample_index]) for sample_index, frame_index in enumerate(target_indices)}
    manifest = []
    frame_index = 0
    last_target = target_indices[-1] if target_indices else -1
    while frame_index <= last_target:
        ok = cap.grab()
        if not ok:
            break
        if frame_index in targets:
            sample_index, source_frame = targets[frame_index]
            ok, frame = cap.retrieve()
            if ok:
                timestamp = times[sample_index]
                if args.image_size and frame.shape[1] > args.image_size:
                    scale = args.image_size / frame.shape[1]
                    frame = cv2.resize(frame, (args.image_size, round(frame.shape[0] * scale)))
                name = f"frame_{sample_index:05d}_{timestamp:09.3f}.jpg"
                cv2.imwrite(str(image_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({"file": f"images/development/{name}", "source_video": Path(args.video).name, "source_frame": source_frame, "timestamp": timestamp})
        frame_index += 1
    cap.release()
    (root / "frame_manifest.json").write_text(json.dumps({"source": str(args.video), "fps": fps, "frames": manifest}, indent=2), encoding="utf-8")
    print(f"saved {len(manifest)} frames to {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
